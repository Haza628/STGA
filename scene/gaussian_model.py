#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import copy
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.ghz_functions import *
from network.MLP import MLP
from network.PositionalEmbedding import get_embedder

#WDD 5-14 
#处理旋转的函数
from utils.ghz_functions import rotmat_to_unitquat,quat_xyzw_to_wxyz,quat_wxyz_to_xyzw,quat_product

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)
def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN
class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        # self.scaling_activation = torch.sigmoid 
        # self.scaling_inverse_activation = inverse_sigmoid
        
        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        pos_freq = 4
        color_offset_net_layer = [90, 256, 256, 48]
        xyz_offset_net_layer = [90, 256, 256, 3]

        self.embed, _ = get_embedder(pos_freq)
        self.color_offset_net = MLP(color_offset_net_layer).cuda()
        self.xyz_offset_net = MLP(xyz_offset_net_layer).cuda()


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        
        #WDD 5-1
        #===================
        self.indices = torch.empty(0)
        #===================
        #WDD 5-6
        #创建keyframe_meshes属性
        #这里他是个字典，每一帧的数据再单独转换为tensor
        #===================
        self.keyframe_meshes = {}
        self.current_keyframe = 0
        #===================

        # ghz
        self.canonical_face_area = torch.empty(0)
        self._scaling_log =  torch.empty(0)
        self.dynamic = True
        self.training_indices = {}

        self._xyz_frozen = torch.empty(0).cuda()
        self._features_dc_frozen = torch.empty(0).cuda()
        self._features_rest_frozen = torch.empty(0).cuda()
        self._scaling_frozen = torch.empty(0).cuda()
        self._rotation_frozen = torch.empty(0).cuda()
        self._opacity_frozen = torch.empty(0).cuda()
        self._indices_frozen = torch.empty(0).cuda()

        self._xyz_training = torch.empty(0)
        self._features_dc_training = torch.empty(0)
        self._features_rest_training = torch.empty(0)
        self._scaling_training = torch.empty(0)
        self._rotation_training = torch.empty(0)
        self._opacity_training = torch.empty(0)
        self._indices_training = torch.empty(0)

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            #WDD 5-1
            #================
            self.indices,
            #================
            # ghz
            self.canonical_face_area,
            self._scaling_log,
            self.dynamic,
            self.training_indices,

            self._xyz_frozen,
            self._features_dc_frozen,
            self._features_rest_frozen,
            self._scaling_frozen,
            self._rotation_frozen,
            self._opacity_frozen,
            self._indices_frozen,

            self._xyz_training,
            self._features_dc_training,
            self._features_rest_training,
            self._scaling_training,
            self._rotation_training,
            self._opacity_training,
            self._indices_training,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 

        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        #WDD 5-1
        #================
        self.indices,
        #================
        # ghz
        self.canonical_face_area,
        self._scaling_log,
        self.dynamic,
        self.training_indices,

        self._xyz_frozen,
        self._features_dc_frozen,
        self._features_rest_frozen,
        self._scaling_frozen,
        self._rotation_frozen,
        self._opacity_frozen,
        self._indices_frozen,

        self._xyz_training,
        self._features_dc_training,
        self._features_rest_training,
        self._scaling_training,
        self._rotation_training,
        self._opacity_training,
        self._indices_training,
        
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        
    # @property
    # def get_k(self):
    #     mesh_data=self.keyframe_meshes[self.current_keyframe]
    #     indices=self.indices

    #     # 解析 indices 中的各个部分         
    #     v1=mesh_data[indices,0,:].squeeze(1)
    #     v2=mesh_data[indices,1,:].squeeze(1)
    #     v3=mesh_data[indices,2,:].squeeze(1)

    #     area = 0.5 * torch.norm(torch.cross(v2 - v1, v3 - v1), dim=1)[:, None]
              
    #     k = torch.sqrt(area / self.canonical_face_area[indices].squeeze(-1))

    #     return k
       
    
    @property
    def get_scaling(self):
        # ghz

        # # self._scaling_log = self._scaling
        # mesh_data=self.keyframe_meshes[self.current_keyframe]
        # indices=self.indices

        # # 解析 indices 中的各个部分         
        # v1=mesh_data[indices,0,:].squeeze(1)
        # v2=mesh_data[indices,1,:].squeeze(1)
        # v3=mesh_data[indices,2,:].squeeze(1)

        # k = torch.sqrt(torch.norm(torch.cross(v2 - v1, v3 - v1), dim=1)[:, None]) 
        
        if not self.dynamic:
            # self._scaling_log = self._scaling
            ret  = self.scaling_activation(self._scaling)# * k
            
        else:

            # training_indices = self.training_indices[self.current_keyframe]
            # gaussian_indices = torch.isin(self.indices, training_indices).squeeze()

            # ret  = self.scaling_activation(torch.cat((self._scaling_training, self._scaling_frozen)))                   # * torch.cat((k[gaussian_indices], k[~gaussian_indices]))
            # ret = ret[: 2636]
            
            ret  = self.scaling_activation(self._scaling_training)

        self._scaling_log = torch.log(ret)


        return ret

        
    @property
    def get_all_rotation(self):
        mesh_data=self.keyframe_meshes[self.current_keyframe]
     
        # indices=torch.cat((self._indices_training, self._indices_frozen)).to(torch.long)
        indices=self.indices
            
        # 解析 indices 中的各个部分         
        v1=mesh_data[indices,0,:].squeeze(1)
        v2=mesh_data[indices,1,:].squeeze(1)
        v3=mesh_data[indices,2,:].squeeze(1)
        
        # 计算局部坐标系的原点，即  
        o = (v1 + v2 + v3) / 3
        
        # 计算局部坐标系的基向量
        x = (v1 - o).float()
        x /= torch.norm(x, dim=1, keepdim=True)
        
        n = torch.cross(v2 - v1, v3 - v1, dim=1).float()
        n /= torch.norm(n, dim=1, keepdim=True)

        y = torch.cross(n, x, dim=1)
        y /= torch.norm(y, dim=1, keepdim=True)

        # 组成变换矩阵，每个点一个矩阵
        transform_matrices = torch.stack([x, y, n], dim=-1)

        transform_rotation_q = self.rotation_activation(quat_xyzw_to_wxyz(rotmat_to_unitquat(transform_matrices)))

        indices_training = self.training_indices[self.current_keyframe]
        gaussian_indices = torch.isin(indices, indices_training).squeeze()
        
        rot = self.rotation_activation(torch.cat((self._rotation_training, self._rotation_frozen)))
        world_rotation_q = quat_xyzw_to_wxyz(quat_product(quat_wxyz_to_xyzw(torch.cat((transform_rotation_q[self.gaussian_indices], transform_rotation_q[self.non_gaussian_indices]))), quat_wxyz_to_xyzw(rot)))
        return world_rotation_q

    
    @property
    def get_rotation(self):
        # ghz

        mesh_data=self.keyframe_meshes[self.current_keyframe]
        if self.dynamic:
            # indices=torch.cat((self._indices_training, self._indices_frozen)).to(torch.long)
            indices=self._indices_training.to(torch.long)

        else:
            indices=self.indices

        # 解析 indices 中的各个部分         
        v1=mesh_data[indices,0,:].squeeze(1)
        v2=mesh_data[indices,1,:].squeeze(1)
        v3=mesh_data[indices,2,:].squeeze(1)
        
        # 计算局部坐标系的原点，即  
        o = (v1 + v2 + v3) / 3
        
        # 计算局部坐标系的基向量
        x = (v1 - o).float()
        x /= torch.norm(x, dim=1, keepdim=True)
        
        n = torch.cross(v2 - v1, v3 - v1, dim=1).float()
        n /= torch.norm(n, dim=1, keepdim=True)

        y = torch.cross(n, x, dim=1)
        y /= torch.norm(y, dim=1, keepdim=True)

        # 组成变换矩阵，每个点一个矩阵
        transform_matrices = torch.stack([x, y, n], dim=-1)

        transform_rotation_q = self.rotation_activation(quat_xyzw_to_wxyz(rotmat_to_unitquat(transform_matrices)))


        # rot = self.rotation_activation(self._rotation)
        # world_rotation_q = quat_xyzw_to_wxyz(quat_product(quat_wxyz_to_xyzw(transform_rotation_q), quat_wxyz_to_xyzw(rot)))
            
       
        if not self.dynamic:
            rot = self.rotation_activation(self._rotation)
            world_rotation_q = quat_xyzw_to_wxyz(quat_product(quat_wxyz_to_xyzw(transform_rotation_q), quat_wxyz_to_xyzw(rot)))
            
        else:

            rot = self.rotation_activation(self._rotation_training)
            world_rotation_q = quat_xyzw_to_wxyz(quat_product(quat_wxyz_to_xyzw(transform_rotation_q), quat_wxyz_to_xyzw(rot)))


            
        return world_rotation_q

    @property
    def get_all_xyz(self):
        mesh_data=self.keyframe_meshes[self.current_keyframe]
        # indices=torch.cat((self._indices_training, self._indices_frozen)).to(torch.long)
        indices = self.indices.to(torch.long)

        # 解析 indices 中的各个部分         
        v1=mesh_data[indices,0,:].squeeze(1)
        v2=mesh_data[indices,1,:].squeeze(1)
        v3=mesh_data[indices,2,:].squeeze(1)

        #v1 = self.indices[:, 4:7]
        #v2 = self.indices[:, 7:10]
        #v3 = self.indices[:, 10:13]
        
        # 计算局部坐标系的原点，即  
        o = (v1 + v2 + v3) / 3
        
        # 计算局部坐标系的基向量
        x = (v1 - o).float()
        x /= torch.norm(x, dim=1, keepdim=True)
        
        n = torch.cross(v2 - v1, v3 - v1, dim=1).float()
        n /= torch.norm(n, dim=1, keepdim=True)

        y = torch.cross(n, x, dim=1)
        y /= torch.norm(y, dim=1, keepdim=True)

        # 组成变换矩阵，每个点一个矩阵
        transform_matrices = torch.stack([x, y, n], dim=-1)
        
        # indices_training = self.training_indices[self.current_keyframe]
        # gaussian_indices = torch.isin(indices, indices_training).squeeze()
        world_points = torch.matmul(torch.cat((transform_matrices[self.gaussian_indices], transform_matrices[self.non_gaussian_indices])), (torch.cat((self._xyz_training, self._xyz_frozen))).unsqueeze(-1) ).squeeze(-1) + torch.cat((o[self.gaussian_indices], o[self.non_gaussian_indices]))

        return world_points
    

    @property
    def get_xyz(self):
        # ghz

        mesh_data=self.keyframe_meshes[self.current_keyframe]
        if self.dynamic:
            # indices=torch.cat((self._indices_training, self._indices_frozen)).to(torch.long)
            indices = self._indices_training.to(torch.long)

        else:
            indices=self.indices

        # 解析 indices 中的各个部分         
        v1=mesh_data[indices,0,:].squeeze(1)
        v2=mesh_data[indices,1,:].squeeze(1)
        v3=mesh_data[indices,2,:].squeeze(1)

        #v1 = self.indices[:, 4:7]
        #v2 = self.indices[:, 7:10]
        #v3 = self.indices[:, 10:13]
        
        # 计算局部坐标系的原点，即  
        o = (v1 + v2 + v3) / 3
        
        # 计算局部坐标系的基向量
        x = (v1 - o).float()
        x /= torch.norm(x, dim=1, keepdim=True)
        
        n = torch.cross(v2 - v1, v3 - v1, dim=1).float()
        n /= torch.norm(n, dim=1, keepdim=True)

        y = torch.cross(n, x, dim=1)
        y /= torch.norm(y, dim=1, keepdim=True)

        # 组成变换矩阵，每个点一个矩阵
        transform_matrices = torch.stack([x, y, n], dim=-1)


        # world_points = torch.matmul(transform_matrices, (self._xyz_training).unsqueeze(-1) ).squeeze(-1) + o
        if not self.dynamic:
            world_points = torch.matmul(transform_matrices, (self._xyz).unsqueeze(-1) ).squeeze(-1) + o
        else:
            world_points = torch.matmul(transform_matrices, (self._xyz_training).unsqueeze(-1) ).squeeze(-1) + o
 

        #     # # ghz 1029 offset网络
        #     # input_data = torch.cat([self.embed(world_points), self.smplx_pose.squeeze(0).repeat(world_points.shape[0], 1)], dim=-1).cuda()
        #     # offset = self.xyz_offset_net(input_data)
        #     # world_points = world_points + offset

        return world_points
        # return torch.cat((torch.cat((world_points1, world_points3 )), world_points2))

    
    @property
    def get_features(self):

        if not self.dynamic:
            features_dc = self._features_dc
            features_rest = self._features_rest
        else:
            # features_dc = torch.cat((self._features_dc_training, self._features_dc_frozen))
            # features_rest = torch.cat((self._features_rest_training, self._features_rest_frozen))
            # features_dc = features_dc[: 2636]
            # features_rest = features_rest[: 2636]

            features_dc = self._features_dc_training
            features_rest = self._features_rest_training

        return torch.cat((features_dc, features_rest), dim=1)
        # return torch.cat((self._features_dc, self._features_rest), dim=1)
    
    @property
    def get_opacity(self):

        if not self.dynamic:
            return self.opacity_activation(self._opacity)
        else:
            # op = self.opacity_activation(torch.cat((self._opacity_training, self._opacity_frozen)))
            # op = op[: 2636]
            # return op
            return self.opacity_activation(self._opacity_training)
    
    def get_covariance(self, scaling_modifier = 1):
        # ghz
        # return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1






    def get_global_params(self):
        mesh_data = self.keyframe_meshes[self.current_keyframe]
  
        indices = self._indices_frozen.to(torch.long)

        # 解析 indices 中的各个部分         
        v1=mesh_data[indices,0,:].squeeze(1)
        v2=mesh_data[indices,1,:].squeeze(1)
        v3=mesh_data[indices,2,:].squeeze(1)

        
        # 计算局部坐标系的原点，即  
        o = (v1 + v2 + v3) / 3
        
        # 计算局部坐标系的基向量
        x = (v1 - o).float()
        x /= torch.norm(x, dim=1, keepdim=True)
        
        n = torch.cross(v2 - v1, v3 - v1, dim=1).float()
        n /= torch.norm(n, dim=1, keepdim=True)

        y = torch.cross(n, x, dim=1)
        y /= torch.norm(y, dim=1, keepdim=True)

        # 组成变换矩阵，每个点一个矩阵
        transform_matrices = torch.stack([x, y, n], dim=-1)
        transform_rotation_q = self.rotation_activation(quat_xyzw_to_wxyz(rotmat_to_unitquat(transform_matrices)))

        self._global_xyz = torch.matmul(transform_matrices, (self._xyz_frozen).unsqueeze(-1) ).squeeze(-1) + o
        self._global_rotation = quat_xyzw_to_wxyz(quat_product(quat_wxyz_to_xyzw(transform_rotation_q), quat_wxyz_to_xyzw(self.rotation_activation(self._rotation_frozen))))
        self._global_scaling = self.scaling_activation(self._scaling_frozen)
        self._global_opacity = self.opacity_activation(self._opacity_frozen)
        self._global_features = torch.cat((self._features_dc_frozen, self._features_rest_frozen), dim=1)

    


 






    
    #WDD 5-6
    #增加了keyframe_meshes的输入参数
    #def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
    def create_from_pcd(self, pcd : BasicPointCloud, keyframe_meshes:dict, spatial_lr_scale:float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        #WDD 5-1 /5-6
        #=========
        self.indices         = torch.tensor(np.asarray(pcd.indices)).long().cuda() 
        self.keyframe_meshes = {}
        
        for keyframe, value in keyframe_meshes.items():
            self.keyframe_meshes[keyframe]=torch.tensor(value).float().cuda() 

        #========= 

        # ghz 
        # 计算第一帧的face area
        #=========
        mesh_0_data=self.keyframe_meshes[0]      
        v1=mesh_0_data[:,0,:].squeeze(1)
        v2=mesh_0_data[:,1,:].squeeze(1)  
        v3=mesh_0_data[:,2,:].squeeze(1)

        self.canonical_face_area = 0.5 * torch.norm(torch.cross(v2 - v1, v3 - v1), dim=1)[:, None]
        #=========

        # opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))  # 原始高斯
        opacities = inverse_sigmoid(0.9 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        

    def training_setup(self, training_args):
     
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")

        l = [
                # {'params' : self.xyz_offset_net.parameters(), 'lr' : 1e-5},
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
   
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
            
        

    def update_frame(self, keyframe, len_cam, dynamic):

        self.current_keyframe = keyframe 
        
       
        
        if not dynamic:
            # self._indices_frozen = torch.tensor([]).cuda()
            # self._indices_training = self.indices.cuda()
            return 1
        

        
        if dynamic:            

            # # ghz 隔n帧全部点参与训练
            # if len_cam % 20 == 0:
            #     indices_training = torch.tensor(range(0, torch.max(self.indices))).cuda()
            # else:
            #     indices_training = self.training_indices[self.current_keyframe].cuda()      

            # # ghz 始终部分点参数训练
            # indices_training = self.training_indices[self.current_keyframe].cuda() 

            # ghz 隔n帧部分点参与训练
            if len_cam % 1 == 0:
                indices_training = torch.tensor(range(0, torch.max(self.indices))).cuda()
            else:
                indices_training = self.training_indices[self.current_keyframe].cuda()

            # # ghz 始终全部点参数训练
            # indices_training = torch.tensor(range(0, torch.max(self.indices))).cuda()


            if -1 in indices_training:
                return 0            
            
            # # # ghz 1104 每帧随机增加random_len个面参与训练
            # all_indices = torch.arange(9976).cuda()
            # random_len = 2000
            # indices_non_training = all_indices[~torch.isin(all_indices, indices_training)]
            # random_indices = indices_non_training[torch.randperm(len(indices_non_training))[:random_len]]
            # indices_training = torch.cat((indices_training, random_indices))


            # ghz 处理高斯是否参与训练
            gaussian_indices = torch.where(torch.isin(self.indices, indices_training).squeeze())
            non_gaussian_indices = torch.where(~torch.isin(self.indices, indices_training).squeeze())

            self._indices_frozen = self.indices.clone()[non_gaussian_indices].cuda()
            self._indices_training = self.indices.clone()[gaussian_indices].cuda()
            
            self.gaussian_indices = gaussian_indices
            self.non_gaussian_indices = non_gaussian_indices

            self._xyz_frozen = self._xyz[non_gaussian_indices]
            self._rotation_frozen = self._rotation[non_gaussian_indices]
            self._scaling_frozen = self._scaling[non_gaussian_indices]
            self._opacity_frozen = self._opacity[non_gaussian_indices]
            self._features_dc_frozen = self._features_dc[non_gaussian_indices]
            self._features_rest_frozen = self._features_rest[non_gaussian_indices]

        return 1

    def update_training_param(self):
        self._xyz_training = self._xyz[self.gaussian_indices]  
        self._rotation_training = self._rotation[self.gaussian_indices]
        self._scaling_training = self._scaling[self.gaussian_indices]
        self._opacity_training = self._opacity[self.gaussian_indices]
        self._features_dc_training = self._features_dc[self.gaussian_indices]
        self._features_rest_training = self._features_rest[self.gaussian_indices]

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'em_x', 'em_y', 'em_z','nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')

        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        # ghz
        #=================
        for i in range(self._scaling.shape[1]):
            l.append('em_scale_{}'.format(i))
        #=================

        for i in range(self._rotation.shape[1]):
                    l.append('rot_{}'.format(i))
        # ghz
        #=================
        for i in range(self._rotation.shape[1]):
            l.append('em_rot_{}'.format(i))
        #=================

        # ghz
        #=================
        l.extend(['index'])
        #=================

        # WDD 5-1 / 5-6
        # =================
        # l.extend(['index','loc_x', 'loc_y', 'loc_z'])
        # =================

        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        if not self.dynamic:
            xyz = self.get_xyz.detach().cpu().numpy()
            em_xyz=self._xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self._opacity.detach().cpu().numpy()
            rotation = self.get_rotation.detach().cpu().numpy()
            em_rotation = self._rotation.detach().cpu().numpy()
            scale = self._scaling_log.detach().cpu().numpy()
            em_scale = self._scaling.detach().cpu().numpy()
            # can_area = self.canonical_face_area[self.indices].squeeze(-1).detach().cpu().numpy()
            indices = self.indices.detach().cpu().numpy()
        else:
            xyz = self.get_all_xyz.detach().cpu().numpy()
            em_xyz = torch.cat((self._xyz_training, self._xyz_frozen)).detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = torch.cat((self._features_dc_training, self._features_dc_frozen)).detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = torch.cat((self._features_rest_training, self._features_rest_frozen)).detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = torch.cat((self._opacity_training, self._opacity_frozen)).detach().cpu().numpy()
            rotation = self.get_all_rotation.detach().cpu().numpy()
            em_rotation = torch.cat((self._rotation_training, self._rotation_frozen)).detach().cpu().numpy()
            scale = torch.cat((self._scaling_training, self._scaling_frozen)).detach().cpu().numpy()
            em_scale = torch.cat((self._scaling_training, self._scaling_frozen)).detach().cpu().numpy()
            # can_area = self.canonical_face_area[self.indices].squeeze(-1).detach().cpu().numpy()

            # indices_training = self.training_indices[self.current_keyframe]
            # gaussian_indices = torch.isin(self.indices, indices_training).squeeze()
            indices = torch.cat((self.indices[self.gaussian_indices], self.indices[self.non_gaussian_indices])).detach().cpu().numpy()

            # xyz = self.get_xyz.detach().cpu().numpy()
            # em_xyz = self._xyz_training.detach().cpu().numpy()
            # normals = np.zeros_like(xyz)
            # f_dc = self._features_dc_training.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            # f_rest = self._features_rest_training.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            # opacities = self._opacity_training.detach().cpu().numpy()
            # rotation = self.get_rotation.detach().cpu().numpy()
            # em_rotation = self._rotation_training.detach().cpu().numpy()
            # scale = self._scaling_log.detach().cpu().numpy()
            # em_scale = self._scaling_log.detach().cpu().numpy()
            

            # training_indices = self.training_indices[self.current_keyframe]
            # gaussian_indices = torch.isin(self.indices, training_indices).squeeze()
            # can_area = self.canonical_face_area[self.indices][gaussian_indices].squeeze(-1).detach().cpu().numpy()
            # indices = self.indices[gaussian_indices].detach().cpu().numpy()

            # # can_area = self.canonical_face_area[self.indices].squeeze(-1).detach().cpu().numpy()
            # # indices = self.indices.detach().cpu().numpy()



        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        #attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        #elements[:] = list(map(tuple, attributes))
        #el = PlyElement.describe(elements, 'vertex')
        #PlyData([el]).write(path)

        attributes = np.concatenate((xyz, em_xyz, normals, f_dc, f_rest, opacities, scale, em_scale, rotation, em_rotation, indices), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        #==============================

        # net_path = path.replace('.ply', '_net')
        # torch.save(self.xyz_offset_net.state_dict(), net_path)


    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity_training = optimizable_tensors["opacity"]


    # ghz 渲染用
    def load_render_ply(self, path:str):
        plydata = PlyData.read(path)
    
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                                np.asarray(plydata.elements[0]["y"]),
                                np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        features_dc = np.stack((np.asarray(plydata.elements[0]["f_dc_0"]),
                                np.asarray(plydata.elements[0]["f_dc_1"]),
                                np.asarray(plydata.elements[0]["f_dc_2"])),  axis=1)[..., None]
        
        # features_extra = np.stack(([np.asarray(plydata.elements[0][f"f_rest_{i}"]) for i in range(45)]),  axis=1).reshape(features_dc.shape[0], features_dc.shape[1], -1)
        features_extra = np.zeros([features_dc.shape[0], 3, 15])

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., None]

        scales = np.stack((np.asarray(plydata.elements[0]["scale_0"]),
                           np.asarray(plydata.elements[0]["scale_1"]),
                           np.asarray(plydata.elements[0]["scale_2"])),  axis=1)
        
        rots = np.stack((np.asarray(plydata.elements[0]["rot_0"]),
                         np.asarray(plydata.elements[0]["rot_1"]),
                         np.asarray(plydata.elements[0]["rot_2"]),
                         np.asarray(plydata.elements[0]["rot_3"])),  axis=1)
        
        self._xyz =  torch.tensor(xyz, dtype=torch.float, device="cuda")
        self._features_dc = torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
        self._features_rest = torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
        self._features = torch.cat((self._features_dc, self._features_rest), dim=1)
        self._opacity = torch.tensor(opacities, dtype=torch.float, device="cuda")
        self._scaling = torch.tensor(scales, dtype=torch.float, device="cuda")
        self._rotation = torch.tensor(rots, dtype=torch.float, device="cuda")

        self.active_sh_degree = self.max_sh_degree

    # ghz 计算PSNR用
    def load_params(self, gaussian_params):
        xyz = gaussian_params.xyz
        features_dc = gaussian_params.features_dc
        features_extra = gaussian_params.features_rest
        opacities = gaussian_params.opacity
        scales = gaussian_params.scale
        rots = gaussian_params.rot
        
        self._xyz =  torch.tensor(xyz, dtype=torch.float, device="cuda")
        self._features_dc = torch.tensor(features_dc, dtype=torch.float, device="cuda")[:, None].contiguous()
        self._features_rest = torch.tensor(features_extra, dtype=torch.float, device="cuda").reshape(-1, 3, 15).transpose(1, 2).contiguous()
        self._features = torch.cat((self._features_dc, self._features_rest), dim=1)
        self._opacity = torch.tensor(opacities, dtype=torch.float, device="cuda")
        self._scaling = torch.tensor(scales, dtype=torch.float, device="cuda")
        self._rotation = torch.tensor(rots, dtype=torch.float, device="cuda")

        self.active_sh_degree = self.max_sh_degree
        
        




    # ghz 读取新的ply文件
    def load_ply(self, path:str, keyframe_meshes:dict, training_indices:dict, spatial_lr_scale:float):
        self.spatial_lr_scale = spatial_lr_scale
        plydata = PlyData.read(path)
    
        xyz = np.stack((np.asarray(plydata.elements[0]["em_x"]),
                                np.asarray(plydata.elements[0]["em_y"]),
                                np.asarray(plydata.elements[0]["em_z"])),  axis=1)
        
        features_dc = np.stack((np.asarray(plydata.elements[0]["f_dc_0"]),
                                np.asarray(plydata.elements[0]["f_dc_1"]),
                                np.asarray(plydata.elements[0]["f_dc_2"])),  axis=1)[..., None]
        
        features_extra = np.stack(([np.asarray(plydata.elements[0][f"f_rest_{i}"]) for i in range(45)]),  axis=1).reshape(features_dc.shape[0], features_dc.shape[1], -1)

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., None]

        scales = np.stack((np.asarray(plydata.elements[0]["em_scale_0"]),
                           np.asarray(plydata.elements[0]["em_scale_1"]),
                           np.asarray(plydata.elements[0]["em_scale_2"])),  axis=1)
        
        rots = np.stack((np.asarray(plydata.elements[0]["em_rot_0"]),
                         np.asarray(plydata.elements[0]["em_rot_1"]),
                         np.asarray(plydata.elements[0]["em_rot_2"]),
                         np.asarray(plydata.elements[0]["em_rot_3"])),  axis=1)
    
        self.indices = torch.tensor(np.stack(np.asarray(plydata.elements[0]['index'])))[..., None].long().cuda()
    

        
        for keyframe, value in keyframe_meshes.items():
            self.keyframe_meshes[keyframe]=torch.tensor(value).float().cuda() 

        for keyframe, value in training_indices.items():
            self.training_indices[keyframe]=torch.tensor(value).float().cuda() 


        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

    
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
