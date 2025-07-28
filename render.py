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
import json
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
# from gaussian_renderer import GaussianModel
from PIL import Image
from pathlib import Path
from typing import NamedTuple
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import getWorld2View2, getProjectionMatrix





class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    K: np.array
    FoVx: np.array
    FoVy: np.array
    image: np.array
    image_path: str
    image_name: str
    image_width: int
    image_height: int
    full_proj_transform: np.array
    world_view_transform: np.array
    camera_center: np.array

def readCamerasFromTransforms(path, transformsfile):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
 
        frames = [contents["frames"][0]]
        for idx, frame in enumerate(frames):

            width=0
            height=0

            
            #WDD
            #这里之前有个bug，目录多了一层
            # cam_name = os.path.join(path, frame["file_path"] + extension)
            cam_name = frame["file_path"] # + extension

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            if "K" not in frame:
                K = None
            else:
                K = np.array(frame["K"])
 
            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem


            FoVx = frame["camera_angle_x"]
            FoVy = frame["camera_angle_y"]

            g_image = Image.open(image_path)
            width   =   g_image.size[0]
            height  =   g_image.size[1]
            image = None 

            world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
            projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=FoVx, fovY=FoVy, K=K, img_h=height, img_w=width).transpose(0, 1).cuda()
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            camera_center = world_view_transform.inverse()[3, :3]
    
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, K=K, FoVx=FoVx, FoVy=FoVy, image=image,
                            image_path=image_path, image_name=image_name, image_width=width, image_height=height,
                            full_proj_transform=full_proj_transform, world_view_transform=world_view_transform, camera_center=camera_center))
 

    return cam_infos


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc._xyz, dtype=pc._xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc._xyz
    means2D = screenspace_points
    opacity = pc.opacity_activation(pc._opacity)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.scaling_activation(pc._scaling)
        rotations = pc.rotation_activation(pc._rotation)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = torch.cat((pc._features_dc, pc.features_rest), dim=1).transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc._xyz - viewpoint_camera.camera_center.repeat(torch.cat((pc._features_dc, pc._features_rest), dim=1).shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = torch.cat((pc._features_dc, pc._features_rest), dim=1)
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}





if __name__ == "__main__":

    root_path = r'D:\Guo\data\GA_data\074'
    # root_path = r'D:\Guo\code\GaussianAvatars\data\304\304_EMO-2_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine'
    views = readCamerasFromTransforms(root_path, 'transforms_val.json')



    with torch.no_grad():
        parser = ArgumentParser(description="Testing script parameters")
        pipeline = PipelineParams(parser)

        gaussians = GaussianModel(sh_degree=0)


        # ply_path = r'D:\Guo\code\body\GStest\000_dynamic_gaussian_finetune\@result\GA_074_frame0000\point_cloud\iteration_2000\point_cloud.ply'
        # ply_path = r'D:\Guo\code\body\GStest\000_dynamic_gaussian_finetune\@result\GA_074_frame0060\point_cloud\iteration_7000\point_cloud.ply'
        # ply_path = r'D:\Guo\code\body\GStest\000_dynamic_gaussian_finetune\@result\GA_074_frame0132\point_cloud\iteration_7000\point_cloud.ply'
        # ply_path = r'D:\Guo\code\body\GStest\000_dynamic_gaussian_finetune\@result\GA_074_frame0125\point_cloud\iteration_8000\point_cloud.ply'

        # ply_path = r'D:\Guo\code\body\GStest\000_dynamic_gaussian_finetune\@result\GA_104_frame0000\point_cloud\iteration_13000\point_cloud.ply'
        # ply_path = r'D:\Guo\code\body\GStest\000_dynamic_gaussian_finetune\@result\GA_104_frame0015\point_cloud\iteration_5000\point_cloud.ply'
        # ply_path = r'D:\Guo\code\body\GStest\000_dynamic_gaussian_finetune\@result\GA_104_frame0040\point_cloud\iteration_4000\point_cloud.ply'

        # ply_path = r'D:\Guo\code\body\GStest\000_dynamic_gaussian_finetune\@result\GA_218_frame0000\point_cloud\iteration_20000\point_cloud.ply'
        # ply_path = r'D:\Guo\code\body\GStest\000_dynamic_gaussian_finetune\@result\GA_218_frame0023\point_cloud\iteration_15000\point_cloud.ply'
        # ply_path = r'D:\Guo\code\body\GStest\000_dynamic_gaussian_finetune\@result\GA_218_frame0046\point_cloud\iteration_8000\point_cloud.ply'

        # ply_path = r'D:\Guo\code\body\GStest\000_dynamic_gaussian_finetune\@result\GA_304_frame0000\point_cloud\iteration_11000\point_cloud.ply'
        # ply_path = r'D:\Guo\code\body\GStest\000_dynamic_gaussian_finetune\@result\GA_304_frame0027\point_cloud\iteration_11000\point_cloud.ply'
        # ply_path = r'D:\Guo\code\body\GStest\000_dynamic_gaussian_finetune\@result\GA_304_frame0062\point_cloud\iteration_5000\point_cloud.ply'

        # ply_path = r'D:\Guo\code\body\GStest\000_dynamic_gaussian_finetune\@result\GA_460_frame0000\point_cloud\iteration_11000\point_cloud.ply'
        # ply_path = r'D:\Guo\code\body\GStest\000_dynamic_gaussian_finetune\@result\GA_460_frame0028\point_cloud\iteration_8000\point_cloud.ply'
        ply_path = r'D:\Guo\code\body\GStest\000_dynamic_gaussian_finetune\@result\GA_460_frame0072\point_cloud\iteration_9000\point_cloud.ply'
        
        # ply_path = r'D:\Guo\code\gaussian-splatting-Windows\output\mia\frame_0029\point_cloud\iteration_30000\point_cloud.ply'
        gaussians.load_render_ply(ply_path)
            
            
        bg_color = [1,1,1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = 'render_results'
        makedirs(render_path, exist_ok=True)

        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            rendering = render(view, gaussians, pipeline, background)["render"]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0}'.format(ply_path.split("\\")[-4]) + ".png"))


        # # ply_folder = r'animatable_result'
        # ply_folder = r'animatable_result_074_exp5'

        # # id = '218'S
        # for ply in os.listdir(ply_folder):
        #     ply_path = os.path.join(ply_folder, ply)
        
        #     gaussians.load_render_ply(ply_path)
            
            
        #     bg_color = [1,1,1]
        #     background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        #     render_path = 'render_results'
        #     makedirs(render_path, exist_ok=True)

        #     for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        #         rendering = render(view, gaussians, pipeline, background)["render"]
        #         # torchvision.utils.save_image(rendering, os.path.join(render_path, f'{id}_{ply.split(".")[0]}_STGS' + ".png"))
        #         torchvision.utils.save_image(rendering, os.path.join(render_path, f'{ply.split(".")[0]}' + ".png"))

