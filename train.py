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
import datetime

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import dynamic_render, render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import torch.nn.functional as F

from PIL import Image
import cv2
import numpy as np
from utils.camera_utils import Camera

from utils.general_utils import PILtoTorch 
import copy

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import threading
import time


from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

WARNED = False
#WDD 5-13 
#一个新的类，用来将多个数据进行缓冲
class Viewpoint_Buffer:
    def __init__(self,args, index,white_background=True):
        self.index=index
        self.viewpoint_stack    =   []                  #保存缓存的viewpoint数据
        self.is_loaded          =   False               #判断是否完成了数据的读取
        self.white_background   =   white_background
        self.current_index      =   0
        self.args = args

        #self.lock = threading.Lock()

    # 新增加数据
    def append(self,viewpoint_cam):
        self.viewpoint_stack.append(viewpoint_cam)

    # 判断是否是最终的一个元素
    def is_last(self):
        return self.current_index==len(self.viewpoint_stack)
    
    # 判断是否是第一个元素
    def is_first(self):
        return self.current_index==0
    
    # 获得数据
    def pop(self): 
        if self.current_index>=len(self.viewpoint_stack):
            self.current_index=0
        viewpoint=self.viewpoint_stack[self.current_index] 
        self.current_index=self.current_index+1
      

        return viewpoint
    
    # 获得读取shuju
    def get_is_load(self):
        return   self.is_loaded 
    

    # 读取数据
    def load_images(self): 
        # 创建线程对象
        thread = threading.Thread(target=self.load_images_in_thr) 
        # 启动线程
        thread.start()

    #在多线程中读取数据
    def load_images_in_thr(self): 
        
        # 假设这里是加载图片的代码
        #with self.lock:  # 使用锁来确保线程安全

        self.is_loaded = False  
        for i in range(len(self.viewpoint_stack)):
            self.viewpoint_stack[i]=self.load_image_to_viewpoint(self.args, self.viewpoint_stack[i])
        self.is_loaded = True



    #在训练过程中 批量读取图象的代码
    def load_image_to_viewpoint(self, args, viewpoint_cam):
        image_path = viewpoint_cam.image_fullname  
        image = Image.open(image_path)

        

        # im_data = np.array(image.convert("RGBA"))
 
        # bg = np.array([1,1,1]) if self.white_background else np.array([0, 0, 0])

        # norm_data = im_data / 255.0
        # arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

        # resized_image_rgb = PILtoTorch(image, image.size)  
        # image=resized_image_rgb[:3, ...] 

        orig_w, orig_h = image.size
        
        resolution_scale = 1.0

        if args.resolution in [1, 2, 4, 8]:
            resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
        else:  # should be a type that converts to float
            if args.resolution == -1:
                if orig_w > 1600:
                    global WARNED
                    if not WARNED:
                        print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                            "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                        WARNED = True
                    global_down = orig_w / 1600
                else:
                    global_down = 1
            else:
                global_down = orig_w / args.resolution

            scale = float(global_down) * float(resolution_scale)
            resolution = (int(orig_w / scale), int(orig_h / scale))

        resized_image_rgb = PILtoTorch(image, resolution)
        image = resized_image_rgb[:3, ...]

        new_viewpoint_cam= Camera(colmap_id=viewpoint_cam.colmap_id,
                            image_fullname=viewpoint_cam.image_fullname,
                            R=viewpoint_cam.R, 
                            T=viewpoint_cam.T, 
                            FoVx=viewpoint_cam.FoVx, 
                            FoVy=viewpoint_cam.FoVy, 
                            K = viewpoint_cam.K, 
                            img_h=viewpoint_cam.img_h, 
                            img_w=viewpoint_cam.img_w,
                            image=image, 
                            gt_alpha_mask=None,
                            image_name=viewpoint_cam.image_name, 
                            uid=viewpoint_cam.uid, 
                            data_device=viewpoint_cam.data_device,
                            keyframe=viewpoint_cam.keyframe,
                           )
        return new_viewpoint_cam





def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, dynamic):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, dynamic=dynamic, shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # 公共变量初始化
    # 用来分配view的缓冲
    image_count = 16 #一次读取进入内存的图象数量 
    viewpoint_buffer_stack=[]
    temp_viewpoint_buffer=None

    gaussians.dynamic = dynamic

    # # ghz 1029 读取smplx参数
    # smplx_data = np.load('data_avatarrex\zxc\smpl_params.npz', allow_pickle=True)
    # smplx_data = dict(smplx_data)
    # smplx_data = {k:torch.from_numpy(v.astype(np.float32)) for k,v in smplx_data.items()}

    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        #if not viewpoint_stack:
        #    viewpoint_stack = scene.getTrainCameras().copy() 
        #viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
   
        #WDD 5-13
        #增加viewpoint的缓冲分配和多线程读取
        #当所有缓冲已经都训练完毕了，重新分配缓冲
        #=======================================================================
        if not viewpoint_buffer_stack or len(viewpoint_buffer_stack)==1:
            viewpoint_stack = scene.getTrainCameras().copy()
            index=0
            while viewpoint_stack:
                sub_stack = Viewpoint_Buffer(dataset, index, white_background=dataset.white_background)
                sub_start_frame = randint(0, len(viewpoint_stack)/image_count - 1) * image_count
                for i in range(image_count):
                    if viewpoint_stack:
                        # sub_stack.append(viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)))

                        # sub_stack.append(viewpoint_stack.pop(0))
                        sub_stack.append(viewpoint_stack.pop(sub_start_frame))
                 
                viewpoint_buffer_stack.append(sub_stack)
                index = index + 1

            # 初始的时候读取buffer的数据
            if viewpoint_buffer_stack: 
                viewpoint_buffer_stack[0].load_images()    
                pre_view = viewpoint_buffer_stack[0].viewpoint_stack[0].keyframe
                with torch.no_grad():
                    ret = gaussians.update_frame(viewpoint_buffer_stack[0].viewpoint_stack[0].keyframe, len(viewpoint_buffer_stack), dynamic)
                    
                    gaussians.get_global_params()

                        
        #初始的时候等待 读取
        while not viewpoint_buffer_stack[0].get_is_load() and not temp_viewpoint_buffer:
            # 当前进程暂停1秒
            time.sleep(1)

 

        if viewpoint_buffer_stack[0].get_is_load():
            # if viewpoint_buffer_stack[0].is_first(): 
            #     with torch.no_grad():
            #         ret = gaussians.update_frame(viewpoint_buffer_stack[0].viewpoint_stack[0].keyframe, len(viewpoint_buffer_stack), dynamic)                    
            #         if dynamic:
            #             gaussians.get_global_params()
            #         else:
            #             pass    
            viewpoint_cam=viewpoint_buffer_stack[0].pop()
        else:
            viewpoint_cam=temp_viewpoint_buffer.pop()

        if viewpoint_buffer_stack[0].is_last(): 
            temp_viewpoint_buffer=copy.deepcopy(viewpoint_buffer_stack[0]) 
            # viewpoint_buffer_stack.pop(0)
            if len(viewpoint_buffer_stack)>1:
                viewpoint_buffer_stack.pop(0)
                viewpoint_buffer_stack[0].load_images()     
                # else:    
            #     temp_viewpoint_buffer=copy.deepcopy(viewpoint_buffer_stack[0]) 
            #     viewpoint_buffer_stack.pop(0)

        #=======================================================================
        # print(viewpoint_cam.image_fullname)

        # if not ret:
        #     if (iteration in saving_iterations):
        #         print("\n[ITER {}] Saving Gaussians, frame={}".format(iteration, viewpoint_cam.keyframe))
        #         scene.save(iteration)
        #     continue
            


        #WDD 5-10 
        #为了防止内存溢出，每词都重新读取图象
        #训练会慢很多
        #====================================================================
        is_LotsofImage=True

        #if is_LotsofImage:
        #    viewpoint_cam=load_image_to_viewpoint(viewpoint_cam)
        #====================================================================
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # ghz 更新冻结参数
        if pre_view != viewpoint_cam.keyframe:
            with torch.no_grad():
                ret = gaussians.update_frame(viewpoint_cam.keyframe, len(viewpoint_buffer_stack), dynamic)
                gaussians.get_global_params()

        pre_view = viewpoint_cam.keyframe

        # ghz 更新当前训练帧的一些参数
        gaussians.update_training_param()

        # with torch.no_grad():
        #     ret = gaussians.update_frame(viewpoint_buffer_stack[0].viewpoint_stack[0].keyframe, len(viewpoint_buffer_stack), dynamic)
        #     if dynamic:
        #         gaussians.get_global_params()

        if dynamic:
            render_pkg = dynamic_render(viewpoint_cam, gaussians, pipe, bg)
        else:
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # mask2d = render_pkg["mask2D"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        # Ll1 = l1_loss(image[mask2d], gt_image[mask2d])
        Ll1 = l1_loss(image, gt_image)

        Lssim = (1.0 - ssim(image, gt_image))



        # cv2.imwrite('temp.png', image.transpose(2, 0).detach().cpu().numpy() * 255)


        # xyz_loss
        # xyz = gaussians._xyz_training
        # xyz_lengths = torch.norm(xyz, dim=1)
        # xy_Loss = torch.sum(xyz_lengths)
        # xy_weight = 0.1 

        # scale loss
        scale_loss = F.relu(torch.exp(gaussians._scaling_training) - 0.6).norm(dim=1).mean() 
        # xyz loss
        xyz_loss = F.relu(gaussians._xyz_training.norm(dim=1) - 1.0).mean()

        loss =  (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim  + 0.01 * xyz_loss + scale_loss

        # =====================
        loss.backward()
        

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            #training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            
            if (iteration in saving_iterations):
                image_np = image.detach().cpu().numpy().transpose(1, 2, 0)
                gt_image_np = gt_image.detach().cpu().numpy().transpose(1, 2, 0)
                SSIM_value = compare_ssim(image_np, gt_image_np, multichannel=True, channel_axis=2, data_range=1.0)
                PSNR_value = compare_psnr(image_np, gt_image_np, data_range=1.0)
                print("\n[ITER {}] Saving Gaussians, frame={}".format(iteration, viewpoint_cam.keyframe), 
                      f'SSIM: {SSIM_value}  PSNR: {PSNR_value}')
                # cv2.imwrite("output/temp/image" + str(iteration) + ".png", (image_np * 255).astype(np.uint8))
                # cv2.imwrite("output/temp/gt_image" + str(iteration) + ".png", (gt_image_np * 255).astype(np.uint8))
                scene.save(iteration)


                # print("\n[ITER {}]".format(iteration))
                

            # # 稠密和剪枝
            # # Densification
            # # if not dynamic:
            # if iteration < opt.densify_until_iter:
            #     # Keep track of max radii in image-space for pruning
            #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
            #         min_opacity = 0.005 #0.005
            #         size_threshold = 20
            #         gaussians.densify_and_prune(opt.densify_grad_threshold, min_opacity, scene.cameras_extent, size_threshold)
                
            #     if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            #         gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                # for name, param in gaussians.xyz_offset_net.named_parameters():
                #     if name == 'conv1.weight':
                #         print(f"iteration {iteration}, Layer {name}, param data: {param.data}")

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # if dynamic:
                # _xyz = torch.cat([gaussians._xyz_frozen, gaussians._xyz_training], dim=0) 
                # _xyz = torch.nn.Parameter(_xyz, requires_grad=True)
                # _scaling = torch.cat([gaussians._scaling_frozen, gaussians._scaling_training], dim=0) 
                # _scaling = torch.nn.Parameter(_scaling, requires_grad=True)
                # _rotation = torch.cat([gaussians._rotation_frozen, gaussians._rotation_training], dim=0)
                # _rotation = torch.nn.Parameter(_rotation, requires_grad=True)
                # _opacity = torch.cat([gaussians._opacity_frozen, gaussians._opacity_training], dim=0)
                # _opacity = torch.nn.Parameter(_opacity, requires_grad=True)
                # _features_dc = torch.cat([gaussians._features_dc_training, gaussians._features_dc_frozen], dim=0)
                # _features_dc = torch.nn.Parameter(_features_dc, requires_grad=True)
                # _features_rest = torch.cat([gaussians._features_rest_training, gaussians._features_rest_frozen], dim=0)
                # _features_rest = torch.nn.Parameter(_features_rest, requires_grad=True)

                # gaussians._xyz.copy_(_xyz)
                # gaussians._scaling.copy_(_scaling)
                # gaussians._rotation.copy_(_rotation)
                # gaussians._opacity.copy_(_opacity)
                # gaussians._features_dc.copy_(_features_dc)
                # gaussians._features_rest.copy_(_features_rest)

                


                # # 更新训练结果
                # _xyz = torch.cat((gaussians._xyz_training, gaussians._xyz_frozen))
                # _xyz.requires_grad = True
                # gaussians._xyz.copy_(_xyz)

                # _scale = torch.cat((gaussians._scaling_training, gaussians._scaling_frozen))
                # _scale.requires_grad = True
                # gaussians._scaling.copy_(_scale)

                # _rotation = torch.cat((gaussians._rotation_training, gaussians._rotation_frozen))
                # _rotation.requires_grad = True
                # gaussians._rotation.copy_(_rotation)

                # _opacity = torch.cat((gaussians._opacity_training, gaussians._opacity_frozen))
                # _opacity.requires_grad = True
                # gaussians._opacity.copy_(_opacity)

                # _features_dc =torch.cat((gaussians._features_dc_training, gaussians._features_dc_frozen))
                # _features_dc.requires_grad = True
                # gaussians._features_dc.copy_(_features_dc)

                # _features_rest = torch.cat((gaussians._features_rest_training, gaussians._features_rest_frozen))
                # _features_rest.requires_grad = True
                # gaussians._features_rest.copy_(_features_rest)

               
                # gaussians._xyz = torch.cat((gaussians._xyz_training, gaussians._xyz_frozen))
                # gaussians._rotation = torch.cat((gaussians._rotation_training, gaussians._rotation_frozen))
                # gaussians._scaling = torch.cat((gaussians._scaling_training, gaussians._scaling_frozen))
                # gaussians._opacity = torch.cat((gaussians._opacity_training, gaussians._opacity_frozen))
                # gaussians._features_dc = torch.cat((gaussians._features_dc_training, gaussians._features_dc_frozen))
                # gaussians._features_rest = torch.cat((gaussians._features_rest_training, gaussians._features_rest_frozen))
                # gaussians.indices = torch.cat((gaussians._indices_training, gaussians._indices_frozen))


          


def prepare_output_and_logger(args):    
    if not args.model_path:
        #WDD=============
        #if os.getenv('OAR_JOB_ID'):
        #    unique_str=os.getenv('OAR_JOB_ID')
        #else:
        #    unique_str = str(uuid.uuid4())
        #args.model_path = os.path.join("./output/", unique_str[0:10])
        now = datetime.datetime.now()
        formatted_date = now.strftime("%Y%m%d_%H%M%S")
        args.model_path = os.path.join("./output/", formatted_date)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)


    args = parser.parse_args([
                                '-w',

                                # '-s', 'D:\Guo\data\GA_data/074',
                                # '-s', 'D:\Guo\data\GA_data/104',
                                # '-s', 'D:\Guo\data\GA_data/104', 
                                # '-s', 'D:\Guo\data\GA_data/304', 
                                '-s', 'D:\Guo\data\GA_data/460',
                                # '-s', 'D:\Guo\data\GA_data/104_',
                                # '-s', 'D:\Guo\data\GA_data/460_1',
                                  

                                # '-r','1',
                                "--save_iterations",'1', '200', '500', '1_000', '2_000', '3_000', '4_000', '5_000', '6_000', '7_000', '8_000', '9_000',
                                '11_000','13_000','15_000','20_000','22_000','24_000','26_000','28_000','30_000','32_000','34_000','36_000','38_000',
                                '40_000', '42_000', '44_000', '46_000', '48_000', '50_000', '52_000', '54_000', '56_000', '58_000', 
                                '60_000', '62_000', '64_000', '66_000', '68_000', '70_000', '72_000', '74_000', '76_000', '78_000', 
                                '80_000', '82_000', '84_000', '86_000', '88_000', '90_000', '92_000', '94_000', '96_000', '98_000', 
                                '100_000','150_000','200_000','300_000','500_000',

                                '--iterations', '50_000', 
                                
                                '--dynamic',
                                
                              ])
    #==========================================================================================
    
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.dynamic)

    # All done
    print("\nTraining complete.")
