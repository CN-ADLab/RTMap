import argparse
import mmcv
import os
import shutil
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet3d.utils import collect_env, get_root_logger
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
import sys
sys.path.append('')
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Rectangle
import cv2
from matplotlib.patches import Ellipse
import matplotlib.lines as mlines

CAMS={
    'ring_front_center':'CAM_FRONT_CENTER',
    'ring_front_right':'CAM_FRONT_RIGHT',
    'ring_front_left': 'CAM_FRONT_LEFT',
    'ring_rear_right': 'CAM_REAR_RIGHT',
    'ring_rear_left': 'CAM_REAT_LEFT',
    'ring_side_right': 'CAM_SIDE_RIGHT',
    'ring_side_left': 'CAM_SIDE_LEFT',
}
             
CANDIDATE=[]

def plot_points_with_laplace_variances(x, y, beta_x, beta_y, color, ax, std):
    ax.plot(x, y, color=color, linewidth=1, alpha=0.8, zorder=-1)
    ax.scatter(x, y, color=color, s=1, alpha=0.8, zorder=-1)

    var_x = 2 * beta_x ** 2
    var_y = 2 * beta_y ** 2
    
    for j in range(len(x)):
        if std:
            width = np.sqrt(var_x[j])*5  # if NLL 5, if corrV2 10
            height = np.sqrt(var_y[j])*5
        else:
            width, height = 0, 0
        ellipse = Ellipse((x[j], y[j]), width=width, height=height,
                            fc=color, lw=0.5, alpha=0.3) 
        ax.add_patch(ellipse)

def parse_args():
    parser = argparse.ArgumentParser(description='vis hdmaptr map gt label')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--score-thresh', default=0.3, type=float, help='samples to visualize')
    parser.add_argument(
        '--show-dir', default='./vis_debug', help='directory where visualizations will be saved')
    parser.add_argument('--show-cam', action='store_true', help='show camera pic')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    mmcv.mkdir_or_exist(osp.abspath(args.show_dir))
    cfg.dump(osp.join(args.show_dir, osp.basename(args.config)))
    logger = get_root_logger()
    logger.info(f'DONE create vis_pred dir: {args.show_dir}')

    dataset = build_dataset(cfg.data.test)
    dataset.is_vis_on_test = True #TODO, this is a hack
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        # workers_per_gpu=cfg.data.workers_per_gpu,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    logger.info('Done build test data set')

    # build the model and load checkpoint
    # import pdb;pdb.set_trace()
    cfg.model.train_cfg = None
    # cfg.model.pts_bbox_head.bbox_coder.max_num=15 # TODO this is a hack
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    logger.info('loading check point')
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE
    logger.info('DONE load check point')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    pc_range = cfg.point_cloud_range
    car_img = Image.open('./figs/lidar_car.png')
    # get color map: gt->y, Qmap->b, Qfake->r, Qnew->g
    colors_plt = ['y', 'b', 'r', 'g']

    logger.info('BEGIN vis test dataset samples gt label & pred')

    dataset = data_loader.dataset
    # prog_bar = mmcv.ProgressBar(len(CANDIDATE))
    prog_bar = mmcv.ProgressBar(len(dataset)/10)
    for i, data in enumerate(data_loader):
        if i % 10 != 0:
            continue
        if ~(data['gt_labels_3d'].data[0][0] != -1).any():
            # import pdb;pdb.set_trace()
            logger.error(f'\n empty gt for index {i}, continue')
            prog_bar.update()  
            continue
        
        img = data['img'][0].data[0]
        img_metas = data['img_metas'][0].data[0]
        gt_bboxes_3d = data['gt_bboxes_3d'].data[0]
        gt_labels_3d = data['gt_labels_3d'].data[0]
        gt_delta_pose = data['gt_delta_pose'].data[0][0]
        token = img_metas[0]['scene_token'] + '_' + img_metas[0]['timestamp']

        # pts_filename = img_metas[0]['pts_filename']
        # pts_filename = osp.basename(pts_filename)
        # pts_filename = pts_filename.replace('__LIDAR_TOP__', '_').split('.')[0]
        # if pts_filename not in CANDIDATE:
        #     continue

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        sample_dir = args.show_dir

        result_dic = result[0]['pts_bbox']
        boxes_3d = result_dic['boxes_3d'] # bbox: xmin, ymin, xmax, ymax
        scores_3d = result_dic['scores_3d']
        labels_3d = result_dic['labels_3d']
        pts_3d = result_dic['pts_3d']
        betas_3d = result_dic['betas_3d']
        pred_delta_pose = result_dic['poses_3d']
        pose_error = abs(pred_delta_pose - gt_delta_pose).numpy()
        pose_error[2] *= 180 / np.pi
        gt_delta_pose = abs(gt_delta_pose).numpy()
        gt_delta_pose[2] *= 180 / np.pi
        keep = scores_3d > args.score_thresh

        fig, axs = plt.subplots(1, 3, figsize=(4, 3))
        for ax_i in axs:
            ax_i.set_xlim(pc_range[1], pc_range[4])
            ax_i.set_ylim(pc_range[0], pc_range[3])
            ax_i.axis('off')
        axs[0].set_title("Prior-Map")
        axs[1].set_title("Pred")
        axs[2].set_title("GT")

        # show GT
        gt_lines_fixed_num_pts = gt_bboxes_3d[0].fixed_num_sampled_points
        for gt_bbox_3d, gt_label_3d in zip(gt_lines_fixed_num_pts, gt_labels_3d[0]):
            pts = gt_bbox_3d.numpy()
            if gt_label_3d == 1:  # if ped crossing
                pts = np.vstack([pts, pts[0]])
            x = np.array([pt[0] for pt in pts])
            y = np.array([pt[1] for pt in pts])
            gt_label_3d += 1
            axs[2].plot(-y, x, color=colors_plt[gt_label_3d], linewidth=1, alpha=0.8, zorder=-1)
            axs[2].scatter(-y, x, color=colors_plt[gt_label_3d], s=2, alpha=0.8, zorder=-1)
        axs[2].imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

        # show Pred
        for pred_score_3d, pred_label_3d, pred_pts_3d, pred_beta_3d in zip(scores_3d[keep], labels_3d[keep], pts_3d[keep], betas_3d[keep]):
            pred_pts_3d = pred_pts_3d.numpy()
            if pred_label_3d == 1:
                pred_pts_3d = np.vstack([pred_pts_3d, pred_pts_3d[0]])
                pred_beta_3d = np.vstack([pred_beta_3d, pred_beta_3d[0]])

            pred_label_3d += 1
            plot_points_with_laplace_variances(-pred_pts_3d[:,1], pred_pts_3d[:,0], pred_beta_3d[:,1], pred_beta_3d[:,0], colors_plt[pred_label_3d], axs[1], True)
        axs[1].imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

        # show Prior-Map
        num_vec = cfg['num_vec']
        num_pts_per_vec = cfg['fixed_ptsnum_per_gt_line']
        num_classes = cfg['num_map_classes']
        rtmap_prior = data['rtmap_prior'].data[0][0].reshape(num_vec, num_pts_per_vec, 2+num_classes)[:,:,:2]
        prior_instances_num = data['prior_indices_list'].data[0][0][0].shape[0]
        prior_instances = [np.array(rtmap_prior[i]) for i in range(prior_instances_num)]
        prior_lables_onehot = data['rtmap_prior'].data[0][0].reshape(num_vec, num_pts_per_vec, 2+num_classes)[:prior_instances_num,0,-3:]
        prior_lables_origin = torch.argmax(prior_lables_onehot, dim=1).tolist()
        prior_lables = [-1 if data['prior_indices_list'].data[0][0][0][i] == -1 else prior_lables_origin[i] for i in range(prior_instances_num)]

        for ins, label in zip(prior_instances, prior_lables):
            if label == 1:
                ins = np.vstack([ins, ins[0]])
            label = label + 1
            axs[0].plot(-ins[:, 1], ins[:, 0], color=colors_plt[label], linewidth=1, alpha=0.8, zorder=-1)
            axs[0].scatter(-ins[:, 1], ins[:, 0], color=colors_plt[label], s=2, alpha=0.8, zorder=-1)

        axs[0].imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
        

        legend_items = [
            (colors_plt[0], 'Fake Qeury'),
            (colors_plt[1], 'Divider'),
            (colors_plt[2], 'Ped Crossing'),
            (colors_plt[3], 'Boundary'),
        ]
        legend_handles = [mlines.Line2D([], [], color=color, linewidth=3, label=label) for color, label in legend_items]
        fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=4, fontsize=6)

        gt_delta_pose_str = '(' + ', '.join(f"{x:.3f}" for x in gt_delta_pose) + ')'
        pose_error_str = '(' + ', '.join(f"{x:.3f}" for x in pose_error) + ')'
        fig.text(0.5, -0.02, f'Init Pose Error(x/y/r): {gt_delta_pose_str}, Refined Pose Error(x/y/r): {pose_error_str}', ha='center', va='top', fontsize=6)

        combined_image_path = osp.join(sample_dir, token + '_map.png')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.2)
        plt.savefig(combined_image_path, bbox_inches='tight', format='png', dpi=1200)
        plt.close()

        ## save surrounding imgs
        if args.show_cam:
            from pathlib import Path
            data_path_prefix = str(Path(__file__).resolve().parent.parent.parent)
            filename_list = img_metas[0]['filename']
            img_path_dict = {}
            for filepath, lidar2img, img_aug in zip(filename_list,img_metas[0]['lidar2img'],img_metas[0]['img_aug_matrix']):
                inv_aug = np.linalg.inv(img_aug)
                lidar2orimg = np.dot(inv_aug, lidar2img)
                cam_name = os.path.dirname(filepath).split('/')[-1]
                img_path_dict[cam_name] = dict(
                    filepath=filepath,
                    lidar2img = lidar2orimg)
            rendered_cams_dict = {}
            for key, cam_dict in img_path_dict.items():
                cam_img = cv2.imread(osp.join(data_path_prefix,cam_dict['filepath']))

                if key != 'ring_front_center':
                    cam_img = cv2.resize(cam_img, (2048,1550), interpolation=cv2.INTER_LINEAR)

                if 'front' not in key:
                    # cam_img = cam_img[:,::-1,:]
                    cam_img = cv2.flip(cam_img, 1)
                lw = 8
                tf = max(lw - 1, 1)
                w, h = cv2.getTextSize(CAMS[key], 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
                p1 = (0,0)
                p2 = (w,h+3)
                color=(0, 0, 0)
                txt_color=(255, 255, 255)
                cv2.rectangle(cam_img, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(cam_img,
                            CAMS[key], (p1[0], p1[1] + h + 2),
                            0,
                            lw / 3,
                            txt_color,
                            thickness=tf,
                            lineType=cv2.LINE_AA)
                rendered_cams_dict[key] = cam_img

            new_image_height = 2048
            new_image_width = 1550+2048*2
            color = (255,255,255)
            first_row_canvas = np.full((new_image_height,new_image_width, 3), color, dtype=np.uint8)
            first_row_canvas[(2048-1550):, :2048,:] = rendered_cams_dict['ring_front_left']
            first_row_canvas[:,2048:(2048+1550),:] = rendered_cams_dict['ring_front_center']
            first_row_canvas[(2048-1550):,3598:,:] = rendered_cams_dict['ring_front_right']

            new_image_height = 1550
            new_image_width = 2048*4
            color = (255,255,255)
            second_row_canvas = np.full((new_image_height,new_image_width, 3), color, dtype=np.uint8)
            second_row_canvas[:,:2048,:] = rendered_cams_dict['ring_side_left']
            second_row_canvas[:,2048:4096,:] = rendered_cams_dict['ring_rear_left']
            second_row_canvas[:,4096:6144,:] = rendered_cams_dict['ring_rear_right']
            second_row_canvas[:,6144:,:] = rendered_cams_dict['ring_side_right']

            resized_first_row_canvas = cv2.resize(first_row_canvas,(8192,2972))
            full_canvas = np.full((2972+1550,8192,3),color,dtype=np.uint8)
            full_canvas[:2972,:,:] = resized_first_row_canvas
            full_canvas[2972:,:,:] = second_row_canvas
            cams_img_path = osp.join(sample_dir, token + '_surroudview.jpg')
            cv2.imwrite(cams_img_path, full_canvas,[cv2.IMWRITE_JPEG_QUALITY, 50])

        prog_bar.update()

    logger.info('\n DONE vis val dataset samples pr & gt & pred!')

if __name__ == '__main__':
    main()
