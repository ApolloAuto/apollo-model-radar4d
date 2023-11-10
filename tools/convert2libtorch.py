import argparse
import glob
from pathlib import Path
import numpy as np
import torch
import torch.onnx
from torch.onnx import OperatorExportTypes

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/radar_object_detection.yaml')
    parser.add_argument('--ckpt', type=str, default='../output/kitti_models/radar_object_detection/default/ckpt/checkpoint_epoch_80.pth')
    parser.add_argument('--output_path', type=str, default='./radar_detection_libtorch_model.zip')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()

    logger = common_utils.create_logger('1.txt', rank=cfg.LOCAL_RANK)

    dist_test = False
    total_gpus = 1

    # data
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=dist_test, workers=4, logger=logger, training=True
    )
    
    # model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), 
                          dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()
    
    with torch.no_grad():
        for idx, data_dict in enumerate(test_set):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = test_set.collate_batch([data_dict])
            # # data_dict['calib']=torch.tensor(data_dict['calib'])   
            data_dict['batch_size']=torch.tensor(data_dict['batch_size'])   
            data_dict['frame_id']=torch.tensor(int(data_dict['frame_id']))
            load_data_to_gpu(data_dict)
            model.forward = model.forward_for_apollo
            traced_script_module = torch.jit.trace(model, \
                (data_dict['voxels'], data_dict['voxel_num_points'], data_dict['voxel_coords']))
            traced_script_module.save(args.output_path)
            break


if __name__ == '__main__':
    main()
    print("Model to libtorch is over!!")
