from .detector3d_template import Detector3DTemplate
import numpy as np
import torch
from ..model_utils import model_nms_utils


class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.model_cfg = model_cfg

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
    
    def forward_for_apollo(self, voxels, num_points, coors):

        data_dict = {}
        data_dict['voxels'] = voxels
        data_dict['voxel_num_points'] = num_points
        data_dict['voxel_coords'] = coors
        data_dict['batch_size'] = torch.tensor(1)

        for cur_module in self.module_list[0:-1]:
            data_dict = cur_module(data_dict)
        x = data_dict['spatial_features_2d']
    
        self.detection_head = self.module_list[-1]
        cls_preds = self.detection_head.conv_cls(x).permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = self.detection_head.conv_box(x).permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        
        if self.detection_head.conv_dir_cls is not None:
            dir_cls_preds = self.detection_head.conv_dir_cls(x).permute(0, 2, 3, 1).contiguous()
        else:
            dir_cls_preds = None

        return [cls_preds, box_preds, dir_cls_preds]
