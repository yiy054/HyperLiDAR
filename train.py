from omegaconf import OmegaConf
import torch
import importlib
from models.kpconv.architecture import KPFCNN
import importlib
import argparse
import os
from loader import Loader_Data
from auxiliary.process_data.npm3d.npm3d_dataset import DatasetTrainVal, DatasetTest
from tqdm import tqdm
import numpy as np
from models.HD.online_hd import OnlineHD

parser = argparse.ArgumentParser()
parser.add_argument('-cfg', '--config', help='the path to the setup config file', default='cfg/args.yaml')
args = parser.parse_args()

cfg = OmegaConf.load(args.config)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

module = importlib.import_module('models.kpconv.kpconv')
model_information = getattr(module, "KPFCNN")()
model_information.num_classes = 16
model_information.ignore_label = -1
model_information.in_features_dim = 1
from models.kpconv_model import SemanticSegmentationModel
module = importlib.import_module('models.kpconv.architecture')
model_type = getattr(module, "KPFCNN")
semantic_model = SemanticSegmentationModel(model_information,cfg,model_type)

lbl_values = [i for i in range(16)]
ign_lbls = [-1]
model = KPFCNN(model_information, lbl_values, ign_lbls)
model.to(device)
model.load_state_dict(torch.load(os.path.join(cfg.logger.save_path, cfg.logger.model_name)))

# Data Load
filelist_train = [os.path.join(cfg.target_path, 'train_pointclouds', fname) for fname in os.listdir(os.path.join(cfg.target_path, 'train_pointclouds')) if os.path.splitext(fname)[1]==".npy"]
filelist_train.sort()

print("Creating dataloader...", flush=True)
ds = DatasetTrainVal(filelist_train, os.path.join(cfg.target_path, 'train_pointclouds'),
                            training=True,
                            npoints=cfg.npoints,
                            iteration_number=cfg.batchsize*cfg.trainer.epoch,
                            jitter=cfg.jitter)
train_loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batchsize, shuffle=True,
                                    num_workers=cfg.threads)



hd_model = OnlineHD(256, 5000, 9, cfg, model, device=device)

for epoch in range(0, cfg.trainer.epoch):

    """t = tqdm(train_loader, ncols=100, desc="Train Epoch {}".format(epoch), disable=False)
    for data in t:
        pts = data['pts']#.to(device)
        features = data['features']#.to(device)
        seg = data['target']#.to(device)
        #print(seg)
        #pointcloud = np.hstack((pts.reshape((cfg.batchsize, 3, pts.shape[2])), np.zeros((cfg.batchsize, 1, pts.shape[2])), (seg-1).reshape((cfg.batchsize, 1, seg.shape[1])),np.zeros((cfg.batchsize, 1, pts.shape[2]))))
        pointcloud = np.concatenate((
            pts.reshape((cfg.batchsize, pts.shape[2], 3)), 
            np.zeros((cfg.batchsize, pts.shape[2], 1)), 
            (seg - 1).reshape((cfg.batchsize, seg.shape[1], 1)), 
            np.zeros((cfg.batchsize, pts.shape[2], 1))
        ), axis=2)
        #print("Pointcloud:", pointcloud.shape)
        r_clouds, r_inds_list = semantic_model.prepare_data(pointcloud,False,True)
        x = hd_model.feature_extractor(r_clouds)
        hd_model.fit(x, r_clouds.labels)
        
        # Free the memory
        del r_clouds
        del x
        torch.cuda.empty_cache()

        print(hd_model.model.weight)"""

    ds_val = DatasetTest("ajaccio_57.ply", os.path.join('/root/main/dataset/', 'test_10_classes'))
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=cfg.batchsize, shuffle=False,
                                        num_workers=cfg.threads)

    t_val = tqdm(val_loader, ncols=100, desc="Val Epoch {}".format(epoch), disable=False)
    print("Total Length:", ds_val.__len__())
    pred_complete = np.zeros(())
    for data_val in t_val:
        pts = data_val['pts']#.to(device)
        features = data_val['features']#.to(device)
        pts_ids = data_val['pts_ids']
        print("points_ids: ", pts_ids.shape)
        total_num_points = pts.shape[2]*cfg.batchsize
        pointcloud = np.concatenate((
            pts.reshape((1, total_num_points, 3)), 
            np.zeros((1, total_num_points, 1)), 
            np.zeros((1, total_num_points, 1)) -1, 
            np.zeros((1, total_num_points, 1))
        ), axis=2)
        r_clouds, r_inds_list = semantic_model.prepare_data(pointcloud,False,True)
        #print("r_clouds: ", r_clouds.points[0].shape)
        #print("r_clouds: ", r_clouds.labels.shape)
        #print("r_inds_list len: ", len(r_inds_list))
        #print("r_inds_list shape: ", r_inds_list[0].shape)
        #print("r_inds_list: ", max(r_inds_list[0]))
        y = hd_model.forward(r_clouds)
        #print("y: ", y.shape)
        preds = np.zeros((total_num_points))
        preds = y[r_inds_list[0]]
        preds = preds.reshape((cfg.batchsize, pts.shape[2]))
        print("Preds from batch:", preds)

        enter = input("Enter")




    