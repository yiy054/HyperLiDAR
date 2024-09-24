from omegaconf import OmegaConf
import torch
import importlib
from models.kpconv.architecture import KPFCNN
import importlib
import argparse
import os
from loader import Loader_Data

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
data_loader = Loader_Data(cfg, device)
data_train_xt, labels_train_x, data_test_xt, labels_test_x = data_loader.load('/root/main/dataset/training_10_classes/Lille1_1.ply')