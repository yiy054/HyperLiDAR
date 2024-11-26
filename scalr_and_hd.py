from models.waffleiron.segmenter import Segmenter
import torch
from datasets import LIST_DATASETS, Collate
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
import warnings
import copy
import random
import numpy as np
import os

import argparse
import wandb
from torchmetrics.classification import MulticlassJaccardIndex
import torchmetrics
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

import torchhd
from torchhd.models import Centroid
from torchhd import embeddings

class Encoder(nn.Module):
    def __init__(self, hd_dim, size):
        super(Encoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.projection = embeddings.Projection(size, hd_dim)

    def forward(self, x):
        sample_hv = self.projection(x)
        return torchhd.hard_quantize(sample_hv)


class HD_Model:
    def __init__(self, in_dim, out_dim, num_classes, device):

        encode = Encoder(out_dim, in_dim)
        self.encode = encode.to(device)

        model = Centroid(out_dim, num_classes)
        self.model = model.to(device)
        self.device = device

    def normalize(self, samples):

        """ Normalize with Z-score"""

        mean = torch.mean(samples, dim=0)
        std = torch.std(samples, dim=0)

        samples = (samples - mean) / (std + 1e-8)

        return samples

    def train(self, features, labels, num_voxels):

        """ Initial training pass """

        assert len(features) == len(labels)

        print("\nTrain First\n")

        for i in tqdm(range(len(features)), desc="1st Training:"):
            first_sample = torch.Tensor(features[i][:int(num_voxels[i])]).to(self.device)
            first_label = torch.Tensor(labels[i][:int(num_voxels[i])]).to(torch.int64).to(self.device)

            first_sample = self.normalize(first_sample) # Z1 score seems to work

            #for vox in range(len(first_sample)):
                
            #    label = first_label[vox]
            #    if vox % 5000 == 0:
            #        print(f"Sample {i}: Voxel {vox}")
                
            # HD training
            samples_hv = self.encode(first_sample)
            #samples_hv = samples_hv.reshape((1,samples_hv.shape[0]))
            self.model.add(samples_hv, first_label)

    def retrain(self, features, labels, num_voxels):
        
        """ Retrain with misclassified samples (also substract)"""
        
        for e in tqdm(range(10), desc="Epoch"):
            count = 0

            for i in range(len(features)):
                first_sample = torch.Tensor(features[i][:int(num_voxels[i])]).to(self.device)
                first_label = torch.Tensor(labels[i][:int(num_voxels[i])]).to(torch.int64).to(self.device)

                first_sample = self.normalize(first_sample) # Z1 score seems to work

                #for vox in range(len(first_sample)):
                samples_hv = self.encode(first_sample)
                sim = self.model(samples_hv, dot=True)
                pred_hd = sim.argmax(1).data

                is_wrong = first_label != pred_hd

                # cancel update if all predictions were correct
                if is_wrong.sum().item() == 0:
                    continue

                # only update wrongly predicted inputs
                samples_hv = samples_hv[is_wrong]
                first_label = first_label[is_wrong]
                pred_hd = pred_hd[is_wrong]

                count = first_label.shape[0]

                self.model.weight.index_add_(0, first_label, samples_hv)
                self.model.weight.index_add_(0, pred_hd, samples_hv, alpha=-1.0)

                print(f"Misclassified for {i}: ", count)

            # If you want to test for each sample
            self.test_hd(features, labels, num_voxels)

    def test_hd(self, features, labels, num_voxels, epoch=0):

        """ Testing over all the samples in all the scans given """

        assert len(features) == len(labels)
        
        # Metric
        miou = MulticlassJaccardIndex(num_classes=16, average=None).to(self.device)
        final_shape = int(np.sum(num_voxels))
        final_labels = torch.empty((final_shape), device=self.device)
        final_pred = torch.empty((final_shape), device=self.device)
        
        start_idx = 0
        for i in tqdm(range(len(features)), desc="Testing"):
            shape_sample = int(num_voxels[i])
            first_sample = torch.Tensor(features[i][:shape_sample]).to(self.device)
            first_label = torch.Tensor(labels[i][:shape_sample]).to(torch.int64)
            final_labels[start_idx:start_idx+shape_sample] = first_label

            first_sample = self.normalize(first_sample) # Z1 score seems to work

            # HD inference
            samples_hv = self.encode(first_sample)
            pred_hd = self.model(samples_hv, dot=True).argmax(1).data
            final_pred[start_idx:start_idx+shape_sample] = pred_hd

            start_idx += shape_sample

        print("================================")

        #print('pred_ts', pred_ts)
        print('pred_hd', final_pred, "\tShape: ", final_pred.shape)
        print('label', final_labels, "\tShape: ", final_labels.shape)
        accuracy = miou(final_pred, final_labels)
        avg_acc = torch.mean(accuracy)
        print(f'accuracy: {accuracy}')
        print(f'avg acc: {avg_acc}')

        log_data = {f"Training class_{i}_IoU": c for i, c in enumerate(accuracy)}
        log_data["Retraining epoch"] = avg_acc
        wandb.log(log_data)

        #cm = confusion_matrix(pred_hd, first_label, labels=torch.Tensor(range(0,15)))
        #print("Confusion matrix \n")
        #print(cm)

        print("================================")

class Feature_Extractor:
    def __init__(self, input_channels=5, feat_channels=768, depth=48, 
                 grid_shape=[[256, 256], [256, 32], [256, 32]], nb_class=16, layer_norm=True, 
                 device=torch.device("cpu")):
        self.model = Segmenter(
            input_channels=input_channels,
            feat_channels=feat_channels,
            depth=depth,
            grid_shape=grid_shape,
            nb_class=nb_class, # class for prediction
            #drop_path_prob=config["waffleiron"]["drop_path"],
            layer_norm=layer_norm,
        )

        classif = torch.nn.Conv1d(
            feat_channels, nb_class, 1
        )
        torch.nn.init.constant_(classif.bias, 0)
        torch.nn.init.constant_(classif.weight, 0)
        self.model.classif = torch.nn.Sequential(
            torch.nn.BatchNorm1d(feat_channels),
            classif,
        )

        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.classif.parameters():
            p.requires_grad = True

        def get_optimizer(parameters):
            return torch.optim.AdamW(
                parameters,
                lr=0.001,
                weight_decay=0.003,
            )

        optim = get_optimizer(self.model.parameters())
        self.device = device
        self.device_string = "cuda:0" if torch.cuda.is_available() else "cpu"

    
    def load_pretrained(self, path):
        # Load pretrained model
        path_to_ckpt = path
        checkpoint = torch.load(path_to_ckpt,
            map_location=device_string)
        state_dict = checkpoint["net"]  # Adjust key as needed
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            new_key = k.replace("module.", "")  # Remove "module." prefix
            new_state_dict[new_key] = v

        self.model.load_state_dict(new_state_dict)

        print(
            f"Checkpoint loaded on {self.device_string}: {path_to_ckpt}"
        )

        if self.device_string != 'cpu':
            torch.cuda.set_device(self.device_string) # cuda:0
            self.model = self.model.cuda(self.device_string) # cuda:0

        self.model.eval()


    def test(self, results, labels, num_voxels, device):
        assert len(results) == len(labels)
        
        # Metric
        miou = MulticlassJaccardIndex(num_classes=16, average=None).to(device)
        final_shape = int(np.sum(num_voxels))
        final_labels = torch.empty((final_shape), device=device)
        final_pred = torch.empty((final_shape), device=device)
        
        start_idx = 0
        for i in tqdm(range(len(results)), desc="Testing SoA"):
            shape_sample = int(num_voxels[i])
            first_sample = torch.Tensor(results[i][:shape_sample]).to(device)
            first_label = torch.Tensor(labels[i][:shape_sample]).to(torch.int64)
            final_labels[start_idx:start_idx+shape_sample] = first_label

            pred = first_sample.max(1)[1]
            final_pred[start_idx:start_idx+shape_sample] = pred

            start_idx += shape_sample

        print("================================")

        print('pred', final_pred, "\tShape: ", final_pred.shape)
        print('label', final_labels, "\tShape: ", final_labels.shape)
        accuracy = miou(final_pred, final_labels)
        avg_acc = torch.mean(accuracy)
        print(f'accuracy: {accuracy}')
        print(f'avg acc: {avg_acc}')

        #cm = confusion_matrix(pred_hd, first_label, labels=torch.Tensor(range(0,15)))
        #print("Confusion matrix \n")
        #print(cm)

        print("================================")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-stop', '--layers', type=int, help='how many layers deep', default=48)
    #parser.add_argument('-soa', '--soa', action="store_true", default=False, help='Plot SOA')
    parser.add_argument('-number_samples', '--number_samples', type=int, help='how many scans to train', default=10000)
    parser.add_argument(
            "--seed", default=None, type=int, help="Seed for initializing training"
        )
    #parser.add_argument('-val', '--val', action="store_true", default=False, help='Train with validation for each scan')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_arguments()

    # Set seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)

    wandb.login(key="9487c04b8eff0c16cac4e785f2b57c3a475767d3")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))
    device_string = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    fe = Feature_Extractor()
    fe.load_pretrained('/root/main/ScaLR/saved_models/ckpt_last_scalr.pth')
