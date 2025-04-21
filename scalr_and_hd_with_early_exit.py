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
import time

import matplotlib.pyplot as plt

class Encoder(nn.Module):
    def __init__(self, hd_dim, size):
        super(Encoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.projection = embeddings.Projection(size, hd_dim)
        # self.projection = embeddings.Sinusoid(size, hd_dim)

        ## EDIT - remove this line, not sure what's the point
        #self.projection.weight = nn.Parameter(torchhd.normalize(self.projection.weight), requires_grad=False) # Binary

    def forward(self, x):
        sample_hv = self.projection(x)
        return torchhd.hard_quantize(sample_hv)

class Feature_Extractor:
    def __init__(self, input_channels=5, feat_channels=768, depth=48, 
                 grid_shape=[[256, 256], [256, 32], [256, 32]], nb_class=16, layer_norm=True, 
                 device=torch.device("cpu"), early_exit = [48], **kwargs):
        self.model = Segmenter(
            input_channels=input_channels,
            feat_channels=feat_channels,
            depth=depth,
            grid_shape=grid_shape,
            nb_class=nb_class, # class for prediction
            #drop_path_prob=config["waffleiron"]["drop_path"],
            layer_norm=layer_norm,
            early_exit = early_exit
        )

        classif = torch.nn.Conv1d(
            feat_channels, nb_class, 1 # So it fits 16 = nb_class but classifier is not used
        )
        torch.nn.init.constant_(classif.bias, 0)
        torch.nn.init.constant_(classif.weight, 0)
        self.model.classif = torch.nn.Sequential(
            torch.nn.BatchNorm1d(feat_channels),
            classif,
        )

        for p in self.model.parameters():
            p.requires_grad = False
        #for p in self.model.classif.parameters():
        #    p.requires_grad = True

        #def get_optimizer(parameters):
        #    return torch.optim.AdamW(
        #        parameters,
        #        lr=0.001,
        #        weight_decay=0.003,
        #    )

        #optim = get_optimizer(self.model.parameters())
        self.device = device
        self.device_string = "cuda:0" if (torch.cuda.is_available() and kwargs['args'].device == 'gpu') else "cpu"
        self.num_classes = nb_class
        self.early_exit = early_exit
        self.kwargs = kwargs
        self.model.waffleiron.separate_model()
    
    def load_pretrained(self, path):
        # Load pretrained model
        path_to_ckpt = path
        checkpoint = torch.load(path_to_ckpt,
            map_location=self.device_string)
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

        #self.model.waffleiron.crop_model(self.early_exit)

    def forward_model(self, it, batch, step_type):

        # Checking all of the parameters needed for feature extractor
        # Obj: only pass what you need
        feat = batch["feat"]
        labels = batch["labels_orig"]
        cell_ind = batch["cell_ind"]
        occupied_cell = batch["occupied_cells"]
        neighbors_emb = batch["neighbors_emb"]
        if self.device_string != 'cpu':
            feat = feat.cuda(0, non_blocking=True)
            labels = labels.cuda(0, non_blocking=True)
            batch["upsample"] = [
                up.cuda(0, non_blocking=True) for up in batch["upsample"]
            ]
            cell_ind = cell_ind.cuda(0, non_blocking=True)
            occupied_cell = occupied_cell.cuda(0, non_blocking=True)
            neighbors_emb = neighbors_emb.cuda(0, non_blocking=True)
        self.net_input = (feat, cell_ind, occupied_cell, neighbors_emb)
        self.where = labels != 255
        self.labels = labels[self.where]

        return self.continue_with_model(step_type, flag = 'new_iter', step = 0)

    def continue_with_model(self, step_type, flag='new_iter', tokens = None, step=0):

        if flag == 'new_iter':
            if self.device_string != 'cpu':
                with torch.autocast("cuda", enabled=True):
                    # Logits
                    with torch.no_grad():
                        out = self.model(*self.net_input, step_type)
                        _, tokens, tokens_norm, out, exit_layer = out[0], out[1], out[2], out[3], out[4]
            else:
                with torch.no_grad():
                    out = self.model(*self.net_input, step_type)
                    _, tokens, tokens_norm, out, exit_layer = out[0], out[1], out[2], out[3], out[4]
        
        if flag == 'continue_iter':
            if self.device_string != 'cpu':
                with torch.autocast("cuda", enabled=True):
                    # Logits
                    with torch.no_grad():
                        out = self.model.continue_forward(tokens_init = tokens, iteration = step, step_type = step_type)
                        _, tokens, tokens_norm, out, exit_layer = out[0], out[1], out[2], out[3], out[4]
            else:
                with torch.no_grad():
                    out = self.model.continue_forward(tokens_init = tokens, iteration = step, step_type = step_type)
                    _, tokens, tokens_norm, out, exit_layer = out[0], out[1], out[2], out[3], out[4]

        pred_label = out.max(1)[1]

        return tokens, tokens_norm[0,:,self.where], pred_label[0, self.where], exit_layer

    def test(self, loader, total_voxels):        
        # Metric
        miou = MulticlassJaccardIndex(num_classes=self.num_classes, average=None).to(self.device, non_blocking=True)
        final_labels = torch.empty((total_voxels), device=self.device)
        final_pred = torch.empty((total_voxels), device=self.device)
        
        start_idx = 0
        for it, batch in tqdm(enumerate(loader), desc="SoA testing"):
            features, labels, soa_result = self.forward_model(it, batch)
            shape_sample = labels.shape[0]
            labels = labels.to(dtype = torch.int64, device = self.device, non_blocking=True)
            soa_result = soa_result.to(device=self.device, non_blocking=True)
            final_labels[start_idx:start_idx+shape_sample] = labels

            final_pred[start_idx:start_idx+shape_sample] = soa_result

            start_idx += shape_sample

        final_labels = final_labels[:start_idx]
        final_pred = final_pred[:start_idx]

        print("================================")

        print('Pred FE', final_pred, "\tShape: ", final_pred.shape)
        print('Label', final_labels, "\tShape: ", final_labels.shape)
        accuracy = miou(final_pred, final_labels)
        avg_acc = torch.mean(accuracy)
        print(f'accuracy: {accuracy}')
        print(f'avg acc: {avg_acc}')

        #cm = confusion_matrix(pred_hd, first_label, labels=torch.Tensor(range(0,15)))
        #print("Confusion matrix \n")
        #print(cm)

        print("================================")

class HD_Model:
    def __init__(self, in_dim, out_dim, num_classes, path_pretrained, 
                 device=torch.device("cpu"), **kwargs):

        encode = Encoder(out_dim, in_dim)
        self.encode = encode.to(device=device, non_blocking=True)

        ## Edit - use self-managed linear
        #model = Centroid(out_dim, num_classes)
        classify = nn.Linear(out_dim, num_classes, bias=False)
        classify.weight.data.fill_(0.0)
        self.classify = classify.to(device=device, non_blocking=True)
        # Need a copy of the weight as the unnormalized version
        self.classify_weights = copy.deepcopy(self.classify.weight)
        self.device = device
        self.feature_extractor = Feature_Extractor(nb_class = num_classes, device=self.device, 
                                                   early_exit=[int(i) for i in kwargs['args'].layers], 
                                                   args=kwargs['args'])
        self.feature_extractor.load_pretrained(path_pretrained)
        self.stop = kwargs['args'].layers
        self.point_per_iter = kwargs['args'].batch_points
        self.num_classes = num_classes
        self.max_samples = kwargs['args'].number_samples
        self.test_max_samples = kwargs['args'].test_number_samples
        self.kwargs = kwargs
        self.threshold = {}
        self.exit_val_dict = {}
        self.exit_counter = {}
        for i in kwargs['args'].layers:
            self.threshold[int(i)] = 1
            self.exit_val_dict[int(i)] = []
            self.exit_counter[int(i)] = 0
        self.threshold[48] = 1
        self.exit_counter[48] = 0
        self.alpha_exp_average = 0.05
        # self.update = True
        self.past_update = self.threshold
        self.quantile = kwargs['args'].quantile
        self.epochs = kwargs['args'].epochs
        # self.mean_confidences = [[] for _ in range(self.kwargs['args'].epochs)]
        # self.correct_percentages = [[] for _ in range(self.kwargs['args'].epochs)]
        # self.mean_confidences = np.zeros((kwargs['args'].epochs, len(self.train_loader)))
        # self.correct_percentages = np.zeros((kwargs['args'].epochs, len(self.train_loader)))

    def normalize(self, samples):

        """ Normalize with Z-score"""

        mean = torch.mean(samples, dim=0)
        std = torch.std(samples, dim=0)

        samples = (samples - mean) / (std + 1e-8)

        return samples

    def set_loaders(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_vox_train = 0
        self.num_vox_val = 0

        for loader, desc, attr in [(self.train_loader, "Training loader", "num_vox_train"),
                           (self.val_loader, "Validation loader", "num_vox_val")]:
            for batch in tqdm(loader, desc=desc):
                labels = batch["labels_orig"]

                # Ensure labels are tensors
                if isinstance(labels, list):
                    labels = torch.stack(labels)  # Convert list of tensors to a single tensor
                
                # Move to GPU if applicable
                if self.device.type == "cuda":
                    labels = labels.cuda(non_blocking=True)

                # Compute the number of valid voxels
                setattr(self, attr, getattr(self, attr) + (labels != 255).sum().item())

        print("Finished loading data loaders")
        # self.mean_confidences = np.zeros((self.epochs, len(self.train_loader)))
        # self.correct_percentages = np.zeros((self.epochs, len(self.train_loader)))
    
    def sample_to_encode(self, it, batch, step_type="train"):
        tokens, tokens_norm, soa_labels, exit_layer = self.feature_extractor.forward_model(it, batch, step_type=step_type) # Everything for what hasn't been dropped
        samples_hv = self.encode(torch.transpose(tokens_norm, 0, 1).float())
        self.classify.weight[:] = F.normalize(self.classify_weights)
        logits = None
    
        ### Check if we need to do another iteration:
        steps = 1
        if step_type == 'retrain' or step_type == 'test':
            #print(self.threshold)
            #x = input()

            while exit_layer != 47:
                max_dist, logits = self.check_early_exit(samples_hv)
                val = torch.mean(max_dist)
                self.exit_val_dict[exit_layer+1].append(val.item())
                #print("Exit layer: ", exit_layer)
                #print("Value: ", val)
                #x = input()
                # print("Before Threshold: ", self.threshold)
                # print("Steps: ", steps)
                if val > self.threshold[exit_layer+1] and step_type == 'test':
                    logits = self.classify(F.normalize(samples_hv))
                    break

                # Update threshold
                # if self.update:
                #     self.threshold[exit_layer+1] = ((1-self.alpha_exp_average)*self.threshold[exit_layer+1]) + (self.alpha_exp_average*torch.quantile(max_dist, self.quantile))

                tokens, tokens_norm, soa_labels, exit_layer = self.feature_extractor.continue_with_model(step_type=step_type, flag='continue_iter', tokens = tokens, step = steps)
                samples_hv_next = self.encode(torch.transpose(tokens_norm, 0, 1).float())
                samples_hv = torchhd.bundle(samples_hv_next, samples_hv)
                steps += 1
            
            # if exit_layer != 47 and not self.update:
            #     self.threshold[exit_layer+1] = ((1-self.alpha_exp_average)*self.threshold[exit_layer+1]) + (self.alpha_exp_average*torch.quantile(max_dist, self.quantile))

            # if it % 10 == 9 and self.update:
            #     if self.past_update.values == self.threshold.values:
            #         self.update = False
            #         print(it, "Update stop!!!")
            #     else:
            #         self.past_update = self.threshold
            #         print(self.past_update)
            # (self.alpha_exp_average*val) + ((1-self.alpha_exp_average)*self.threshold[exit_layer+1])
            # print("After Threshold: ", self.threshold)

        if step_type == 'train':
            while exit_layer != 47:
                tokens, tokens_norm, soa_labels, exit_layer = self.feature_extractor.continue_with_model(step_type=step_type, flag='continue_iter', tokens = tokens, step = steps)
                samples_hv_next = self.encode(torch.transpose(tokens_norm, 0, 1).float())
                samples_hv = torchhd.bundle(samples_hv_next, samples_hv)
                steps += 1

        if exit_layer == 47:
            logits = self.classify(F.normalize(samples_hv))
            # Last update
        
        self.exit_counter[exit_layer+1] += 1
        labels = self.feature_extractor.labels
        labels = labels.to(dtype=torch.int64, device = self.device, non_blocking=True)

        #features = self.normalize(features) # Z1 score seems to work
        return samples_hv, labels, soa_labels, logits
    
    def check_early_exit(self, samples_hv):
        # logits = self.classify(F.normalize(samples_hv))
        # max_dist = torch.max(logits, axis=1).values
        # # val = torch.quantile(max_dist, self.quantile)
        # # return val, logits
        # return max_dist, logits
        subset_ratio = 0.1  # You can make this a class parameter
        subset_size = int(samples_hv.size(0) * subset_ratio)
        indices = torch.randperm(samples_hv.size(0))[:subset_size]
        samples_hv_subset = samples_hv[indices]

        logits = self.classify(F.normalize(samples_hv_subset))
        max_dist = torch.max(logits, axis=1).values
        return max_dist, logits

    
    def train(self, weights=None):

        """ Initial training pass """

        print("\nTrain First\n")

        with torch.no_grad():
            for it, batch in tqdm(enumerate(self.train_loader), desc="Training"):
                
                samples_hv, labels, soa_labels, _ = self.sample_to_encode(it, batch, step_type="train")
                
                for b in range(0, samples_hv.shape[0], self.point_per_iter):
                    end = min(b + self.point_per_iter, int(samples_hv.shape[0]))  # Ensure we don't exceed num_voxels[i]
            
                    ##samples_hv = samples_hv.reshape((1,samples_hv.shape[0]))
                        
                    #self.model.add(samples_hv[b:end], labels[b:end])

                    ## EDIT - training with the new linear layer
                    #for i in range(b, end):
                    #    self.classify_weights[labels[i]] += samples_hv[i]
                    if weights is not None:
                        self.classify_weights.index_add_(0, labels[b:end], weights[labels[b:end]].reshape(-1, 1) * samples_hv[b:end])
                    else:
                        self.classify_weights.index_add_(0, labels[b:end], samples_hv[b:end])

                    if self.device == torch.device("cuda:0"):
                        torch.cuda.synchronize(device=self.device)
                
                if it == self.max_samples:
                    break
                ###### End of one scan
            
            ####### End of all scans in single-pass training
                
            #self.model.weight = nn.Parameter(torchhd.normalize(self.model.weight), requires_grad=False) # Binary

            # EDIT - normalize classify_weights for filling in self.classify.weight. 
            # Note, they are different! The first is the unnormalized, the 2nd is the normalized
            self.classify.weight[:] = F.normalize(self.classify_weights)

    def retrain(self, epochs, weights=None):
        
        """ Retrain with misclassified samples (also substract)"""

        misclassified_cnts = []
        acc_results = []
        
        for e in tqdm(range(epochs), desc="Epoch"):
            num_wrong = []

            with torch.no_grad():
                count = 0
                for it, batch in tqdm(enumerate(self.train_loader), desc=f"Retraining epoch {e}"):
                    
                    # if self.start_early_exit:
                    #     # print("Early exit started")
                    #     samples_hv, labels, _, logits = self.sample_to_encode(it, batch, step_type='retrain')
                    # else:
                        # print("Early exit not started")
                    if e >= epochs - len(self.stop):
                        samples_hv, labels, _, logits = self.sample_to_encode(it, batch, step_type="retrain")
                    else:
                        samples_hv, labels, _, logits = self.sample_to_encode(it, batch, step_type="train")
                    
                    is_wrong_count = 0
                    for b in range(0, samples_hv.shape[0], self.point_per_iter):
                        end = min(b + self.point_per_iter, int(samples_hv.shape[0]))  # Ensure we don't exceed num_voxels[i]
                        samples_hv_here = samples_hv[b:end]
                        labels_here = labels[b:end]
                        if logits != None:
                            logits_here = logits[b:end]
                        
                        #sim = self.model(samples_hv_here, dot=True)
                        #pred_hd = sim.argmax(1).data

                        # EDIT - normalize classify_weights for filling in self.classify.weight. 
                        # Need to do normalization right before the classification during retraining!
                        #self.classify.weight[:] = F.normalize(self.classify_weights)

                        # EDIT - with new classify
                        if logits == None:
                            self.classify.weight[:] = F.normalize(self.classify_weights)
                            logits_here = self.classify(F.normalize(samples_hv_here))
                        pred_hd = torch.argmax(logits_here, axis=1)

                        is_wrong = labels_here != pred_hd
                        is_wrong_count += is_wrong.sum().item()

                        # cancel update if all predictions were correct
                        if is_wrong.sum().item() == 0:
                            continue

                        count += is_wrong.sum().item()
                        
                        #print(f'Wrong predictions: {is_wrong.sum().item()}')

                        # only update wrongly predicted inputs
                        samples_hv_here = samples_hv_here[is_wrong]
                        labels_here = labels_here[is_wrong]
                        pred_hd = pred_hd[is_wrong]

                        #self.model.weight.index_add_(0, labels_here, samples_hv_here)
                        #self.model.weight.index_add_(0, pred_hd, samples_hv_here, alpha=-1.0)

                        ## EDIT - retraining with the new linear layer
                        #for i in range(len(labels_here)):
                        #    self.classify_weights[labels_here[i]] += samples_hv_here[i]
                        #    self.classify_weights[pred_hd[i]] -= samples_hv_here[i]
                        if weights is not None:
                            self.classify_weights.index_add_(0, labels_here, weights[labels_here].reshape(-1, 1) * samples_hv_here)
                            self.classify_weights.index_add_(0, pred_hd, - weights[pred_hd].reshape(-1, 1) * samples_hv_here)
                        else:
                            self.classify_weights.index_add_(0, labels_here, samples_hv_here)
                            self.classify_weights.index_add_(0, pred_hd, -samples_hv_here)
                    
                    num_wrong.append(is_wrong_count)
                    # self.mean_confidences[e, it] = torch.mean(logits)
                    # self.correct_percentages[e, it] = is_wrong_count/len(logits)

                    #torch.cuda.synchronize(device=self.device)

                    if it == self.max_samples:
                        break
                    ########## End of one scan

                ######### End of all scans
                if e >= epochs - len(self.stop):  # only after the LAST epoch
                    # print("Plotting exit value distribution after last epoch...")
                    # plot_exit_val_histogram(self.exit_val_dict, f'exit_val_hist{e}.png')
                    layer = self.stop[len(self.stop) - epochs + e]
                    vals_tensor = torch.tensor(self.exit_val_dict[layer])
                    new_threshold = torch.quantile(vals_tensor, self.quantile)
                    self.threshold[layer] = new_threshold
                    print(f"New threshold for layer {layer} in retraining epoch {e}: ", self.threshold)
                    print(f"Total exit_counter for retraining epoch {e}: ", self.exit_counter)
                    self.exit_val_dict = {}
                    self.exit_counter = {}
                    for i in self.stop:
                        self.exit_val_dict[int(i)] = []
                        self.exit_counter[int(i)] = 0
                    self.exit_counter[48] = 0

                # Print total misclassified samples in the current retraining epoch
                print("###########################")
                print(f"Total misclassified for retraining epoch {e}: ", count)
                print("###########################")

                misclassified_cnts.append(count)

            ######## End of one retraining epoch

            # Retraining test
            #if (e + 1) % 2 == 0:
            # hd_model.update = False
            avg_acc = self.test_hd()
            acc_results.append(avg_acc)
            # hd_model.update = True

        return acc_results, misclassified_cnts

    def test_hd(self, loader='val'):

        """ Testing over all the samples in all the scans given """

        if loader == 'val':
            loader = self.val_loader
            num_vox = self.num_vox_val
        else:
            loader = self.train_loader
            num_vox = self.num_vox_train
        
        # Metric
        miou = MulticlassJaccardIndex(num_classes=self.num_classes, average=None).to(self.device, non_blocking=True)
        final_labels = torch.empty((num_vox+1000), dtype=torch.int64, device=self.device)
        final_pred = torch.empty((num_vox+1000), dtype=torch.int64, device=self.device)
        soa_pred = torch.empty((num_vox+1000), dtype=torch.int64, device=self.device)
        
        start_idx = 0
        with torch.no_grad():
            for it, batch in tqdm(enumerate(loader), desc="Validation:"):
                samples_hv, labels, soa_labels, logits = self.sample_to_encode(it, batch, step_type='test') # Only return the features that haven't been dropped
                for b in range(0, samples_hv.shape[0], self.point_per_iter):
                    end = min(b + self.point_per_iter, int(samples_hv.shape[0]))  # Ensure we don't exceed num_voxels[i]
                    samples_hv_here = samples_hv[b:end]
                    labels_here = labels[b:end]
                    soa_here = soa_labels[b:end]  # EDIT: Add soa results
                    if logits != None:
                        logits_here = logits[b:end]

                    #torch.cuda.synchronize(device=self.device)
                
                    shape_sample = labels_here.shape[0]

                    #pred_hd = self.model(samples_hv, dot=True).argmax(1).data
                    #sim = self.model(samples_hv_here, dot=True)
                    #torch.cuda.synchronize(device=self.device)

                    ## EDIT - new classify
                    #logits = self.classify(F.normalize(samples_hv_here))

                    if logits == None:
                        logits_here = self.classify(F.normalize(samples_hv_here))

                    pred_hd = torch.argmax(logits_here, axis=1)
                    #torch.cuda.synchronize(device=self.device)

                    #print("Labels: ", labels.shape[0])
                    #print(start_idx, start_idx+shape_sample)
                    #print(shape_sample)

                    final_labels[start_idx:start_idx+shape_sample] = labels_here
                    final_pred[start_idx:start_idx+shape_sample] = pred_hd
                    soa_pred[start_idx:start_idx+shape_sample] = soa_here

                    start_idx += shape_sample

                if it == self.test_max_samples:
                    break

        final_labels = final_labels[:start_idx]
        final_pred = final_pred[:start_idx]
        soa_pred = soa_pred[:start_idx]

        print("================================")
        print("Plotting exit value distribution on test...")
        plot_exit_val_histogram(self.exit_val_dict, 'test_exit_val_hist.png')
        print(f"Threshold under test: ", self.threshold)
        print(f"Total exit_counter for test: ", self.exit_counter)
        self.exit_val_dict = {}
        self.exit_counter = {}
        for i in self.stop:
            self.exit_val_dict[int(i)] = []
            self.exit_counter[int(i)] = 0
        self.exit_counter[48] = 0

        #print('pred_ts', pred_ts)
        print('pred_hd', final_pred, "\tShape: ", final_pred.shape)
        print('label', final_labels, "\tShape: ", final_labels.shape)
        accuracy = miou(final_pred, final_labels)
        avg_acc = torch.mean(accuracy)
        print(f'accuracy: {accuracy}')
        print(f'avg acc: {avg_acc}')

        # if abs(avg_acc - self.past_acc) < 0.1 and self.start_early_exit == False:
        #     self.start_early_exit = True
        #     print("Start early exit")
        # else:
        #     self.past_acc = avg_acc
        if args.wandb_run:
            log_data = {f"Training class_{i}_IoU": c for i, c in enumerate(accuracy)}
            log_data["Retraining epoch"] = avg_acc
            wandb.log(log_data)

        #cm = confusion_matrix(pred_hd, first_label, labels=torch.Tensor(range(0,15)))
        #print("Confusion matrix \n")
        #print(cm)

        print("================================")

        return avg_acc.item()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-stops', '--layers', nargs='+', type=int, help='how many layers deep', default=[12, 24, 36])
    parser.add_argument('--confidence', type=float, help="Confidence threshold", default=1.0)
    #parser.add_argument('-soa', '--soa', action="store_true", default=False, help='Plot SOA')
    parser.add_argument('-number_samples', '--number_samples', type=int, help='how many scans to train', default=500)
    parser.add_argument('-test_number_samples', '--test_number_samples', type=int, help='how many scans to test', default=400)
    parser.add_argument("--seed", default=None, type=int, help="Seed for initializing training")
    #parser.add_argument("--add_lr", action="store_true", default=False, help='Add lr to help class imbalance')
    parser.add_argument("--dataset", choices=['nuscenes', 'semantic_kitti', 'tls'], default='nuscenes', help='Which dataset to train and test on?')
    parser.add_argument("--data_path", type=str, default='./root/main/dataset/', help='data dir path')
    parser.add_argument("--result_path", type=str, default='./results', help='result dir path')

    parser.add_argument("--wandb_run", action="store_true", default=False, help='Pass values to WandDB')
    parser.add_argument("--device", choices=['gpu', 'cpu'], default='gpu', help='Which device to use for training')
    parser.add_argument('--epochs', type=int, help='how many epochs to train', default=10)
    parser.add_argument('--subset', type=float, help='the ratio for dataset subset', default=1.0)

    # HD arguments
    parser.add_argument('--dim', type=int, help='fality of Hypervectors', default=10000)
    parser.add_argument('--batch_points', type=int, help='Number of points to process per scan', default=20000)
    parser.add_argument("--imbalance", action="store_true", default=False, help='Use imbalance weights')
    parser.add_argument("--quantile", type=float, default=0.8, help='Setup the quantile for the threshold')
    #parser.add_argument('-val', '--val', action="store_true", default=False, help='Train with validation for each scan')
    args = parser.parse_args()
    return args

def plot(acc_points, acc_results, misclassified_cnts, output_path):
    init_acc, final_acc = acc_points
    print(acc_results, misclassified_cnts)

    plt.figure()
    # Create the figure and the first axis
    fig, ax1 = plt.subplots()

    # Plot the first curve on the left y-axis
    x = np.arange(len(acc_results)+1)
    y1 = [init_acc] + acc_results
    ax1.plot(x, y1, 'b-*', label='mIoU', color='blue')
    ax1.set_xlabel('Retraining epochs')
    ax1.set_ylabel('mIoU', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Annotate the points on the first curve
    for i in range(len(x)):
        ax1.text(x[i], y1[i], f'{y1[i]:.2f}', color='blue', ha='center', va='bottom')

    # Create a second axis sharing the same x-axis
    ax2 = ax1.twinx()

    # Plot the second curve on the right y-axis
    x = np.arange(1, len(acc_results)+1)
    y2 = misclassified_cnts
    ax2.plot(x, y2, 'r-^', label='Misclassified Cnt', color='red')
    ax2.set_ylabel('Misclassified Cnt', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Annotate the points on the second curve
    for i in range(len(x)):
        ax2.text(x[i], y2[i], f'{y2[i]:.2f}', color='red', ha='center', va='top')

    # Show the plot
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(output_path, 'retraining.png'), dpi=300)

def plot_exit_val_histogram(exit_val_dict, save_path):
    plt.figure(figsize=(15, 4))
    for i, layer in enumerate(sorted(exit_val_dict.keys())):
        plt.subplot(1, 3, i+1)
        plt.hist(exit_val_dict[layer], bins=50, alpha=0.7)
        plt.title(f'Exit Layer {layer} Val Distribution')
        plt.xlabel('Confidence / Similarity Value')
        plt.ylabel('Count')
        plt.grid(True)
        if len(exit_val_dict[layer]) > 0:
            # percentile_95 = np.percentile(exit_val_dict[layer], 95)
            vals_tensor = torch.tensor(exit_val_dict[layer])
            percentile_95 = torch.quantile(vals_tensor, 0.95)
            plt.axvline(percentile_95, color='red', linestyle='dashed', linewidth=1.5)
            plt.text(percentile_95, plt.ylim()[1]*0.9, f'95%: {percentile_95:.2f}', color='red', rotation=90)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved exit value distribution histogram at {save_path}")

def plot_3d_graph(mean_confidences, correct_percentages, save_path="confidence_accuracy_3d.png"):
    # Get the dimensions of the arrays
    num_epochs = mean_confidences.shape[0]  # Number of epochs
    max_iterations = mean_confidences.shape[1]  # Number of iterations per epoch

    # Create a meshgrid for the iterations and epochs
    X, Y = np.meshgrid(np.arange(max_iterations), np.arange(num_epochs))

    # Plot Mean Confidence
    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, mean_confidences, cmap='coolwarm')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Epoch')
    ax1.set_zlabel('Mean Confidence')
    ax1.set_title('Mean Confidence per Iteration and Epoch')

    # Plot Correct Percentage
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, correct_percentages, cmap='coolwarm')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Epoch')
    ax2.set_zlabel('Correct Percentage')
    ax2.set_title('Correct Percentage per Iteration and Epoch')

    # Show and save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved 3D graph at {save_path}")



if __name__ == "__main__":
    
    args = parse_arguments()

    # Set seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)

    DIMENSIONS = args.dim
    FEAT_SIZE = 768
    NUM_LEVELS = 8000
    BATCH_SIZE = 1  # for GPUs with enough memory we can process multiple images at ones

    wandb.login(key="9487c04b8eff0c16cac4e785f2b57c3a475767d3")

    device = torch.device("cuda" if (torch.cuda.is_available() and args.device == 'gpu') else "cpu")
    print("Using {} device".format(device))
    device_string = "cuda:0" if (torch.cuda.is_available() and args.device == 'gpu') else "cpu"

    # Modify the path for each of the folders

    if args.dataset == 'nuscenes':
        path = os.path.join(args.data_path, 'nuscenes')
    elif args.dataset == 'semantic_kitti':
        path = os.path.join(args.data_path, 'semantickitti')
    elif args.dataset == 'tls':
        path = os.path.join(args.data_path, 'tls')


    # Get datatset
    DATASET = LIST_DATASETS.get(args.dataset)
    
    ##### Process dataset #######

    if args.dataset == 'nuscenes':

        kwargs = {
            "rootdir": path,
            "input_feat": ["intensity", "xyz", "radius"],
            "voxel_size": 0.1,
            "num_neighbors": 16,
            "dim_proj": [2, 1, 0],
            "grids_shape": [[256, 256], [256, 32], [256, 32]],
            "fov_xyz": [[-64, -64, -8], [64, 64, 8]], # Check here
        }

        # Train dataset
        dataset = DATASET(
            phase="train",
            **kwargs,
        )

        dataset_train = copy.deepcopy(dataset)
        dataset_val = copy.deepcopy(dataset)
        dataset_train.init_training()
        dataset_val.init_val()

        num_classes = 16

        path_pretrained = '/root/main/ScaLR/saved_models/ckpt_last_scalr.pth'
    
    elif args.dataset == 'semantic_kitti':

        kwargs = {
            "rootdir": path,
            "input_feat": ["intensity", "xyz", "radius"],
            "voxel_size": 0.1,
            "num_neighbors": 16,
            "dim_proj": [2, 1, 0],
            "grids_shape": [[256, 256], [256, 32], [256, 32]],
            "fov_xyz": [[-64, -64, -8], [64, 64, 8]], # Check here
        }
        
        dataset_train = DATASET(
            phase="specific_train",
            **kwargs,
        )

        # Validation dataset
        dataset_val = DATASET(
            phase="val",
            **kwargs,
        )

        num_classes = 19

        path_pretrained = '/root/main/ScaLR/saved_models/ckpt_last_kitti.pth'
    
    else:
        raise Exception("Dataset Not identified")
    
    # Use subsets of dataset
    if args.subset < 1.0 and args.subset > 0:
        subset_len = int(len(dataset_train) * args.subset)
        dataset_train, _ = torch.utils.data.random_split(dataset=dataset_train,
                                                        lengths=[subset_len, len(dataset_train) - subset_len])

        subset_len = int(len(dataset_val) * 0.01)
        dataset_val, _ = torch.utils.data.random_split(dataset=dataset_val,
                                                    lengths=[subset_len, len(dataset_val) - subset_len])

    # Temporal edits - all use training dataset
    # subset_len = int(len(dataset_train) * 0.8)
    # dataset_train, dataset_val = torch.utils.data.random_split(dataset=dataset_train,
    #                                                           lengths=[subset_len, len(dataset_train) - subset_len])
        
    print(f'train dataset length: {len(dataset_train)}')
    print(f'val dataset length: {len(dataset_val)}')
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        collate_fn=Collate(device=device),
        persistent_workers=False,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        collate_fn=Collate(device=device),
        persistent_workers=False,
    )

    ####### Preprocesing for the weights per class ##########
    weights = None
    if args.imbalance:
        weights = torch.zeros(num_classes)
        print('init weights', weights)
        for it, batch in tqdm(enumerate(train_loader), desc="Preprocessing:"):
            labels = batch["labels_orig"]
            labels = labels[labels != 255]
            tmp_cnt = torch.bincount(labels)
            #print('tmp count', tmp_cnt)
            
            delta = len(weights) - len(tmp_cnt)
            if delta > 0:
                tmp_cnt = torch.cat((tmp_cnt, torch.zeros(delta)), dim=0)

            # Update weights
            weights += tmp_cnt

            if it == args.number_samples:
                break

        # Compute the weights during training to remove imbalance
        weights = 1.0 - weights / weights.sum()
        weights = weights.to(device)
        print('class imbalance weights: ', weights)

    ####### HD Model ##########
    hd_model = HD_Model(FEAT_SIZE, DIMENSIONS, num_classes, path_pretrained, device=device, args=args)
    hd_model.set_loaders(train_loader=train_loader, val_loader=val_loader)
    # hd_model.feature_extractor.model.set_compensation({12: '/home/HyperLiDAR/overcompensation_layer/linear_weights_12_0.75_normalize.pth', 
        # 24: '/home/HyperLiDAR/overcompensation_layer/linear_weights_24_0.75_normalize.pth', 
        # 36: '/home/HyperLiDAR/overcompensation_layer/linear_weights_36_1.75_normalize_2.pth'}, device=device)

    if args.wandb_run:
        run = wandb.init(
            # Set the project where this run will be logged
            project="scalr_hd",
            # Track hyperparameters and run metadata
            config={
                "encoding": "Random Projection",
                "hd_dim": DIMENSIONS,
                "training_samples":args.number_samples,
            },
            id=f"{args.dataset}_training_layers_{args.layers}_norm_dim_{DIMENSIONS}_OFA_early_exit",
        )
    ####### Results dir setup ##########
    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)
    model_name = f"{args.dataset}_{args.subset}_{args.number_samples}_{args.test_number_samples}_nn{args.layers}_" \
                 f"hd{FEAT_SIZE}_{DIMENSIONS}_{args.epochs}_{args.batch_points}_imb{int(args.imbalance)}_ee" \
                 f"seed{args.seed}"
    output_path = os.path.join(args.result_path, model_name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    ####### HD Pipeline ##########

    print("Initial Training")
    start = time.time()
    hd_model.train(weights=weights)
    end = time.time()
    total_time = (end-start)
    print(f"Training total_time: {total_time}")

    print("Testing")
    start = time.time()
    init_acc = hd_model.test_hd()
    end = time.time()
    total_time = (end-start)
    print(f"Testing total_time: {total_time}")

    print("Retraining")
    # hd_model.update = True
    start = time.time()
    acc_results, misclassified_cnts = hd_model.retrain(epochs=args.epochs, weights=weights)
    end = time.time()
    total_time = (end-start) 
    print(f"Retraining total_time: {total_time}")

    # plot_3d_graph(hd_model.mean_confidences, hd_model.correct_percentages, save_path='confidence_accuracy_3d.png')
    
    print("Testing")
    start = time.time()
    final_acc = hd_model.test_hd()
    end = time.time()
    total_time = (end-start)
    print(f"Testing total_time: {total_time}")

    plot((init_acc, final_acc), acc_results, misclassified_cnts, output_path)


    ####### SOA results ##########
    #print("SoA results")

    #hd_model.feature_extractor.test(hd_model.val_loader, hd_model.num_vox_val+1000, 48)
