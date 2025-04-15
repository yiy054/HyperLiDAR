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
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    def __init__(self, hd_dim, size):
        super(Encoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.projection = embeddings.Projection(size, hd_dim)

        ## EDIT - remove this line, not sure what's the point
        #self.projection.weight = nn.Parameter(torchhd.normalize(self.projection.weight), requires_grad=False) # Binary

    def forward(self, x):
        sample_hv = self.projection(x)
        return torchhd.hard_quantize(sample_hv)

class Feature_Extractor:
    def __init__(self, input_channels=5, feat_channels=768, depth=48, 
                 grid_shape=[[256, 256], [256, 32], [256, 32]], nb_class=16, layer_norm=True, 
                 device=torch.device("cpu"), early_exit = 48, **kwargs):
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

    def forward_model(self, it, batch, start=0, stop=48):

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
        net_inputs = (feat, cell_ind, occupied_cell, neighbors_emb)

        if self.device_string != 'cpu':
            with torch.autocast("cuda", enabled=True):
                # Logits
                with torch.no_grad():
                    out = self.model(*net_inputs)
                    encode, tokens, out = out[0], out[1], out[2]
                    pred_label = out.max(1)[1]

                    # Only return samples that are not noise
                    #torch.cuda.synchronize(device=self.device)
                    where = labels != 255
                    #torch.cuda.synchronize(device=self.device)
        else:
            with torch.no_grad():
                out = self.model(*net_inputs)
                encode, tokens, out = out[0], out[1], out[2]
                pred_label = out.max(1)[1]

                # Only return samples that are not noise
                where = labels != 255

        return tokens[0,:,where], labels[where], pred_label[0, where]

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
        self.feature_extractor = Feature_Extractor(nb_class = num_classes, device=self.device, early_exit=kwargs['args'].layers, args=kwargs['args'])
        self.feature_extractor.load_pretrained(path_pretrained)
        self.stop = kwargs['args'].layers
        self.point_per_iter = kwargs['args'].number_samples
        self.num_classes = num_classes
        self.max_samples = kwargs['args'].number_samples
        self.kwargs = kwargs

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
    
    def sample_to_encode(self, it, batch, stop_layer=48):
        features, labels, soa_labels = self.feature_extractor.forward_model(it, batch, stop_layer) # Everything for what hasn't been dropped
        features = torch.transpose(features, 0, 1).to(dtype=torch.float32, device = self.device, non_blocking=True)
        labels = labels.to(dtype=torch.int64, device = self.device, non_blocking=True)

        #features = self.normalize(features) # Z1 score seems to work

        # HD training
        samples_hv = self.encode(features)

        return samples_hv, labels, soa_labels
    
    def train(self, weights=None):

        """ Initial training pass """

        print("\nTrain First\n")

        with torch.no_grad():
            for it, batch in tqdm(enumerate(self.train_loader), desc="Training"):
    
                samples_hv, labels, _ = self.sample_to_encode(it, batch)
                
                for b in range(0, samples_hv.shape[0], self.point_per_iter):
                    end = min(b + self.point_per_iter, int(samples_hv.shape[0]))  # Ensure we don't exceed num_voxels[i]
            
                    #samples_hv = samples_hv.reshape((1,samples_hv.shape[0]))
                    
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
        
        for e in range(epochs):
            #self.scramble = np.random.permutation(len(self.im_idx))

            with torch.no_grad():
                count = 0
                for it, batch in tqdm(enumerate(self.train_loader), desc=f"Retraining epoch {e}"):
                    
                    samples_hv, labels, _ = self.sample_to_encode(it, batch)

                    for b in range(0, samples_hv.shape[0], self.point_per_iter):
                        end = min(b + self.point_per_iter, int(samples_hv.shape[0]))  # Ensure we don't exceed num_voxels[i]
                        samples_hv_here = samples_hv[b:end]
                        labels_here = labels[b:end]
                        
                        #sim = self.model(samples_hv_here, dot=True)
                        #pred_hd = sim.argmax(1).data

                        # EDIT - normalize classify_weights for filling in self.classify.weight. 
                        # Need to do normalization right before the classification during retraining!
                        self.classify.weight[:] = F.normalize(self.classify_weights)

                        # EDIT - with new classify
                        logits = self.classify(F.normalize(samples_hv_here))
                        pred_hd = torch.argmax(logits, axis=1)

                        is_wrong = labels_here != pred_hd

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

                    #torch.cuda.synchronize(device=self.device)

                    if it == self.max_samples:
                        break
                    ########## End of one scan

                ######### End of all scans

                # Print total misclassified samples in the current retraining epoch
                print("###########################")
                print(f"Total misclassified for retraining epoch {e}: ", count)

            ######## End of one retraining epoch

            # Retraining test
            if (e + 1) % 2 == 0:
                self.test_hd()

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
        
                samples_hv, labels, soa_labels = self.sample_to_encode(it, batch) # Only return the features that haven't been dropped
                
                for b in range(0, samples_hv.shape[0], self.point_per_iter):
                    end = min(b + self.point_per_iter, int(samples_hv.shape[0]))  # Ensure we don't exceed num_voxels[i]
                    samples_hv_here = samples_hv[b:end]
                    labels_here = labels[b:end]
                    soa_here = soa_labels[b:end]  # EDIT: Add soa results

                    #torch.cuda.synchronize(device=self.device)
                
                    shape_sample = labels_here.shape[0]

                    #pred_hd = self.model(samples_hv, dot=True).argmax(1).data
                    #sim = self.model(samples_hv_here, dot=True)
                    #torch.cuda.synchronize(device=self.device)

                    ## EDIT - new classify
                    logits = self.classify(F.normalize(samples_hv_here))

                    pred_hd = torch.argmax(logits, axis=1)
                    #torch.cuda.synchronize(device=self.device)

                    #print("Labels: ", labels.shape[0])
                    #print(start_idx, start_idx+shape_sample)
                    #print(shape_sample)

                    final_labels[start_idx:start_idx+shape_sample] = labels_here
                    final_pred[start_idx:start_idx+shape_sample] = pred_hd
                    soa_pred[start_idx:start_idx+shape_sample] = soa_here

                    start_idx += shape_sample

                if it == self.max_samples:
                    break

        final_labels = final_labels[:start_idx]
        final_pred = final_pred[:start_idx]
        soa_pred = soa_pred[:start_idx]

        print("================================")

        #print('pred_ts', pred_ts)
        print('pred_hd', final_pred, "\tShape: ", final_pred.shape)
        print('label', final_labels, "\tShape: ", final_labels.shape)
        accuracy = miou(final_pred, final_labels)
        avg_acc = torch.mean(accuracy)
        print(f'accuracy: {accuracy}')
        print(f'avg acc: {avg_acc}')

        ## EDIT: Report soa accuracy
        accuracy = miou(soa_pred, final_labels)
        avg_acc = torch.mean(accuracy)
        print(f'soa accuracy: {accuracy}')
        print(f'avg soa acc: {avg_acc}')

        if args.wandb_run:
            log_data = {f"Training class_{i}_IoU": c for i, c in enumerate(accuracy)}
            log_data["Retraining epoch"] = avg_acc
            wandb.log(log_data)

        #cm = confusion_matrix(pred_hd, first_label, labels=torch.Tensor(range(0,15)))
        #print("Confusion matrix \n")
        #print(cm)

        print("================================")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-stops', '--layers', nargs='+', type=int, help='how many layers deep', default=[48])
    parser.add_argument('--confidence', type=float, help="Confidence threshold", default=1.0)
    #parser.add_argument('-soa', '--soa', action="store_true", default=False, help='Plot SOA')
    parser.add_argument('-number_samples', '--number_samples', type=int, help='how many scans to train', default=500)
    parser.add_argument(
            "--seed", default=None, type=int, help="Seed for initializing training"
        )
    parser.add_argument(
            "--add_lr", action="store_true", default=False, help='Add lr to help class imbalance'
        )
    parser.add_argument(
            "--dataset", choices=['nuscenes', 'semantic_kitti', 'tls'], default='nuscenes', help='Which dataset to train and test on?'
        )
    parser.add_argument(
            "--data_path", type=str, default='./root/main/dataset/', help='data dir path'
        )
    parser.add_argument("--imbalance", action="store_true", default=False, help='Use imbalance weights')

    parser.add_argument("--wandb_run", action="store_true", default=False, help='Pass values to WandDB')
    parser.add_argument("--device", choices=['gpu', 'cpu'], default='gpu', help='Which device to use for training')
    parser.add_argument('--epochs', type=int, help='how many epochs to train', default=10)
    parser.add_argument('--subset', type=float, help='the ratio for dataset subset', default=1.0)

    # HD arguments
    parser.add_argument('--dim', type=int, help='Dimensionality of Hypervectors', default=10000)
    parser.add_argument('--batch_points', type=int, help='Number of points to process per scan', default=20000)
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
    
    # Temporary edit - use subsets of dataset
    if args.subset < 1.0 and args.subset > 0:
        subset_len = int(len(dataset_train) * args.subset)
        dataset_train, _ = torch.utils.data.random_split(dataset=dataset_train,
                                                        lengths=[subset_len, len(dataset_train) - subset_len])

        subset_len = int(len(dataset_val) * 0.01)
        dataset_val, _ = torch.utils.data.random_split(dataset=dataset_val,
                                                    lengths=[subset_len, len(dataset_val) - subset_len])
        
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
            id=f"{args.dataset}_training_layers_{args.layers}_norm_dim_{DIMENSIONS}",
        )

    ####### HD Pipeline ##########

    print("Initial Training")
    hd_model.train(weights=weights)

    print("Testing")
    hd_model.test_hd()

    print("Retraining")
    hd_model.retrain(epochs=args.epochs, weights=weights)
    
    print("Testing")
    hd_model.test_hd()

    plot((init_acc, final_acc), acc_results, misclassified_cnts, output_path)
    ####### SOA results ##########
    #print("SoA results")

    #hd_model.feature_extractor.test(hd_model.val_loader, hd_model.num_vox_val+1000, 48)from models.waffleiron.segmenter import Segmenter
