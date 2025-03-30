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

from timm.models.layers import _assert, trunc_normal_

import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

def ofa_loss(logits_student, logits_teacher, target_mask, eps, temperature=1.):
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    prod = (pred_teacher + target_mask) ** eps
    loss = torch.sum(- (prod - target_mask) * torch.log(pred_student), dim=-1)
    return loss.mean()

def init_weights(module):
    for n, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

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
        net_inputs = (feat, cell_ind, occupied_cell, neighbors_emb)

        if self.device_string != 'cpu':
            with torch.autocast("cuda", enabled=True):
                # Logits
                with torch.no_grad():
                    out = self.model(*net_inputs, step_type)
                    encode, tokens, out, exit_layer = out[0], out[1], out[2], out[3]
                    pred_label = out.max(1)[1]

                    # Only return samples that are not noise
                    #torch.cuda.synchronize(device=self.device)
                    where = labels != 255
                    #torch.cuda.synchronize(device=self.device)
        else:
            with torch.no_grad():
                out = self.model(*net_inputs, step_type)
                encode, tokens, out, exit_layer = out[0], out[1], out[2], out[3]
                pred_label = out.max(1)[1]

                # Only return samples that are not noise
                where = labels != 255

        return tokens[0,:,where], labels[where], pred_label[0, where], exit_layer
            

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
    parser.add_argument("--data_path", type=str, default='/mnt/data/', help='data dir path')
    parser.add_argument("--result_path", type=str, default='./results', help='result dir path')

    parser.add_argument("--wandb_run", action="store_true", default=False, help='Pass values to WandDB')
    parser.add_argument("--device", choices=['gpu', 'cpu'], default='gpu', help='Which device to use for training')
    parser.add_argument('--epochs', type=int, help='how many epochs to train', default=10)
    parser.add_argument('--subset', type=float, help='the ratio for dataset subset', default=1.0)

    # HD arguments
    parser.add_argument('--dim', type=int, help='fality of Hypervectors', default=10000)
    parser.add_argument('--batch_points', type=int, help='Number of points to process per scan', default=20000)
    parser.add_argument("--imbalance", action="store_true", default=False, help='Use imbalance weights')
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
    #subset_len = int(len(dataset_train) * 0.8)
    #dataset_train, dataset_val = torch.utils.data.random_split(dataset=dataset_train,
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

    feature_extractor_complete = Feature_Extractor(nb_class = num_classes, device=device, early_exit=[48], args=args)
    feature_extractor_complete.load_pretrained(path_pretrained)

    feature_extractor_small = Feature_Extractor(nb_class = num_classes, device=device, early_exit=[36], args=args)
    feature_extractor_small.load_pretrained(path_pretrained)

    linear = nn.Linear(768, 768)

    optimizer = optim.SGD(linear.parameters(), lr=0.01)  # Optimizing only the Linear layer

    # Initialize the first layer -> Second one keep it intact
    nn.init.zeros_(linear.bias)
    trunc_normal_(linear.weight, std=.02)
    linear.to(device)

    loss_epochs = []
    for e in range(10):
        loss_epoch = []
        for it, batch in tqdm(enumerate(train_loader), desc=f"Transfer Learning at epoch {e}: "):
            features_complete, labels, soa_result_complete, exit_layer = feature_extractor_complete.forward_model(it, batch, step_type = "distill")
            features_small, _, _, exit_layer_small = feature_extractor_small.forward_model(it, batch, step_type = "distill")
            
            linear_output = linear(torch.transpose(features_small, 0, 1).to(torch.float32))
            linear_output = torch.reshape(linear_output, (1, linear_output.shape[1], linear_output.shape[0]))

            tokens_student = feature_extractor_small.model.classif(linear_output)

            features_complete = torch.reshape(features_complete, (1, features_complete.shape[0], features_complete.shape[1])).to(torch.float32)
            tokens_teacher = feature_extractor_complete.model.classif(features_complete)

            target_mask = F.one_hot(labels, num_classes).to(device)

            loss = ofa_loss(torch.transpose(tokens_student[0], 0, 1), torch.transpose(tokens_teacher[0], 0, 1), target_mask, eps=1.75)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.save(linear.state_dict(), 'linear_weights_36_1.75_normalize_2.pth')

            loss_epoch.append(float(loss.cpu()))
        loss_at_epoch = np.mean(np.array(loss_epoch))
        print(f"Loss at epoch {e} = {loss_at_epoch}")
        loss_epochs.append(loss_at_epoch)

    # Create a range of epochs (assuming the loss array corresponds to these epochs)
    epochs = np.arange(1, len(loss_epochs) + 1)

    # Plot the loss values
    plt.plot(epochs, loss_epochs, marker='o', color='b', label='Loss')

    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')

    # Show the legend
    plt.legend()

    # Save the plot as an image file
    plt.savefig('loss_per_epoch.png')
    








