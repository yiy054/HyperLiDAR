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
from ofa import OFA
from timm.models import convert_splitbn_model, create_model, model_parameters, safe_model_name, load_checkpoint
from timm.loss import *
from timm.optim import create_optimizer_v2, optimizer_kwargs

import argparse
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.utils
import yaml
from timm.data import AugMixDataset, create_dataset, create_loader, FastCollateMixup, Mixup, \
    resolve_data_config
from timm.loss import *
from timm.models import convert_splitbn_model, create_model, model_parameters, safe_model_name, load_checkpoint
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import *
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from custom_forward import register_new_forward
from distillers import get_distiller
from utils import CIFAR100InstanceSample, ImageNetInstanceSample, TimePredictor
from custom_model import *

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

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
                    out = self.model(*net_inputs, stop=stop)
                    encode, tokens, out = out[0], out[1], out[2]
                    pred_label = out.max(1)[1]

                    # Only return samples that are not noise
                    #torch.cuda.synchronize(device=self.device)
                    where = labels != 255
                    #torch.cuda.synchronize(device=self.device)
        else:
            with torch.no_grad():
                out = self.model(*net_inputs, stop=stop)
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
    parser.add_argument("--wandb_run", action="store_true", default=False, help='Pass values to WandDB')
    parser.add_argument("--device", choices=['gpu', 'cpu'], default='gpu', help='Which device to use for training')

    # HD arguments
    parser.add_argument('--dim', type=int, help='Dimensionality of Hypervectors', default=10000)
    parser.add_argument('--batch_points', type=int, help='Number of points to process per scan', default=20000)
    #parser.add_argument('-val', '--val', action="store_true", default=False, help='Train with validation for each scan')

    # ------------------------------------- My params ---------------------------------------
    # Basic parameters
    parser.add_argument('--model', default='resnet18', type=str)
    parser.add_argument('--teacher', default='deit_tiny_patch16_224', type=str)
    parser.add_argument('--teacher-pretrained', default='', type=str)
    parser.add_argument('--use-ema-teacher', action='store_true')

    parser.add_argument('--distiller', default='ofa', type=str)
    parser.add_argument('--gt-loss-weight', default=1., type=float)
    parser.add_argument('--kd-loss-weight', default=1., type=float)

    # KD parameters
    parser.add_argument('--kd-temperature', default=1, type=float)

    # OFA parameters
    parser.add_argument('--ofa-eps', default=[1], nargs='+', type=float)
    parser.add_argument('--ofa-stage', default=[1, 2, 3, 4], nargs='+', type=int)
    parser.add_argument('--ofa-loss-weight', default=1, type=float)
    parser.add_argument('--ofa-temperature', default=1, type=float)

    # DIST parameters
    parser.add_argument('--dist-beta', default=1, type=float)
    parser.add_argument('--dist-gamma', default=1, type=float)
    parser.add_argument('--dist-tau', default=1, type=float)

    # DKD parameters
    parser.add_argument('--dkd-alpha', default=1, type=float)
    parser.add_argument('--dkd-beta', default=2, type=float)
    parser.add_argument('--dkd-temperature', default=1, type=float)

    # Correlation parameters
    parser.add_argument('--correlation-scale', default=0.02, type=float)
    parser.add_argument('--correlation-feat-dim', default=128, type=int)

    # CRD parameters
    parser.add_argument('--crd-feat-dim', default=128, type=int)
    parser.add_argument('--crd-k', default=16384, type=int)
    parser.add_argument('--crd-momentum', default=0.5, type=float)
    parser.add_argument('--crd-temperature', default=0.07, type=float)

    # RKD parameters
    parser.add_argument('--rkd-distance-weight', default=25, type=float)
    parser.add_argument('--rkd-angle-weight', default=50, type=float)
    parser.add_argument('--rkd-eps', default=1e-12, type=float)
    parser.add_argument('--rkd-squared', action='store_true', default=False)

    # FitNet parameters
    parser.add_argument('--fitnet-stage', default=[1, 2, 3, 4], nargs='+', type=int)
    parser.add_argument('--fitnet-loss-weight', default=1, type=float)

    # Misc
    parser.add_argument('--speedtest', action='store_true')

    parser.add_argument('--eval-interval', default=1, type=int)  # eval every 1 epochs before epochs * eval_interval_end
    parser.add_argument('--eval-interval-end', default=0.75, type=float)
    # ---------------------------------------------------------------------------------------

    # Dataset parameters
    parser.add_argument('data_dir', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                        help='dataset type (default: ImageFolder/ImageTar if empty)')
    parser.add_argument('--train-split', metavar='NAME', default='train',
                        help='dataset train split (default: train)')
    parser.add_argument('--val-split', metavar='NAME', default='validation',
                        help='dataset validation split (default: validation)')
    parser.add_argument('--dataset-download', action='store_true', default=False,
                        help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
    parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                        help='path to class to idx mapping file (default: "")')

    # Model parameters
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Start with pretrained version of specified network (if avail)')
    parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                        help='Initialize model from this checkpoint (default: none)')
    parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                        help='number of label classes (Model default if None)')
    parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                        help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
    parser.add_argument('--img-size', type=int, default=None, metavar='N',
                        help='Image patch size (default: None => model default)')
    parser.add_argument('--input-size', default=None, nargs=3, type=int,
                        metavar='N N N',
                        help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
    parser.add_argument('--crop-pct', default=None, type=float,
                        metavar='N', help='Input image center crop percent (for validation only)')
    parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                        help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                        help='Override std deviation of dataset')
    parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                        help='Image resize interpolation type (overrides model)')
    parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                        help='Input batch size for training (default: 128)')
    parser.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                        help='Validation batch size override (default: None)')

    # Optimizer parameters
    parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "sgd"')
    parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: None, use opt default)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=2e-5,
                        help='weight decay (default: 2e-5)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip-mode', type=str, default='norm',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--layer-decay', type=float, default=None,
                        help='layer-wise learning rate decay (default: None)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "step"')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.05)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                        help='amount to decay each learning rate cycle (default: 0.5)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                        help='learning rate cycle limit, cycles enabled if > 1')
    parser.add_argument('--lr-k-decay', type=float, default=1.0,
                        help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                        help='warmup learning rate (default: 0.0001)')
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                        help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation & regularization parameters
    parser.add_argument('--no-aug', action='store_true', default=False,
                        help='Disable all training augmentation, override other train aug args')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--hflip', type=float, default=0.5,
                        help='Horizontal flip training aug probability')
    parser.add_argument('--vflip', type=float, default=0.,
                        help='Vertical flip training aug probability')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". (default: None)'),
    parser.add_argument('--aug-repeats', type=float, default=0,
                        help='Number of augmentation repetitions (distributed training only) (default: 0)')
    parser.add_argument('--aug-splits', type=int, default=0,
                        help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
    parser.add_argument('--jsd-loss', action='store_true', default=False,
                        help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
    parser.add_argument('--bce-loss', action='store_true', default=False,
                        help='Enable BCE loss w/ Mixup/CutMix use.')
    parser.add_argument('--bce-target-thresh', type=float, default=None,
                        help='Threshold for binarizing softened BCE targets (default: None, disabled)')
    parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                        help='Random erase prob (default: 0.)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                        help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='random',
                        help='Training interpolation (random, bilinear, bicubic default: "random")')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                        help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
    parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                        help='Drop path rate (default: None)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    # Batch norm parameters (only works with gen_efficientnet based models currently)
    parser.add_argument('--bn-momentum', type=float, default=None,
                        help='BatchNorm momentum override (if not None)')
    parser.add_argument('--bn-eps', type=float, default=None,
                        help='BatchNorm epsilon override (if not None)')
    parser.add_argument('--sync-bn', action='store_true',
                        help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
    parser.add_argument('--dist-bn', type=str, default='reduce',
                        help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
    parser.add_argument('--split-bn', action='store_true',
                        help='Enable separate BN layers per augmentation split.')

    # Model Exponential Moving Average
    parser.add_argument('--model-ema', action='store_true', default=False,
                        help='Enable tracking moving average of model weights')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                        help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    parser.add_argument('--model-ema-decay', type=float, default=[0.9998], nargs='+',
                        help='decay factor for model weights moving average (default: 0.9998)')

    # Misc
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--worker-seeding', type=str, default='all',
                        help='worker seed mode (default: all)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                        help='how many batches to wait before writing recovery checkpoint')
    parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                        help='number of checkpoints to keep (default: 10)')
    parser.add_argument('-j', '--workers', type=int, default=8, metavar='N',
                        help='how many training processes to use (default: 4)')
    parser.add_argument('--save-images', action='store_true', default=False,
                        help='save images of input bathes every log interval for debugging')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
    parser.add_argument('--apex-amp', action='store_true', default=False,
                        help='Use NVIDIA Apex AMP mixed precision')
    parser.add_argument('--native-amp', action='store_true', default=False,
                        help='Use Native Torch AMP mixed precision')
    parser.add_argument('--no-ddp-bb', action='store_true', default=False,
                        help='Force broadcast buffers for native DDP to off.')
    parser.add_argument('--pin-mem', action='store_true', default=False,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-prefetcher', action='store_true', default=False,
                        help='disable fast prefetcher')
    parser.add_argument('--output', default='', type=str, metavar='PATH',
                        help='path to output folder (default: none, current dir)')
    parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                        help='name of train experiment, name of sub-folder for output')
    parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                        help='Best metric (default: "top1"')
    parser.add_argument('--tta', type=int, default=0, metavar='N',
                        help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                        help='use the multi-epochs-loader to save time at the beginning of every epoch')

    args = parser.parse_args()
    return args

def train_one_epoch(
        epoch, distiller, loader, optimizer, args,
        lr_scheduler=None, saver=None, output_dir=None, amp_autocast=suppress,
        loss_scaler=None, model_emas=None, mixup_fn=None):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    losses_gt_m = AverageMeter()
    losses_kd_m = AverageMeter()

    from collections import defaultdict
    losses_m_dict = defaultdict(AverageMeter)

    distiller.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target, *additional_input) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            additional_input = [i.cuda() for i in additional_input]
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)

        with amp_autocast():

            output, losses_dict = distiller(input, target, *additional_input, epoch=epoch)
            loss = sum(losses_dict.values())

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))
            for k in losses_dict:
                losses_m_dict[k].update(losses_dict[k].item(), input.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(distiller, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(distiller, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            optimizer.step()

        if model_emas is not None:
            for ema, _ in model_emas:
                if hasattr(distiller, 'module'):
                    ema.update(distiller.module.student)
                else:
                    ema.update(distiller.student)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                reduced_loss_dict = {}
                for k in losses_dict:
                    reduced_loss_dict[k] = reduce_tensor(losses_dict[k].data, args.world_size)

                losses_m.update(reduced_loss.item(), input.size(0))
                for k in reduced_loss_dict:
                    losses_m_dict[k].update(reduced_loss_dict[k].item(), input.size(0))

            if args.rank == 0:
                losses_infos = []
                for k, v in losses_m_dict.items():
                    info = f'{k.capitalize()}: {v.val:#.4g} ({v.avg:#.3g})'
                    losses_infos.append(info)
                losses_info = '  '.join(losses_infos)

                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                    '{losses_info} '
                    'LR: {lr:.3e}'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        loss_gt=losses_gt_m,
                        loss_kd=losses_kd_m,
                        losses_info=losses_info,
                        lr=lr))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics

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
        path = '/root/main/dataset/nuscenes'
    elif args.dataset == 'semantic_kitti':
        path = '/root/main/dataset/semantickitti'
    elif args.dataset == 'tls':
        path = '/root/main/dataset/tls'


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
    
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=1,
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

    ### Distiller ####

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    Distiller = OFA('ofa')

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        start_epoch = args.start_epoch

    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    ####### Setup teacher model ##########

    feature_extractor = Feature_Extractor(nb_class = num_classes, device=device, early_exit=kwargs['args'].layers, args=kwargs['args'])
    feature_extractor.load_pretrained(path_pretrained)

    ########## Student model ##########

    model = create_model(
        'resmlp_12_224',
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        checkpoint_path=args.initial_checkpoint)
    
    # setup loss function
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()
    
    distiller = Distiller(model, teacher=feature_extractor, criterion=train_loss_fn, args=args, num_data=len(dataset_train))
    distiller = distiller.cuda()

    optimizer = create_optimizer_v2(distiller, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        distiller, optimizer = amp.initialize(distiller, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.rank == 0:
            _logger.info('AMP not enabled. Training in float32.')

    # setup exponential moving average of model weights, SWA could be used here too
    model_emas = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_emas = []
        for decay in args.model_ema_decay:
            model_ema = ModelEmaV2(model, decay=decay, device='cpu' if args.model_ema_force_cpu else None)
            model_emas.append((model_ema, decay))

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)
    
    ###### Passing the dataset ######

    for epoch in range(start_epoch, num_epochs):
        for it, batch in tqdm(enumerate(train_loader), desc="Training"):
            features, labels, soa_result = feature_extractor.forward_model(it, batch) # Everything for what hasn't been dropped
            train_metrics = train_one_epoch(
                    epoch, distiller, train_loader, optimizer, args,
                    lr_scheduler=lr_scheduler, output_dir='/home/output_dist',
                    amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_emas=model_emas, mixup_fn=mixup_fn)

            is_eval = epoch > int(args.eval_interval_end * args.epochs) or epoch % args.eval_interval == 0
            if not args.speedtest and is_eval:
                eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)

                if saver is not None:
                    # save proper checkpoint with eval metric
                    save_metric = eval_metrics[eval_metric]
                    best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

                if model_emas is not None and not args.model_ema_force_cpu:
                    for j, ((ema, decay), ema_saver) in enumerate(zip(model_emas, ema_savers)):

                        if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                            distribute_bn(ema, args.world_size, args.dist_bn == 'reduce')

                        ema_eval_metrics = validate(ema.module, loader_eval, validate_loss_fn, args,
                                                    amp_autocast=amp_autocast, log_suffix=f' (EMA {decay:.5f})')

                        if ema_saver is not None:
                            # save proper checkpoint with eval metric
                            save_metric = ema_eval_metrics[eval_metric]
                            ema_saver.save_checkpoint(epoch, metric=save_metric)

                if output_dir is not None:
                    update_summary(
                        epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                        write_header=best_metric is None)

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            tp.update()
            if args.rank == 0:
                print(f'Will finish at {tp.get_pred_text()}')
                print(f'Avg running time of latest {len(tp.time_list)} epochs: {np.mean(tp.time_list):.2f}s/ep.')

