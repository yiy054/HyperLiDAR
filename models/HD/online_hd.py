import torch
import torchhd
from torchhd.classifiers import Classifier
from typing import Optional, Callable, Iterable, Tuple
from typing_extensions import Literal

import math
import scipy.linalg
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, LongTensor

import torchhd.functional as functional
from torchhd.embeddings import Random, Level, Projection, Sinusoid, Density
from torchhd.models import Centroid, IntRVFL as IntRVFLModel

DataLoader = Iterable[Tuple[Tensor, LongTensor]]

# Adapted from: https://gitlab.com/biaslab/onlinehd/
class OnlineHD(Classifier):
    r"""Implements `OnlineHD: Robust, Efficient, and Single-Pass Online Learning Using Hyperdimensional System <https://ieeexplore.ieee.org/abstract/document/9474107>`_.

    Args:
        n_features (int): Size of each input sample.
        n_dimensions (int): The number of hidden dimensions to use.
        n_classes (int): The number of classes.
        epochs (int, optional): The number of iteration over the training data.
        lr (float, optional): The learning rate.
        device (``torch.device``, optional):  the desired device of the weights. Default: if ``None``, uses the current device for the default tensor type (see ``torch.set_default_tensor_type()``). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        dtype (``torch.dtype``, optional): the desired data type of the weights. Default: if ``None``, uses ``torch.get_default_dtype()``.

    """

    encoder: Sinusoid
    model: Centroid

    def __init__(
        self,
        n_features: int,
        n_dimensions: int,
        n_classes: int,
        cfg,
        feat_model,
        *,
        epochs: int = 120,
        lr: float = 0.050,
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> None:
        super().__init__(
            n_features, n_dimensions, n_classes, device=device, dtype=dtype
        )

        self.epochs = epochs
        self.lr = lr
        self.cfg = cfg
        self.feat_model = feat_model

        self.encoder = Sinusoid(n_features, n_dimensions, device=device, dtype=dtype)
        self.model = Centroid(n_dimensions, n_classes, device=device, dtype=dtype)

    def fit(self, samples, labels):
                  
        samples = samples.to(self.device)
        labels = labels.to(self.device)
        enter = labels != -1

        count = torch.bincount(labels[enter])
        samples_per_label = torch.ones((self.n_classes)).to(self.device)
        samples_per_label[:len(count)] += count

        
        encoded = self.encoder(samples[enter])
        self.model.add_online(encoded, labels[enter], lr=self.lr)
        #adjusted_weight = self.model.weight * (1 / samples_per_label).view(-1, 1)
        #self.model.weight = nn.Parameter(adjusted_weight)

        del samples
        del samples_per_label
        del labels
        del encoded
        #del adjusted_weight
        del count
        torch.cuda.empty_cache()

        return self
    
    def feature_extractor(self, r_clouds):
        r_clouds.to(self.device)
        x = r_clouds.features.clone().detach()
        # Loop over consecutive blocks
        skip_x = []
        for block_i, block_op in enumerate(self.feat_model.encoder_blocks):
            if block_i == self.cfg.hd_block_stop:
                break
            if block_i in self.feat_model.encoder_skips:
                skip_x.append(x)
            x = block_op(x, r_clouds)

        continue_dec = (((-2)*(self.cfg.hd_block_stop - 2))/3) + 8

        for block_i, block_op in enumerate(self.feat_model.decoder_blocks):
            if block_i >= continue_dec and block_i % 2 == 0:
            #if block_i in self.decoder_concats and block_i % 2 == 0:
            #    x = torch.cat([x, skip_x.pop()], dim=1)
                x = block_op(x, r_clouds)
            else:
                continue
        return x
    
    def forward(self, r_clouds):
        x = self.feature_extractor(r_clouds)
        encoded = self.encoder(x)
        y = torch.argmax(torchhd.functional.cosine_similarity(encoded, self.model.weight), dim=1)

        del x
        del encoded
        torch.cuda.empty_cache()

        return y