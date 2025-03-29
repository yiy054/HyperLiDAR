# Copyright 2022 - Valeo Comfort and Driving Assistance - Gilles Puy @ valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch.nn as nn
import torch
from .backbone import WaffleIron
from .embedding import Embedding


class Segmenter(nn.Module):
    def __init__(
        self,
        input_channels,
        feat_channels,
        nb_class,
        depth,
        grid_shape,
        drop_path_prob=0,
        layer_norm=False,
        early_exit=None,
    ):
        super().__init__()
        # Embedding layer
        self.embed = Embedding(input_channels, feat_channels)
        # WaffleIron backbone
        self.waffleiron = WaffleIron(feat_channels, depth, grid_shape, drop_path_prob, layer_norm, early_exit)
        # Classification layer
        self.classif = nn.Conv1d(feat_channels, nb_class, 1)

        self.early_exit = early_exit

    def compress(self):
        self.embed.compress()
        self.waffleiron.compress()

    def set_compensation(self, inter_weights_path, device):

        """Load all the paths for every exit"""

        self.linear_weights = {}

        for layer, path in inter_weights_path.items():
            self.linear_weights[layer] = nn.Linear(768, 768)
            state_dict = torch.load(path)
            self.linear_weights[layer].load_state_dict(state_dict)
            self.linear_weights[layer] = self.linear_weights[layer].to(device)
        self.compensation = True
        print("Compensation values updated")

    def forward(self, feats, cell_ind, occupied_cell, neighbors, step_type=None):

        ## If there is only one iter this should run only ###

        tokens_1 = self.embed(feats, neighbors) # radius can change based on the local density 
        # Node classification -> self and its neighbors
        tokens, exit_layer = self.waffleiron(tokens_1, cell_ind, occupied_cell, step_type)

        #if all_features:
        #    return tokens_1, tokens, self.classif(tokens[-1])
        #else:

        return self.exit(tokens_1, tokens, exit_layer, step_type)

    def continue_forward(self, tokens_init, iteration, step_type):

        ### If there is more than one stop ###

        tokens, exit_layer = self.waffleiron.tokenize(iteration, tokens_init, step_type)

        #if all_features:
        #    return tokens_1, tokens, self.classif(tokens[-1])
        #else:
        return self.exit(tokens_init, tokens, exit_layer, step_type)

    def exit(self, tokens_1, tokens, exit_layer, step_type):

        ### Final normalization ###

        if exit_layer != 47 and step_type != "distill":
            tokens = self.linear_weights[exit_layer+1](torch.transpose(tokens[0], 0, 1).to(torch.float32))
            tokens = torch.reshape(tokens, (1, tokens.shape[1], tokens.shape[0]))

        norm_feat = self.classif[0](tokens)
        soa_pred = self.classif[1](norm_feat)

        if step_type == "distill":
            if self.early_exit != [48]:
                return tokens_1, tokens, soa_pred, exit_layer
            else:
                return tokens_1, norm_feat, soa_pred, exit_layer

        else:

            return tokens_1, tokens, norm_feat, soa_pred, exit_layer