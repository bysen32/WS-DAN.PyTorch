"""
WS-DAN models

Hu et al.,
"See Better Before Looking Closer: Weakly Supervised Data Augmentation Network for Fine-Grained Visual Classification",
arXiv:1901.09891

Created: May 04,2019 - Yuchong Gu
Revised: Dec 03,2019 - Yuchong Gu
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.vgg as vgg
import models.resnet as resnet
from models.inception import inception_v3, BasicConv2d

__all__ = ['WSDAN']
EPSILON = 1e-12


# Bilinear Attention Pooling
class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix, dim=-1)
        return feature_matrix

class tri_att(nn.Module):
    def __init__(self):
        super(tri_att, self).__init__()
        self.feature_norm = nn.Softmax(dim=2)
        self.bilinear_norm = nn.Softmax(dim=2)
    
    def forward(self, x):
        n = x.size(0)
        c = x.size(1)
        h = x.size(2)
        w = x.size(3)
        f = x.reshape(n, c, -1)

        # *7 to obtain an appropriate scale for the input of softmax function.
        f_norm = self.feature_norm(f * 2)

        bilinear = f_norm.bmm(f.transpose(1, 2))
        bilinear = self.bilinear_norm(bilinear)
        trilinear_atts = bilinear.bmm(f).view(n, c, h, w).detach()
        # structure_att = torch.sum(trilinear_atts, dim=1, keepdim=True)

        # index = torch.randint(c, (n,))
        # detail_att = trilinear_atts[torch.arange(n), index] + 0.01
        
        # return structure_att, detail_att.unsqueeze(1)
        return trilinear_atts

# WS-DAN: Weakly Supervised Data Augmentation Network for FGVC
class WSDAN(nn.Module):
    def __init__(self, num_classes, M=32, net='inception_mixed_6e', pretrained=False):
        super(WSDAN, self).__init__()
        self.num_classes = num_classes
        self.M = M
        self.net = net

        # Network Initialization
        if 'inception' in net:
            if net == 'inception_mixed_6e':
                self.features = inception_v3(pretrained=pretrained).get_features_mixed_6e()
                self.num_features = 768
            elif net == 'inception_mixed_7c':
                self.features = inception_v3(pretrained=pretrained).get_features_mixed_7c()
                self.num_features = 2048
            else:
                raise ValueError('Unsupported net: %s' % net)
        elif 'vgg' in net:
            self.features = getattr(vgg, net)(pretrained=pretrained).get_features()
            self.num_features = 512
        elif 'resnet' in net:
            self.features = getattr(resnet, net)(pretrained=pretrained).get_features()
            self.num_features = 512 * self.features[-1][-1].expansion
        else:
            raise ValueError('Unsupported net: %s' % net)

        # Attention Maps
        self.attentions = BasicConv2d(self.num_features, self.M, kernel_size=1)

        # Bilinear Attention Pooling
        self.bap = BAP(pool='GAP')
        # self.tri_att = tri_att()

        # Classification Layer
        self.fc = nn.Linear(self.M * self.num_features, self.num_classes, bias=False)

        logging.info('WSDAN: using {} as feature extractor, num_classes: {}, num_attentions: {}'.format(net, self.num_classes, self.M))

    def forward(self, x):
        batch_size = x.size(0)

        # Feature Maps, Attention Maps and Feature Matrix
        feature_maps = self.features(x)
        # feature_maps = self.tri_att(feature_maps)
        if self.net != 'inception_mixed_7c':
            attention_maps = self.attentions(feature_maps)
        else:
            attention_maps = feature_maps[:, :self.M, ...]
        feature_matrix = self.bap(feature_maps, attention_maps)

        # Classification
        p = self.fc(feature_matrix * 100.)

        # p: (B, self.num_classes)
        # feature_matrix: (B, M * C)
        # attention_map: (B, 2, H, W) in training, (B, 1, H, W) in val/testing
        return p, feature_matrix, attention_maps

    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and model_dict[k].size() == v.size()}

        if len(pretrained_dict) == len(state_dict):
            logging.info('%s: All params loaded' % type(self).__name__)
        else:
            logging.info('%s: Some params were not loaded:' % type(self).__name__)
            not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
            logging.info(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))

        model_dict.update(pretrained_dict)
        super(WSDAN, self).load_state_dict(model_dict)
