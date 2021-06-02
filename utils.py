"""Utils
Created: Nov 11,2019 - Yuchong Gu
Revised: Dec 03,2019 - Yuchong Gu
"""
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import att_grid_generator_cuda


##############################################
# Center Loss for Attention Regularization
##############################################
class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, outputs, targets):
        return self.l2_loss(outputs, targets) / outputs.size(0)

###################################
# Contrastive Loss for Attention
###################################
class AttenContraLoss(nn.Module):
    def __init__(self):
        super(AttenContraLoss, self).__init__()
    
    def forward(self, attention_maps):
        B, M, W, H = attention_maps.size()
        # flatten (B, M, W*H)
        attention_maps = attention_maps.reshape(B, M, -1)
        # normalize (B, M, W*H)
        norm_atten_maps = F.normalize(attention_maps, dim=2)
        # mm (B, M, M)
        cos_matrix = torch.einsum('bik,bjk->bij', [norm_atten_maps, norm_atten_maps])
        indexs = torch.arange(M)
        mask = torch.stack([indexs != i for i in range(M)]).float().cuda()
        cos_matrix = cos_matrix * mask
        # sum / B * B
        loss = cos_matrix.sum()
        loss /= (B * M * (M - 1))
        return loss

####################################
# Contrastive Loss for Part
####################################
class PartContraLoss(nn.Module):
    def __init__(self):
        super(PartContraLoss, self).__init__()
    
    def forward(self, feature_matrix):
        B, M, C = feature_matrix.size()
        norm_feature_matrix = F.normalize(feature_matrix, dim=2)
        cos_matrix = torch.einsum('bik,bjk->bij', [norm_feature_matrix, norm_feature_matrix])
        indexs = torch.arange(M)
        mask = torch.stack([indexs != i for i in range(M)]).float().cuda()
        cos_matrix = cos_matrix * mask
        loss = cos_matrix.sum()
        loss /= (B * M * M)
        return loss


##################################
# Metric
##################################
class Metric(object):
    pass


class AverageMeter(Metric):
    def __init__(self, name='loss'):
        self.name = name
        self.reset()

    def reset(self):
        self.scores = 0.
        self.total_num = 0.

    def __call__(self, batch_score, sample_num=1):
        self.scores += batch_score
        self.total_num += sample_num
        return self.scores / self.total_num


class TopKAccuracyMetric(Metric):
    def __init__(self, topk=(1,)):
        self.name = 'topk_accuracy'
        self.topk = topk
        self.maxk = max(topk)
        self.reset()

    def reset(self):
        self.corrects = np.zeros(len(self.topk))
        self.num_samples = 0.

    def __call__(self, output, target):
        """Computes the precision@k for the specified values of k"""
        self.num_samples += target.size(0)
        _, pred = output.topk(self.maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for i, k in enumerate(self.topk):
            correct_k = correct[:k].reshape(-1).float().sum(0)
            self.corrects[i] += correct_k.item()

        return self.corrects * 100. / self.num_samples


##################################
# Callback
##################################
class Callback(object):
    def __init__(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, *args):
        pass


class ModelCheckpoint(Callback):
    def __init__(self, savepath, monitor='val_topk_accuracy', mode='max'):
        self.savepath = savepath
        self.monitor = monitor
        self.mode = mode
        self.reset()
        super(ModelCheckpoint, self).__init__()

    def reset(self):
        if self.mode == 'max':
            self.best_score = float('-inf')
        else:
            self.best_score = float('inf')

    def set_best_score(self, score):
        if isinstance(score, np.ndarray):
            self.best_score = score[0]
        else:
            self.best_score = score

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, logs, net, **kwargs):
        current_score = logs[self.monitor]
        if isinstance(current_score, np.ndarray):
            current_score = current_score[0]

        if (self.mode == 'max' and current_score > self.best_score) or \
                (self.mode == 'min' and current_score < self.best_score):
            self.best_score = current_score

            if isinstance(net, torch.nn.DataParallel):
                state_dict = net.module.state_dict()
            else:
                state_dict = net.state_dict()

            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()

            if 'feature_center' in kwargs:
                feature_center = kwargs['feature_center']
                feature_center = feature_center.cpu()

                torch.save({
                    'logs': logs,
                    'state_dict': state_dict,
                    'feature_center': feature_center}, self.savepath)
            else:
                torch.save({
                    'logs': logs,
                    'state_dict': state_dict}, self.savepath)

##################################
# att sampler 注意力采样
##################################
def attention_sample(data, att, out_size=448):
    n = data.size(0)
    h = data.size(2)
    att = F.interpolate(att, (h, h), mode='bilinear', align_corners=False).squeeze(1)
    map_sx, _ = torch.max(att, 2)
    map_sx = map_sx.unsqueeze(2)

    map_sy, _ = torch.max(att, 1)
    map_sy = map_sy.unsqueeze(2)

    sum_sx = torch.sum(map_sx, (1, 2), keepdim=True)
    sum_sy = torch.sum(map_sy, (1, 2), keepdim=True)

    map_sx = torch.div(map_sx, sum_sx)
    map_sy = torch.div(map_sy, sum_sy)
    map_xi = torch.zeros_like(map_sx)
    map_yi = torch.zeros_like(map_sy)

    index_x = torch.zeros((n, out_size, 1)).cuda()
    index_y = torch.zeros((n, out_size, 1)).cuda()

    att_grid_generator_cuda.forward(map_sx, map_sy, map_xi, map_yi, index_x, index_y, h, out_size, 4, 5, out_size/h)

    one_vector = torch.ones_like(index_x)
    grid_x = torch.matmul(one_vector, index_x.transpose(1, 2)).unsqueeze(-1)
    grid_y = torch.matmul(index_y, one_vector.transpose(1, 2)).unsqueeze(-1)
    grid = torch.cat((grid_x, grid_y), 3)
    data = F.grid_sample(data, grid)
    return data

##################################
# augment function
##################################
def batch_augment(images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1):
    batches, _, imgH, imgW = images.size()

    if mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            crop_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) >= theta_c
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

            crop_images.append(
                F.upsample_bilinear(images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                                    size=(imgH, imgW)))
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images

    elif mode == 'drop':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max()
            else:
                theta_d = theta * atten_map.max()

            drop_masks.append(F.upsample_bilinear(atten_map, size=(imgH, imgW)) < theta_d)
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()
        return drop_images

    else:
        raise ValueError('Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)


##################################
# transform in dataset
##################################
def get_transform(resize, phase='train'):
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
            transforms.RandomCrop(resize),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.126, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[ 0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
