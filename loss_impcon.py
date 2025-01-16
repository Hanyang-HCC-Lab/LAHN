import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class Similarity_diag(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, a, b):
        cos_sim_diag = torch.diag(self.cos(a, b) / self.temp)
        cos_sim_diag = cos_sim_diag.view(-1, 1)

        return cos_sim_diag


class Ours(nn.Module):
    def __init__(self, temperature=0.07):
        super(Ours, self).__init__()
        self.temperature = temperature
        self.sim = Similarity(temperature)
        self.sim_diag = Similarity_diag(temperature)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, features, labels=None, mask=None, momentum_features=None, momentum_features_pos=None, momentum_labels=None, anchor_labels=None, all_features_m=None, all_labels_m=None):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0] ## Original + Augmentation
        origin_labels, _ = labels.chunk(2, dim=0)


        z1, z2 = features.chunk(2, dim=0)
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        target = torch.zeros_like(cos_sim)
        for idx, i in enumerate(origin_labels):
            if i==1:
                target[idx] = origin_labels
            elif i==0:
                target[idx] = 1 - origin_labels

        if momentum_features is None:
            loss = self.bce(cos_sim, target)
        else:
            cos_sim_diag = self.sim_diag(z1.unsqueeze(1), z2.unsqueeze(0))
            cos_sim_moco = self.sim(z1.unsqueeze(1), momentum_features.unsqueeze(0).to(device))
            cos_sim_moco = cos_sim_moco.squeeze()
            concat_all_sim = torch.cat((cos_sim_diag, cos_sim_moco),1).to(device)
            target = torch.zeros_like(concat_all_sim, device='cuda:0')
            target[:, 0] = 1
            loss = self.bce(concat_all_sim, target)

        return loss
    


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07, ):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.sim = Similarity(temperature)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, features, labels=None, mask=None, momentum_features=None):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0] ## Original + Augmentation
        origin_labels, _ = labels.chunk(2, dim=0)

        z1, z2 = features.chunk(2, dim=0)
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        target = torch.zeros_like(cos_sim)

        for idx, i in enumerate(origin_labels):
            if i==1:
                target[idx] = origin_labels
            elif i==0:
                target[idx] = 1 - origin_labels
        loss = self.bce(cos_sim, target)
            
        return loss


class UnSupConLoss(nn.Module):

    def __init__(self, temperature=0.07):
        super(UnSupConLoss, self).__init__()
        self.temperature = temperature
        self.sim = Similarity(temperature)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, features, labels=None, mask=None):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0] ## Original + Augmentation
        z1, z2 = features.chunk(2, dim=0)

        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        # cos_labels = torch.arange(cos_sim.size(0)).long().to(device)

        target = torch.zeros_like(cos_sim)
        for idx, i in enumerate(range(cos_sim.size(0))): 
            target[idx,idx] = 1
        loss = self.bce(cos_sim, target)

        return loss
    
class ImpCon(nn.Module):

    def __init__(self, temperature=0.07):
        super(ImpCon, self).__init__()
        self.temperature = temperature
        self.sim = Similarity(temperature)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, features, labels=None, mask=None):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0] ## Original + Augmentation
        z1, z2 = features.chunk(2, dim=0)

        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        # cos_labels = torch.arange(cos_sim.size(0)).long().to(device)

        target = torch.zeros_like(cos_sim)
        for idx, i in enumerate(range(cos_sim.size(0))): # 
            target[idx,idx] = 1
        loss = self.bce(cos_sim, target)

        return loss


### Credits https://github.com/HobbitLong/SupContrast
class SupConLoss_Original(nn.Module):

    def __init__(self, temperature=0.07):
        super(SupConLoss_Original, self).__init__()
        self.temperature = temperature


    def forward(self, features, labels=None, mask=None):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0] ## 2*N
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            contrast_count = 2
            anchor_count = contrast_count
            assert batch_size % 2 == 0
            mask = torch.eye(batch_size//2, dtype=torch.float32).to(device)
            mask = mask.repeat(anchor_count, contrast_count)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            raise NotImplementedError


        contrast_feature = features
        anchor_feature = contrast_feature


        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        ## it produces 1 for the non-matching places and 0 for matching places i.e its opposite of mask
        mask = mask * logits_mask

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        
        logits = anchor_dot_contrast - logits_max.detach() 

        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) 
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) 

        # loss
        loss = -1 * mean_log_prob_pos
        loss = loss.mean()

        return loss


#######################################################################################
### Credits https://github.com/HobbitLong/SupContrast
class SupConLoss_for_double(nn.Module):

    def __init__(self, temperature=0.07):
        super(SupConLoss_for_double, self).__init__()
        self.temperature = temperature


    def forward(self, features, labels=None, mask=None):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0] ## 3*N

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            contrast_count = 3
            anchor_count = contrast_count
            assert batch_size % 3 == 0
            mask = torch.eye(batch_size//3, dtype=torch.float32).to(device)
            mask = mask.repeat(anchor_count, contrast_count)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device) 
        else:
            raise NotImplementedError

        contrast_feature = features
        anchor_feature = contrast_feature

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        ## it produces 1 for the non-matching places and 0 for matching places i.e its opposite of mask
        mask = mask * logits_mask 

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) 

        loss = -1 * mean_log_prob_pos
        loss = loss.mean()

        return loss

