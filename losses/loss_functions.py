import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
import numpy as np

class MyBinaryCrossEntropyLoss(Module):
    """
    Binary Cross Entropy with ignore regions, not balanced.
    """

    def __init__(self, init_bg_weight, RESUME):
        super(MyBinaryCrossEntropyLoss, self).__init__()
        if RESUME:
            init_bg_weight *= 5
        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.crossEntropyLoss = nn.CrossEntropyLoss(torch.tensor([init_bg_weight, 0.99]).cuda())
        self.bg_weight = init_bg_weight
        # self.alpha = init_bg_weight * 0.03
        self.alpha = init_bg_weight * 0.01
        if RESUME:
            self.alpha *= 0.1
        self.iter = 0
        self.threshold = 0.7
        if init_bg_weight < 0.05:
            self.threshold = 0.35

    # 考虑如何动态更新权重
    def forward(self, output, label, void_pixels=None):
        assert not label.requires_grad

        # if label[0, 0] == -1:
        #     print("空标签")
        #     return 0
        batch_size = output.shape[0]
        valid_idx = []
        for i in range(batch_size):
            if label[i, 0, 0] != -1:
                valid_idx.append(i)
        if len(valid_idx) == 0:
            return torch.tensor(0).float().cuda()
        output = output[valid_idx, :, :, :]
        label = label[valid_idx, :, :]

        self.iter += 1
        if self.iter % 10 == 0 and self.bg_weight < self.threshold:
            self.bg_weight += self.alpha
            self.crossEntropyLoss.weight = torch.tensor([self.bg_weight, 0.99]).cuda()
        # if self.iter % 27 == 0:
        #     print("Now the bg_weight is{}".format(str(self.bg_weight)))
        if self.iter == 1000:
            self.alpha *= 0.15
        if self.iter == 4000:
            self.alpha = 0

        label = label.squeeze().ge(0.5).long()
        if len(label.shape) == 2:
            label = label.unsqueeze(dim=0)

        final_loss = self.crossEntropyLoss(output, label)

        return final_loss
