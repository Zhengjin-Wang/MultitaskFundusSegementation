import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss(nn.Module):
    def __init__(self, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict):
        super(MultiTaskLoss, self).__init__()
        # assert (set(tasks) == set(loss_ft.keys()))
        # assert (set(tasks) == set(loss_weights.keys()))
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

    def forward(self, pred, gt):
        out = {task: self.loss_ft[task](pred[task], gt[task]) for task in self.tasks}
        out['total'] = torch.sum(torch.stack([self.loss_weights[t] * out[t] for t in self.tasks]))
        return out