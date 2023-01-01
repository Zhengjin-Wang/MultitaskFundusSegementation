import torch.nn as nn
import torch.nn.functional as F

class MultiTaskModel(nn.Module):
    """ Multi-task baseline model with shared encoder + task-specific decoders """
    def __init__(self, backbone: nn.Module, decoders: nn.ModuleDict, tasks: list):
        super(MultiTaskModel, self).__init__()
        assert(set(decoders.keys()) == set(tasks))
        self.backbone = backbone
        self.decoders = decoders
        self.tasks = tasks

    def forward(self, x):
        out_size = x.size()[2:]
        shared_representation = self.backbone(x)
        return {task: F.interpolate(self.decoders[task](shared_representation), out_size, mode='bilinear') for task in self.tasks}