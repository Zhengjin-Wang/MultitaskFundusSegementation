import torch

"""模型相关"""

def get_backbone(cfg):
    """ Return the backbone """
    if cfg['net'] == 'hrnet':
        from models.seg_hrnet import hrnet_w18
        backbone = hrnet_w18(pretrained=True)
        backbone_channels = [18, 36, 72, 144]

    elif cfg['net'] == 'resnet':
        from models.resnet import resnet18
        backbone = resnet18(pretrained=True)
        from models.resnet_dilated import ResnetDilated
        backbone = ResnetDilated(backbone)
        backbone_channels = 512

    return backbone, backbone_channels


def get_head(backbone_channels, task, cfg):
    """ Return the decoder head """
    if cfg['head'] == 'hrnet':
        if task == 'biff' or task == 'cross' or task == 'endpoint':
            from models.seg_hrnet import SoftmaxHighResolutionHead
            return SoftmaxHighResolutionHead(backbone_channels, 2)
        else:
            from models.seg_hrnet import HighResolutionHead
            return HighResolutionHead(backbone_channels, 2)
    elif cfg['head'] == 'deeplab':
        from models.aspp import DeepLabHead
        return DeepLabHead(backbone_channels, 2)
    else:
        raise NotImplementedError


# task_list 任务名列表
def get_model(task_list, cfg, GRADNORM):
    """ Return the model """

    backbone, backbone_channels = get_backbone(cfg)

    from models.MultiTaskModel import MultiTaskModel
    heads = torch.nn.ModuleDict({task: get_head(backbone_channels, task, cfg) for task in task_list})
    model = MultiTaskModel(backbone, heads, task_list, GRADNORM)

    return model

"""损失函数相关"""

def get_loss(init_bg_weight, task, RESUME):
    if task == 'biff' or task == 'cross' or task == 'endpoint':
        from losses.loss_functions import MyMSELoss
        criterion = MyMSELoss(init_bg_weight)
    else:
        from losses.loss_functions import MyBinaryCrossEntropyLoss
        criterion = MyBinaryCrossEntropyLoss(init_bg_weight, RESUME)

    return criterion

def get_criterion(task_list, loss_weights, init_bg_weights, RESUME):

    from losses.loss_schemes import MultiTaskLoss
    loss_ft = torch.nn.ModuleDict({task: get_loss(init_bg_weights[task], task, RESUME) for task in task_list})
    return MultiTaskLoss(task_list, loss_ft, loss_weights)

"""优化器相关"""
def get_optimizer(model, optimizer_kwargs):
    params = model.parameters()
    optimizer = torch.optim.Adam(params, **optimizer_kwargs)
    return optimizer
