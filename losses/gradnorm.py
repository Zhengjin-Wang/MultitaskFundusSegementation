import numpy as np
import torch

def gradnorm(model, criterion, task_list, optimizer, init_loss, loss_dict, gradnorm_alpha):
    model = model.module

    task_loss = torch.tensor([]).float().cuda()
    for task in task_list:
        task_loss = torch.cat((task_loss, loss_dict[task].unsqueeze(dim=0)))
    weighted_task_loss = torch.mul(model.loss_weights, task_loss)

    loss = torch.sum(weighted_task_loss)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    model.loss_weights.grad.data = model.loss_weights.grad.data * 0.0

    W = model.backbone.stage4

    norms = []
    for i in range(len(task_list)):
        gygw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)
        norms.append(torch.norm(torch.mul(model.loss_weights[i], gygw[0])))
    norms = torch.stack(norms)

    loss_ratio = task_loss.data.cpu().numpy() / init_loss

    inverse_train_rate = loss_ratio / np.mean(loss_ratio)
    mean_norm = np.mean(norms.data.cpu().numpy())

    constant_term = torch.tensor(mean_norm * (inverse_train_rate ** gradnorm_alpha), requires_grad=False)
    constant_term = constant_term.cuda()
    grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
    model.loss_weights.grad = torch.autograd.grad(grad_norm_loss, model.loss_weights)[0]

    optimizer.step()

    normalize_coeff = len(task_list) / torch.sum(model.loss_weights.data, dim=0)
    model.loss_weights.data = model.loss_weights.data * normalize_coeff

    # update loss weights of criterion
    for i in range(len(task_list)):
        criterion.loss_weights[task_list[i]] = model.loss_weights[i].item()

