import os.path
import sys
import torch

from termcolor import colored
from torch.utils.data import DataLoader
from utils.logger import Logger
from utils.utils import mkdir_if_missing
from utils.common_config import get_model, get_criterion, get_optimizer
from dataset.dataset import MULTIDATASET
from evaluation.evaluate_utils import get_mIOU

# task_list = ['vessel', 'ma', 'he', 'ex', 'se', 'od']
# task_list = ['ma', 'he', 'ex', 'od']
task_list = ['vessel', 'se', 'od']
cfg = {'net': 'hrnet', 'head': 'hrnet'}

loss_weights = {'vessel': 1.0, 'ma': 1.0, 'he': 1.0, 'ex': 1.0, 'se': 3.0, 'od': 1.0}
init_bg_weights = {'vessel': 0.12, 'ma': 0.01, 'he': 0.01, 'ex': 0.015, 'se': 0.01, 'od': 0.1}
optimizer_kwargs = {'lr': 0.001, 'weight_decay': 0.0002}
EPOCH = 100
BATCH_SIZE = 2
best_mIOU = 0
loss_history = []
iou_history = []

RESUME = False

train_folder = 'E:\医学图像\dataset（gt和label文件名一致）\MULTITASK_768x768\\train'
test_folder = 'E:\医学图像\dataset（gt和label文件名一致）\MULTITASK_768x768\\test'
save_folder = 'save'
mkdir_if_missing(save_folder)

sys.stdout = Logger(os.path.join(save_folder, 'log.txt'))

# Get model
print(colored('Retrieve model', 'blue'))
model = get_model(task_list, cfg)
model = torch.nn.DataParallel(model)
model = model.cuda()

# Get criterion
print(colored('Get loss', 'blue'))
criterion = get_criterion(task_list, loss_weights, init_bg_weights, RESUME)
criterion.cuda()
print(criterion)

# CUDNN
print(colored('Set CuDNN benchmark', 'blue'))
torch.backends.cudnn.benchmark = True

# Optimizer
print(colored('Retrieve optimizer', 'blue'))
optimizer = get_optimizer(model, optimizer_kwargs)
print(optimizer)

# transformation 直接在数据集中做

# Dataset
print(colored('Retrieve dataset', 'blue'))
train_dataset = MULTIDATASET(train_folder, task_list)
test_dataset = MULTIDATASET(test_folder, task_list)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(1, EPOCH + 1):
    print(colored('Epoch %d/%d' % (epoch, EPOCH), 'yellow'))
    print(colored('-' * 10, 'yellow'))
    epoch_loss = 0
    model.train()
    for i, batch in enumerate(train_loader):
        # Forward pass
        images = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(non_blocking=True) for task in task_list}
        output = model(images)

        # Measure loss and performance
        loss_dict = criterion(output, targets)

        # Backward
        optimizer.zero_grad()
        loss_dict['total'].backward()
        optimizer.step()

        loss_val = loss_dict['total'].item()
        epoch_loss += loss_val
        if (i + 1) % 10 == 0:
            print('Epoch {}, iter {}/{}, loss: {}'.format(epoch, i+1, len(train_loader), epoch_loss))

    print('Epoch {}, total loss: {}'.format(epoch, epoch_loss))

    mIOU = get_mIOU(test_loader, model, task_list, 2)
    print("mIOU is " + str(mIOU))
    if(mIOU > best_mIOU):
        print("best mIOU changed: {}->{}".format(str(best_mIOU), str(mIOU)))
        print("save model.\n")
        best_mIOU = mIOU
        torch.save(model.state_dict(), './save/best_model_epoch_{}.pth'.format(str(epoch)))

    loss_history.append(epoch_loss)
    iou_history.append(mIOU)

import matplotlib.pyplot as plt
plt.figure(dpi=80,figsize=(12,12))
plt.subplot(2,1,1)
plt.title("Loss")
plt.plot(loss_history)
plt.subplot(2,1,2)
plt.title("IoU")
plt.plot(iou_history)
plt.savefig("./save/plot.png")
plt.show()





