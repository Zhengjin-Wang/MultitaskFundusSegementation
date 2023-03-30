import os
from PIL import Image
import numpy as np
import torch
from utils.utils import get_output, mkdir_if_missing

def get_mIOU(loader, model, task_list, n_classes=2):

    # Iterate
    all_tp = {task: [0] * n_classes for task in task_list}
    all_fp = {task: [0] * n_classes for task in task_list}
    all_fn = {task: [0] * n_classes for task in task_list}
    rsl = 0

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):

            images = batch['image'].cuda(non_blocking=True)
            targets = {task: batch[task].cuda(non_blocking=True) for task in task_list}
            output = model(images)

            for task in task_list:

                if task == 'biff' or task == 'cross' or task == 'endpoint':
                    continue

                label = targets[task]
                pred = output[task]

                batch_size = pred.shape[0]
                valid_idx = []
                for i in range(batch_size):
                    if label[i, 0, 0] != -1:
                        valid_idx.append(i)
                if len(valid_idx) == 0:
                    continue
                pred = pred[valid_idx, :, :, :]
                label = label[valid_idx, :, :]

                # argmax处理
                gt = np.array(label.squeeze().cpu().long())
                mask = np.argmax(np.array(pred.cpu()), axis=1)

                tp = all_tp[task]
                fp = all_fp[task]
                fn = all_fn[task]

                # TP, FP, and FN evaluation
                for i_part in range(0, n_classes):
                    tmp_gt = (gt == i_part) # 当前类别ground truth对的
                    tmp_pred = (mask == i_part) # 当前类别预测对的
                    tp[i_part] += np.sum(tmp_gt & tmp_pred)
                    fp[i_part] += np.sum(~tmp_gt & tmp_pred)
                    fn[i_part] += np.sum(tmp_gt & ~tmp_pred)
    print()
    for task in task_list:
        tp = all_tp[task]
        fp = all_fp[task]
        fn = all_fn[task]
        jac = [0] * n_classes
        for i_part in range(0, n_classes):
            jac[i_part] = float(tp[i_part]) / max(float(tp[i_part] + fp[i_part] + fn[i_part]), 1e-8)

        # print("{} mIoU: ".format(task) + str(np.mean(jac)))
        # print("IoU of {}: ".format('background') + str(jac[0]))
        print("IoU of {}: ".format(task) + str(jac[1]))
        print("Dice of {}: ".format(task) + str(2 * jac[1] / (1 + jac[1])))
        print('----------------------------------------\n')

        rsl += jac[1]

    return rsl / len(task_list)

@torch.no_grad()
def save_model_predictions(save_dir, tasks, val_loader, model):
    """ Save model predictions for all tasks """

    print('Save model predictions to {}'.format(save_dir))
    model.eval()
    save_dirs = {task: os.path.join(save_dir, task) for task in tasks} # 各任务的保存路径
    for save_dir in save_dirs.values():
        mkdir_if_missing(save_dir)

    for ii, sample in enumerate(val_loader):
        inputs = sample['image'].cuda(non_blocking=True)
        output = model(inputs)

        print("saving batch {}".format(str(ii)))

        for task in tasks:
            output_batch = get_output(output[task]).cpu().data.numpy()
            for jj in range(int(inputs.size()[0])):
                result = output_batch[jj].astype(np.uint8)
                result[result == 1] = 255
                Image.fromarray(result).save(os.path.join(save_dirs[task], sample['image_name'][jj]))
