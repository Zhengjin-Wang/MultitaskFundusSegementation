import errno
import os
import torch

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

# param is (batch_size, channel, height, width)
def get_output(output):
    _, output = torch.max(output, dim=1)
    return output

def get_pixel_portion(loader, n_classes, task):
    portion = [0] * n_classes
    with torch.no_grad():
        for i, batch in enumerate(loader):

            y = batch[task].cuda(non_blocking=True)

            batch_size = y.shape[0]
            valid_idx = []
            for i in range(batch_size):
                if y[i, 0, 0] != -1:
                    valid_idx.append(i)
            if len(valid_idx) == 0:
                continue

            y = y[valid_idx, :, :]
            gt = np.array(y.squeeze().cpu().long())

            # TP, FP, and FN evaluation
            for i_part in range(0, n_classes):
                portion[i_part] += np.sum(gt == i_part)
    print("class pixel num: ", portion) # 0是背景 1是前景 让bg_weight=portion[1]
    total = sum(portion)
    return [i / total for i in portion]

# from torch.utils.data import DataLoader
# from dataset.dataset import MULTIDATASET
# task_list = ['ma', 'he', 'ex', 'se', 'od']
# test_folder = 'E:\医学图像\dataset（gt和label文件名一致）\IDRID_768x768\\test'
# test_dataset = MULTIDATASET(test_folder, task_list)
# test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

# for task in task_list:
#     portion = get_pixel_portion(test_loader, 2, task)
#     print(task + " portion: " + str(portion))