import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image
import os


class MULTIDATASET(Dataset):

    def __init__(self, root_folder, task_list):
        """
        Args:
            img_folder (string): Directory with all the training images.
            labels_folder (string): Directory with all the image labels.
        """
        self.root_folder = root_folder
        self.task_list = task_list
        self.img_folder = os.path.join(root_folder, 'images')
        self.images = os.listdir(self.img_folder)
        self.label_folders = {task: os.path.join(root_folder, task) for task in task_list}
        self.label_sets = {task: os.listdir(self.label_folders[task]) for task in task_list}

        self.tx = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.mx = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: torch.cat([x, 1 - x], dim=0))
        ])

    def __len__(self):
        return len(self.images)

    # 由img, labels组成，img是图像，labels是词典
    def __getitem__(self, i):
        img_name = self.images[i]
        i1 = Image.open(os.path.join(self.img_folder, self.images[i]))
        sample = {'image': self.tx(i1)}
        sample['image_name'] = img_name

        for task in self.task_list:
            if task == 'icmd_classify':
                if img_name[0] == 'N':
                    sample[task] = torch.tensor(0).long()
                elif img_name[0] == 'H':
                    sample[task] = torch.tensor(1).long()
                elif img_name[0] == 'P':
                    sample[task] = torch.tensor(2).long()
                continue
            m1t = torch.empty([768, 768])
            m1t[0, 0] = -1
            if img_name in self.label_sets[task]:
                m1 = Image.open(os.path.join(self.label_folders[task], self.images[i]))
                if not(task == 'biff' or task == 'cross' or task == 'endpoint'):
                    m1t = (self.mx(m1) > 0.2).long()
                else:
                    m1t = self.mx(m1)
                m1t = m1t[0, :, :]
            sample[task] = m1t

        return sample

# task_list = ['vessel', 'ma', 'he', 'ex', 'se', 'od']
# dataset = MULTIDATASET('E:\医学图像\dataset（gt和label文件名一致）\MULTITASK_768x768\\train', task_list)
# for i, sample in enumerate(dataset):
#     if sample['image_name'] == 'idrid_43.png':
#         print(sum(sample['he']))
