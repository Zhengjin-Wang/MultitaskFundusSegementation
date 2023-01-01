import torch

from utils.common_config import get_model
from torch.utils.data import DataLoader
from dataset.dataset import MULTIDATASET
from evaluation.evaluate_utils import save_model_predictions

test_folder = '/root/data1/wzj/MULTITASK_768x768/test'
model_path = '/root/data1/wzj/MultitaskFundusSegementation/save/best_model_epoch_40.pth'
task_list = ['vessel', 'ma', 'he', 'ex', 'se', 'od']

model = get_model(task_list)
model = torch.nn.DataParallel(model)
model = model.cuda()
model.load_state_dict(torch.load(model_path))
test_dataset = MULTIDATASET(test_folder, task_list)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
save_model_predictions("/root/data1/wzj/MultitaskFundusSegementation/save/test_rsl", task_list, test_loader, model)