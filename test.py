
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808

import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
from config import Config
import torchvision
opt = Config('training.yml')
from data_RGB import get_training_data, get_validation_data
from Old_Photo_Net import Old_Photo_Net
#from Old_Photo_Net_scratch import Old_Photo_Net
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import torch
import time

parser = argparse.ArgumentParser(description='Image Deraining using MPRNet')
parser.add_argument('--input_dir', default='', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--checkpoints_dir', default='', type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

opt = Config('training.yml')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
  #  print(net)
    print('Total number of parameters: %d' % num_params)

val_dataset = get_validation_data(args.input_dir, {'patch_size': opt.TRAINING.VAL_PS})
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False,pin_memory=True)
model = Old_Photo_Net().cuda()

print_network(model)
load_path = os.path.join(args.checkpoints_dir, '')
state_dict = torch.load(load_path.replace('\\', '/'), map_location=str(torch.device('cuda')))
model.load_state_dict(state_dict['net'])

start = time.time()
for ii, data_val in enumerate((val_loader), 0):
        target = data_val[0]
        input_ = data_val[1]
        mask = data_val[2]
        mask1 = data_val[3]
        mask2 = data_val[4]

        input = input_.to(torch.device('cuda'))
        target = target.to(torch.device('cuda'))
        mask = mask.to(torch.device('cuda'))
        mask1 = mask1.to(torch.device('cuda'))
        mask2 = mask2.to(torch.device('cuda'))
        fake_out_label = model(input, mask, mask1, mask2)

        def get_current_visuals_test():
            input_image = input.data.cpu()
            fake_image_1 = fake_out_label[0].data.cpu()
            return input_image, fake_image_1
        with torch.no_grad():
            input_image, fake_image_1= get_current_visuals_test()
            image_out = fake_image_1
            grid = torchvision.utils.make_grid(image_out)
            file_name = '' + format(str(ii + 1), '0>4s') + '.png'
            torchvision.utils.save_image(grid, file_name, nrow=1)

print('time is', time.time()-start)
