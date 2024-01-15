import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from pdb import set_trace as stx
import random
from torchvision import transforms
import numpy as np


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

class DataLoaderTrain(Dataset):
    def __init__(self, label_dir, img_options=None):
        super(DataLoaderTrain, self).__init__()

        inp_files_label = sorted(os.listdir(os.path.join(label_dir, 'input')))
        tar_files_label = sorted(os.listdir(os.path.join(label_dir, 'target')))
        mask1_files_label = sorted(os.listdir(os.path.join(label_dir,'mask1')))
        mask2_files_label = sorted(os.listdir(os.path.join(label_dir,'mask2')))
        mask_files_label = sorted(os.listdir(os.path.join(label_dir,'mask')))

        self.inp_filenames_label = [os.path.join(label_dir, 'input', x)  for x in inp_files_label if is_image_file(x)]
        self.tar_filenames_label = [os.path.join(label_dir, 'target', x) for x in tar_files_label if is_image_file(x)]
        self.mask1_filenames_label = [os.path.join(label_dir, 'mask1', x) for x in mask1_files_label if is_image_file(x)]
        self.mask2_filenames_label = [os.path.join(label_dir, 'mask2', x) for x in mask2_files_label if is_image_file(x)]
        self.mask_filenames_label = [os.path.join(label_dir, 'mask', x) for x in mask_files_label if is_image_file(x)]

        self.img_options = img_options
        self.sizex_label  = len(self.tar_filenames_label)
        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex_label

    def __getitem__(self, index):
        index_label = index % self.sizex_label

        inp_path_label = self.inp_filenames_label[index_label]
        tar_path_label = self.tar_filenames_label[index_label]
        mask1_path_label = self.mask1_filenames_label[index_label]
        mask2_path_label = self.mask2_filenames_label[index_label]
        mask_path_label = self.mask_filenames_label[index_label]

        inp_img_label = Image.open(inp_path_label)
        tar_img_label = Image.open(tar_path_label)
        mask1_img_label = Image.open(mask1_path_label)
        mask2_img_label = Image.open(mask2_path_label)
        mask_img_label = Image.open(mask_path_label)

        inp_img_label = inp_img_label.resize((int(256), int(256)))
        tar_img_label = tar_img_label.resize((int(256), int(256)))
        mask1_img_label = mask1_img_label.resize((int(256), int(256)))
        mask2_img_label = mask2_img_label.resize((int(256), int(256)))
        mask_img_label = mask_img_label.resize((int(256),int(256)))


        inp_img_label = TF.to_tensor(inp_img_label)
        tar_img_label = TF.to_tensor(tar_img_label)
        mask1_img_label = TF.to_tensor(mask1_img_label)
        mask2_img_label = TF.to_tensor(mask2_img_label)
        mask_img_label = TF.to_tensor(mask_img_label)

        aug = random.randint(0, 8)

        if aug==1:
            inp_img_label = inp_img_label.flip(1)
            tar_img_label = tar_img_label.flip(1)
            mask1_img_label = mask1_img_label.flip(1)
            mask2_img_label = mask2_img_label.flip(1)
            mask_img_label = mask_img_label.flip(1)
        elif aug==2:
            inp_img_label = inp_img_label.flip(2)
            tar_img_label = tar_img_label.flip(2)
            mask1_img_label= mask1_img_label.flip(2)
            mask2_img_label = mask2_img_label.flip(2)
            mask_img_label = mask_img_label.flip(2)
        elif aug==3:
            inp_img_label = torch.rot90(inp_img_label,dims=(1,2))
            tar_img_label = torch.rot90(tar_img_label,dims=(1,2))
            mask1_img_label = torch.rot90(mask1_img_label,dims=(1,2))
            mask2_img_label = torch.rot90(mask2_img_label, dims=(1, 2))
            mask_img_label = torch.rot90(mask_img_label, dims=(1, 2))
        elif aug==4:
            inp_img_label = torch.rot90(inp_img_label,dims=(1,2), k=2)
            tar_img_label = torch.rot90(tar_img_label,dims=(1,2), k=2)
            mask1_img_label = torch.rot90(mask1_img_label,dims=(1,2), k=2)
            mask2_img_label = torch.rot90(mask2_img_label, dims=(1, 2), k=2)
            mask_img_label = torch.rot90(mask_img_label, dims=(1, 2), k=2)
        elif aug==5:
            inp_img_label = torch.rot90(inp_img_label,dims=(1,2), k=3)
            tar_img_label = torch.rot90(tar_img_label,dims=(1,2), k=3)
            mask1_img_label = torch.rot90(mask1_img_label, dims=(1, 2), k=3)
            mask2_img_label = torch.rot90(mask2_img_label, dims=(1, 2), k=3)
            mask_img_label = torch.rot90(mask_img_label, dims=(1, 2), k=3)
        elif aug==6:
            inp_img_label = torch.rot90(inp_img_label.flip(1),dims=(1,2))
            tar_img_label = torch.rot90(tar_img_label.flip(1),dims=(1,2))
            mask1_img_label = torch.rot90(mask1_img_label.flip(1),dims=(1,2))
            mask2_img_label = torch.rot90(mask2_img_label.flip(1), dims=(1, 2))
            mask_img_label = torch.rot90(mask_img_label.flip(1), dims=(1, 2))
        elif aug==7:
            inp_img_label = torch.rot90(inp_img_label.flip(2),dims=(1,2))
            tar_img_label = torch.rot90(tar_img_label.flip(2),dims=(1,2))
            mask1_img_label = torch.rot90(mask1_img_label.flip(2), dims=(1, 2))
            mask2_img_label = torch.rot90(mask2_img_label.flip(2), dims=(1, 2))
            mask_img_label = torch.rot90(mask_img_label.flip(2), dims=(1, 2))

        return tar_img_label, inp_img_label, mask1_img_label, mask2_img_label, mask_img_label


class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        super(DataLoaderVal, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))
        mask_files = sorted(os.listdir(os.path.join(rgb_dir, 'mask')))
        mask1_files = sorted(os.listdir(os.path.join(rgb_dir, 'mask1')))
        mask2_files = sorted(os.listdir(os.path.join(rgb_dir, 'mask2')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x)  for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x)]
        self.mask_filenames = [os.path.join(rgb_dir, 'mask', x) for x in mask_files if is_image_file(x)]
        self.mask1_filenames = [os.path.join(rgb_dir, 'mask1', x) for x in mask1_files if is_image_file(x)]
        self.mask2_filenames = [os.path.join(rgb_dir, 'mask2', x) for x in mask2_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex       = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]
        mask_path = self.mask_filenames[index_]
        mask1_path = self.mask1_filenames[index_]
        mask2_path = self.mask2_filenames[index_]

        inp_img = Image.open(inp_path)
        tar_img = Image.open(tar_path)
        mask_img = Image.open(mask_path)
        mask1_img = Image.open(mask1_path)
        mask2_img = Image.open(mask2_path)

        image_transform_2 = transforms.Compose([
            transforms.Resize(128),
        ])
        image_transform_3 = transforms.Compose([
            transforms.Resize(64),
        ])

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)
        mask_img = TF.to_tensor(mask_img)
        mask1_img = TF.to_tensor(mask1_img)
        mask2_img = TF.to_tensor(mask2_img)

        return tar_img, inp_img, mask_img, mask1_img, mask2_img

class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, mask_dir,img_options):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        mask_files = sorted(os.listdir(mask_dir))

        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]
        self.mask_filenames = [os.path.join(mask_dir, x) for x in mask_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):

        path_inp = self.inp_filenames[index]
        path_mask = self.mask_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]

        inp = Image.open(path_inp)
        mask = Image.open(path_mask)
        inp = TF.to_tensor(inp)
        mask = TF.to_tensor(mask)
        return inp, mask, filename

