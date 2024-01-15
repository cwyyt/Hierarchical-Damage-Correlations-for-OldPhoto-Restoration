
import os
from config import Config
opt = Config('training.yml')
gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
torch.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader
import torchvision
import time
import numpy as np
from data_RGB import get_training_data
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
torch.backends.cudnn.benchmark = True
from Old_Photo_Net import define_D
from Old_Photo_Net import Old_Photo_Net
import losses 
import torch
import random
from base_model import get_scheduler
import matplotlib
matplotlib.use('Agg')
from collections import OrderedDict

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
model_dir  = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models',  session)
train_dir_label = opt.TRAINING.TRAIN_DIR_LABEL
######### DataLoaders ################################################

train_dataset = get_training_data(train_dir_label, {'patch_size':opt.TRAINING.TRAIN_PS})
train_loader = DataLoader(dataset=train_dataset, batch_size = 1, shuffle=True, num_workers=16, drop_last=False, pin_memory=True)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.OPTIM.NUM_EPOCHS + 1))
print('===> Loading datasets')

best_psnr = 0
best_epoch = 0
total_steps = 0

# Create the logs
dir = ''
if not os.path.exists(dir):
    os.mkdir(dir)
writer = SummaryWriter(log_dir=dir, comment='')

def build_model_restoration(ema=False):
    if not ema:
        return Old_Photo_Net().cuda()
    else:
        return Old_Photo_Net().cuda()

def create_model_restoration(ema=False):
        # Network definition
        if ema:
            net = build_model_restoration(ema=True)
            net_cuda = net.cuda()
            for param in net_cuda.parameters():
                param.detach_()
        else:
            net = build_model_restoration()
            net_cuda = net.cuda()
        return net_cuda

def update_learning_rate():
    for scheduler in schedulers:
        scheduler.step()
    lr = optimizers[0].param_groups[0]['lr']
    print('learning rate = %.7f' % lr)


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
# ############################################################define loss functions###############
criterionGAN = losses.GANLoss()
criterionL1 = torch.nn.L1Loss()
criterionL2 = torch.nn.MSELoss()
criterion_bce = torch.nn.BCEWithLogitsLoss()
criterion_fc = losses.FocalLoss()
PerceptualLoss = losses.PerceptualLoss()
StyleLoss = losses.StyleLoss()
##################################################define_models###################################

device = torch.device('cuda')
save_dir = ''

netG_label = create_model_restoration()

model_names = []
model_names = ['G_label']

netD = define_D(input_nc=3, ndf=64, n_layers_D=3, norm='batch', init_type='normal', gpu_ids=[0],init_gain=0.02)
netF = define_D(input_nc=3, ndf=64, n_layers_D=3, norm='batch', init_type='normal', gpu_ids=[0],init_gain=0.02)

print_network(netG_label)
print_network(netD)
print_network(netF)

model_names.append('D')
model_names.append('F')

####################################################################################
# initialize optimizers
schedulers = []
optimizers = []
optimizer_G_label = torch.optim.Adam(netG_label.parameters(),lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(netD.parameters(),lr=0.0002, betas=(0.5, 0.999))
optimizer_F = torch.optim.Adam(netF.parameters(),lr=0.0002, betas=(0.5, 0.999))
optimizers.append(optimizer_G_label)
optimizers.append(optimizer_D)
optimizers.append(optimizer_F)
for optimizer in optimizers:
   schedulers.append(get_scheduler(optimizer, opt))
###################################################

for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    print(epoch)
    epoch_start_time = time.time()
    epoch_iter = 0
    epoch_loss = 0
    train_id = 1
    for i, data in enumerate(tqdm(train_loader), 0):
        total_steps += 1
        epoch_iter += 1
        target = data[0]
        input_ = data[1]
        mask1 = data[2]
        mask2 = data[3]
        mask = data[4]

        input = input_.to(torch.device('cuda'))
        target = target.to(torch.device('cuda'))
        mask1 = mask1.to(torch.device('cuda'))
        mask2 = mask2.to(torch.device('cuda'))
        mask = mask.to(torch.device('cuda'))
        Gt_Local = target.to(torch.device('cuda'))

        # define local area which send to the local discriminator
        crop_x = random.randint(0, 191)
        crop_y = random.randint(0, 191)
        input_local = input[:, :, crop_x:crop_x + 64, crop_y:crop_y + 64]
        mask_local = mask[:, :, crop_x:crop_x + 64, crop_y:crop_y + 64]
        Gt_Local = Gt_Local[:, :, crop_x:crop_x + 64, crop_y:crop_y + 64]
       #######################forward############################################
        fake_out_label = netG_label(input, mask, mask1, mask2)
        ####################################optimize################################
        # Optimize the D and F first
        set_requires_grad(netF, True)
        set_requires_grad(netD, True)
        set_requires_grad(netG_label, False)
        optimizer_D.zero_grad()
        optimizer_F.zero_grad()

        fake_AB = fake_out_label[0]
        real_AB = target
        real_local = Gt_Local
        fake_local = fake_AB[:, :, crop_x:crop_x + 64, crop_y:crop_y + 64]

        # Global Discriminator
        pred_fake = netD(fake_AB.detach())
        pred_real = netD(real_AB)
        loss_D_fake = criterionGAN(pred_fake, pred_real, True)

        # Local discriminator
        pred_fake_F = netF(fake_local.detach())
        pred_real_F = netF(real_local)
        loss_F_fake = criterionGAN(pred_fake_F, pred_real_F, True)

        loss_D = loss_D_fake + loss_F_fake
        loss_D.backward()
        optimizer_D.step()
        optimizer_F.step()

        set_requires_grad(netF, False)
        set_requires_grad(netD, False)
        set_requires_grad(netG_label, True)
        optimizer_G_label.zero_grad()
      
        real_AB = target
        fake_AB_1 = fake_out_label[0]
        real_local = Gt_Local
        fake_local_1= fake_AB_1[:, :, crop_x:crop_x + 64, crop_y:crop_y + 64]
       
        # Global discriminator
        pred_real_1 = netD(real_AB)
        pred_fake_1 = netD(fake_AB_1)
        
        # Local discriminator
        pred_real_F_1 = netF(real_local)
        pred_fake_f_1 = netF(fake_local_1)
      

        loss_G_GAN_1 = criterionGAN(pred_fake_1, pred_real_1, False) + criterionGAN(pred_fake_f_1, pred_real_F_1, False)
        loss_L1_1 = criterionL1(fake_out_label[0], target)
        Perceptual_loss_1 = PerceptualLoss(fake_out_label[0], target)
        Style_Loss_1 = StyleLoss(fake_out_label[0], target)
        total_loss = loss_L1_1 * 1 + loss_G_GAN_1*0.2+Perceptual_loss_1 * 0.2 + Style_Loss_1 * 250
        print('gan_loss', total_loss)
        total_loss.backward(retain_graph=True)
        optimizer_G_label.step()
        ######################################optimize##################################
        def get_current_visuals():

            input_image_label = input.data.cpu()
            stage_1 = fake_out_label[0].data.cpu()
            return input_image_label, stage_1

        def get_current_errors():
            return OrderedDict([
                                ('total_loss', total_loss.data),
                                ])
    #####################################################################
        with torch.no_grad():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            input_image_label, stage_1= get_current_visuals()
            image_out = torch.cat([input_image_label, stage_1], 0)
            grid = torchvision.utils.make_grid(image_out)
            file_name = ''
            torchvision.utils.save_image(grid, file_name, nrow=1)
        if total_steps % 100 == 0:
            errors = get_current_errors()
            writer.add_scalar('total_loss', errors['total_loss'], total_steps +1)
    save_mode_path = os.path.join(save_dir, 'G' + str(epoch) + '.pth')
    torch.save({'net': netG_label.state_dict()}, save_mode_path)
    update_learning_rate()


