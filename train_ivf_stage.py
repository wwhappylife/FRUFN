import os
import argparse

from tqdm import tqdm
import pandas as pd

import glob

from math import exp
import torch.nn.functional as F
from torch.autograd import Variable

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import torch.backends.cudnn as cudnn
# optim三
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

import torchvision.transforms as transforms
from PIL import Image
# dataloader
from torch.utils.data import DataLoader, Dataset
from dataset import Get_UDataset
from ufnet_image import UF_Net as Unfolding_Net
import random
from random import randrange
# loss
from losses import ssim_loss_ir,ssim_loss_vi,sf_loss_ir,sf_loss_vi
from loss_vif import L_Intensity, L_Grad
from loss_vif import fusion_loss_med as fusion_loss_vif
from loss_vif import L_GT_Grad as L_GT_Grad

from MEFSSIM.lossfunction import MEFSSIM_Loss
from losses import Batch_Log_likelyhood as Log_likelyhood
from utils import cal_psnr, compute_ssim
from ema_model.ema import ModelEMA
import cv2
import numpy as np

from sklearn import metrics as mr

from itertools import cycle

seed = 555
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark  = False


device = torch.device('cuda:0')
l1_loss = torch.nn.L1Loss().to(device)
uns_loss = fusion_loss_vif().to(device)
l_grad = L_GT_Grad().to(device)
lf_grad = L_Grad().to(device)
intensity_loss = L_Intensity()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='model', help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=320, type=int)
    parser.add_argument('--ema_decay', default=0.999, type=float)
    parser.add_argument('--use_ema', action='store_false', default=True, help='use EMA model')
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--gamma', default=0.5, type=int)
    parser.add_argument('--ubatch_size', default=2, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float)
    parser.add_argument('--weight', default=[1,1,0.0006, 0.00025], type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)

    args = parser.parse_args()

    return args


def train(args,train_loader_u,model,ema_model, optimizer,epoch):
    
    model.train()
    for i, batch_u in tqdm(enumerate(train_loader_u)):
        u1,u2 = Variable(batch_u[0]), Variable(batch_u[1])
        u1 = u1.to(device)
        u2 = u2.to(device)
        
        optimizer.zero_grad()
        
        wau_out_clear,out_u1,out_u2,mask_loss = model(u1, u2)
        out_clear,_,_,_ = ema_model(u1,u2)
        unsuper_loss = uns_loss(u1,u2,wau_out_clear)
        consis_loss = l_grad(out_clear.detach(),wau_out_clear)
        total_loss = unsuper_loss + 40*consis_loss
        print("total_loss:{:.2e} Unsuper Loss: {:.2e}".format(total_loss.item(),unsuper_loss.item()))

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e-2, norm_type=2)
        optimizer.step()
        #if args.use_ema:
        #    ema_model.update(model)
        model.zero_grad()


def test(scheduler_f, model, test_loader_ir):
    avg_psnr = 0.0
    avg_ssim = 0.0
    torch.set_grad_enabled(False)
    epoch = scheduler_f.last_epoch
    print('\nEvaluation:')
     
    for i, (ir,vi)  in tqdm(enumerate(test_loader_ir), total=len(test_loader_ir)):
        ir = ir.to(device)
        vi = vi.to(device)
        factor = 128
        if ir.shape[-2]%factor != 0:
            new_h = ir.shape[-2] - ir.shape[-2]%factor
            ir = ir[:,:,:new_h,:]
            vi = vi[:,:,:new_h,:]
        if ir.shape[-1]%factor != 0:
            new_w = ir.shape[-1] - ir.shape[-1]%factor
            ir = ir[:,:,:,:new_w]
            vi = vi[:,:,:,:new_w]

        with torch.no_grad():
            model.eval()
            out,_,_,_,_ = model(ir,vi)
            out = out.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1,0)
            ir = ir.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1,0)
            vi = vi.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1,0)
                
            avg_psnr += cal_psnr(out,vi)+cal_psnr(out,ir)
            avg_ssim += compute_ssim(out,vi)+compute_ssim(out,ir)
                
    avg_psnr = avg_psnr / len(test_loader_ir)
    avg_ssim = avg_ssim / len(test_loader_ir)
    
    if avg_psnr >= ckt['psnr']:
        ckt['psnr_epoch'] = epoch
        ckt['psnr'] = avg_psnr
    if avg_ssim >= ckt['ssim']:
        ckt['ssim_epoch'] = epoch
        ckt['ssim'] = avg_ssim
    print("===> Avg.PSNR: {:.4f} dB || ssim: {:.4f} || Best.PSNR: {:.4f} dB || Best_PSNR_Epoch: {}|| Best.SSIM: {:.4f} || Best_SSIM_Epoch: {}"
          .format(avg_psnr, avg_ssim, ckt['psnr'], ckt['psnr_epoch'], ckt['ssim'], ckt['ssim_epoch']))
    torch.set_grad_enabled(True)

def main():
    args = parse_args()

    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))

    joblib.dump(args, 'models/%s/args.pkl' %args.name)
    cudnn.benchmark = True

    # unsupervised train_data
    train_dir_u1 = "/home/wangwu1/MFI-WHU-master/mixed_ivf/imgb_less/" # vi
    train_dir_u2 = "/home/wangwu1/MFI-WHU-master/mixed_ivf/imga_more/" # ir
    train_name_list_u = os.listdir(train_dir_u1)

    transform_train = transforms.Compose([transforms.ToTensor(),
                                          ])
    dataset_train_u = Get_UDataset(train_dir_u1,train_dir_u2,train_name_list_u,
                                                  transform=transform_train)

    train_loader_u = DataLoader(dataset_train_u,
                              shuffle=True,
                              batch_size=args.ubatch_size)
    # test data
    test_dir_f = "/home/wangwu1/general_image_fusion/Test_TNO/ir/" # forground
    test_dir_b = "/home/wangwu1/general_image_fusion/Test_TNO/vi/" # background
    test_name_list = os.listdir(test_dir_f)
    dataset_test_ir = Get_UDataset(test_dir_f,test_dir_b,test_name_list,is_patch=False, is_tno=True,
                                                  transform=transform_train)
    test_loader_ir = DataLoader(dataset_test_ir,
                              shuffle=True,
                              batch_size=1)
    

    t_model = Unfolding_Net(base_filter=32, num_spectral=1).to(device)
    model = Unfolding_Net(base_filter=32, num_spectral=1).to(device)
    model_path = "./models/model/ufnet_meif_deeper.pth"
    t_model.load_state_dict(torch.load(model_path),strict=True)
    model.load_state_dict(torch.load(model_path),strict=True)

    milestones = []
    for i in range(1, args.epochs+1):
        if i == 200:
            milestones.append(i)
        if i == 300:
            milestones.append(i)

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=args.betas, eps=args.eps)
    scheduler_f = lrs.MultiStepLR(optimizer, milestones, args.gamma)

    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch+1, args.epochs))
        model.zero_grad()
        train_log = train(args,train_loader_u,model,t_model, optimizer,epoch)

        scheduler_f.step()
        if (epoch+1) % 1 == 0:
            test(scheduler_f, model, test_loader_ir)
            torch.save(model.state_dict(), 'models/%s/model_{}.pth'.format(epoch+1) %args.name)

if __name__ == '__main__':
    ckt = {'psnr_epoch':0, 'ssim_epoch':0, 'ssim':0.0, 'psnr':0.0} 
    i_iter = 0
    main()
    

