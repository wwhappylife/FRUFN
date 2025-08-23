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
import joblib
import torch.backends.cudnn as cudnn
# optim三
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

import torchvision.transforms as T
import albumentations as A
from PIL import Image
# dataloader
from torch.utils.data import DataLoader, Dataset
# model
from uf_former_4070_new import UF_Net as Unfolding_Net
from ufnet_image import UF_Net as Unfolding_Net
import random
from random import randrange

# Loss
from pytorch_ssim import ssim,tv_loss
from losses import ssim_loss_ir,ssim_loss_vi , sf_loss_ir, sf_loss_vi, vgg_loss, L_Grad
from utils import cal_psnr, print_network, compute_ssim, mixup_data
import cv2
import random
import numpy as np

seed = 555
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device('cuda:0')
l1_loss = torch.nn.L1Loss().to(device)
swin_loss = L_Grad().to(device)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='model', help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=320, type=int)
    parser.add_argument('--gamma', default=0.5, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float)
    parser.add_argument('--weight', default=[1,1,0.0001, 0.0002], type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)

    args = parser.parse_args()

    return args

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class GetDataset(Dataset):
    def __init__(self, train_dir_f,train_dir_b,train_dir_g,train_name_list, is_patch=True, transform=None, transform_affine=None):
        super(GetDataset, self).__init__()
        self.train_name_list = train_name_list
        self.train_dir_f = train_dir_f
        self.train_dir_b = train_dir_b
        self.train_dir_g = train_dir_g
        self.transform = transform
        self.is_patch = is_patch
        self.affine_t = transform_affine

    def __getitem__(self, index):

        train_name = self.train_name_list[index]
        f = cv2.imread(self.train_dir_f + train_name)
        f = cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb)
        f = f[:, :, 0:1]
        if "A.png" in train_name:
            b = cv2.imread(self.train_dir_b + train_name.replace("A","B"))
            b = cv2.cvtColor(b, cv2.COLOR_BGR2YCrCb)
            b = b[:, :, 0:1]
            clear = cv2.imread(self.train_dir_g + train_name.replace("A","F"))
            clear = cv2.cvtColor(clear, cv2.COLOR_BGR2YCrCb)
            clear = clear[:, :, 0:1]
        else:
            b = cv2.imread(self.train_dir_b + train_name)
            b = cv2.cvtColor(b, cv2.COLOR_BGR2YCrCb)
            b = b[:, :, 0:1]
            clear = cv2.imread(self.train_dir_g + train_name)
            clear = cv2.cvtColor(clear, cv2.COLOR_BGR2YCrCb)
            clear = clear[:, :, 0:1]

        if self.is_patch:
            f, b, clear = self.get_patch(f, b,clear, patch_size=128)
        # ------------------To tensor------------------#
        if self.transform is None or self.affine_t is None:
            t = T.ToTensor()
            f = t(f)
            b = t(b)
            clear = t(clear)
        else:
            transformed = self.transform(image=f,source2=b,label=clear)
            f = transformed['image']
            b = transformed['source2']
            clear = transformed['label']
            
            t = T.ToTensor()
            f = t(f)
            b = t(b)
            clear = t(clear)

            f = self.affine_t(f)
            b = self.affine_t(b)

        return f,b,clear

    def __len__(self):
        return len(self.train_name_list)
    
    def get_patch(self, img_in, img_in1,img_tar, patch_size):
        h, w = img_in.shape[:2]
        stride = patch_size
        x = random.randint(0, w - stride)
        y = random.randint(0, h - stride)
        img_in = img_in[y:y + stride, x:x + stride, :]
        img_in1 = img_in1[y:y + stride, x:x + stride, :]
        img_tar = img_tar[y:y + stride, x:x + stride, :]

        return img_in,img_in1, img_tar



class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(args, train_loader_ir, model, criterion_ssim_ir,  criterion_ssim_vi, criterion_sf_ir,criterion_sf_vi,optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    losses_ssim_ir = AverageMeter()
    losses_ssim_vi = AverageMeter()
    losses_sf_ir = AverageMeter()
    losses_sf_vi = AverageMeter()
    weight = args.weight
    model.train()
    
    for i, (ir,vi,clear)  in tqdm(enumerate(train_loader_ir), total=len(train_loader_ir)):
        ir = ir.to(device)
        vi = vi.to(device)
        clear = clear.to(device)
        
        out,out_vi,out_ir,_ = model.forward(vi,ir)
        loss_ssim_ir = criterion_ssim_ir(out,ir)
        loss_ssim_vi= criterion_ssim_vi(out,vi) 
        
        loss_sf_vi = 0.0001 * criterion_sf_vi(out, clear) 
        loss_sf_ir= weight[2] * criterion_sf_ir(out, clear)
        if epoch < 2:
            loss = l1_loss(out_ir,ir) + l1_loss(out_vi,vi)
        else:
            loss = 2*criterion_ssim_vi(out,clear)
        print("Loss: {:.2e}".format(loss.item()))
        
        losses.update(loss.item(), ir.size(0))
        losses_ssim_ir.update(loss_ssim_ir.item(), ir.size(0))
        losses_ssim_vi.update(loss_ssim_vi.item(), ir.size(0))
        losses_sf_ir.update(loss_sf_ir.item(), ir.size(0))
        losses_sf_vi.update(loss_sf_vi.item(), ir.size(0))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e-4, norm_type=2)
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('loss_ssim_ir', losses_ssim_ir.avg),
        ('loss_ssim_vi', losses_ssim_vi.avg),
        ('loss_sf_ir', losses_sf_ir.avg),
        ('loss_sf_vi', losses_sf_vi.avg)
    ])
    return log

def test(scheduler_f, model, test_loader_ir):
    avg_psnr = 0.0
    avg_ssim = 0.0
    torch.set_grad_enabled(False)
    epoch = scheduler_f.last_epoch
    model.eval()
    print('\nEvaluation:')
     
    for i, (ir,vi,clear)  in tqdm(enumerate(test_loader_ir), total=len(test_loader_ir)):
        with torch.no_grad():
            ir = ir.to(device)
            vi = vi.to(device)
            clear = clear.to(device)
            factor = 8
            if ir.shape[-2]%factor != 0:
                new_h = ir.shape[-2] - ir.shape[-2]%factor
                ir = ir[:,:,:new_h,:]
                vi = vi[:,:,:new_h,:]
                clear = clear[:,:,:new_h,:]
            if ir.shape[-1]%factor != 0:
                new_w = ir.shape[-1] - ir.shape[-1]%factor
                ir = ir[:,:,:,:new_w]
                vi = vi[:,:,:,:new_w]
                clear = clear[:,:,:,:new_w]
            
            with torch.no_grad():
                out,_,_,_ = model.forward(vi,ir)
                out = out.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1,0)
                clear = clear.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1,0)
                avg_psnr += cal_psnr(out,clear)
                avg_ssim += compute_ssim(out,clear)
                
    avg_psnr = avg_psnr / len(test_loader_ir)
    avg_ssim = avg_ssim / len(test_loader_ir)
    
    if avg_psnr >= ckt['psnr']:
        ckt['epoch'] = epoch
        ckt['psnr'] = avg_psnr
    print("===> Avg.PSNR: {:.4f} dB || ssim: {:.4f} || Best.PSNR: {:.4f} dB || Epoch: {}"
          .format(avg_psnr, avg_ssim, ckt['psnr'], ckt['epoch']))
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

    # train_data
    train_dir_f = "/home/wangwu1/hybird_mfi/train/big_train/imageA/" # forground
    train_dir_b = "/home/wangwu1/hybird_mfi/train/big_train/imageB/" # background
    train_dir_g = "/home/wangwu1/hybird_mfi/train/big_train/Fusion/" # gt
    train_name_list = os.listdir(train_dir_f)
    print(train_name_list)
    
    transform_train = A.Compose([
        A.RandomRotate90(),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5)],
        additional_targets={'source2':'image','label':'image'}
    )
    dataset_train_ir = GetDataset(train_dir_f,train_dir_b,train_dir_g,train_name_list,
                                                  transform=transform_train)

    train_loader_ir = DataLoader(dataset_train_ir,
                              shuffle=True,
                              batch_size=args.batch_size)

    # test data
    test_dir_f = "/home/wangwu1/hybird_mfi/test/imageA/" # forground
    test_dir_b = "/home/wangwu1/hybird_mfi/test/imageB/" # background
    test_dir_g = "/home/wangwu1/hybird_mfi/test/Fusion/" # gt
    test_name_list = os.listdir(test_dir_f)
    
    dataset_test_ir = GetDataset(test_dir_f,test_dir_b,test_dir_g,test_name_list,is_patch=False,
                                                  transform=transform_train)
    test_loader_ir = DataLoader(dataset_test_ir,
                              shuffle=True,
                              batch_size=1)
    model = Unfolding_Net(base_filter=32, num_spectral=1).to(device)
    print_network(model)

    criterion_ssim_ir = ssim_loss_ir
    criterion_ssim_vi = ssim_loss_vi
    criterion_sf_ir = sf_loss_ir
    criterion_sf_vi= sf_loss_vi

    milestones = []
    for i in range(1, args.epochs+1):
        if i == 50:
            milestones.append(i)
        if i == 100:
            milestones.append(i)
        if i == 200:
            milestones.append(i)
        if i == 300:
            milestones.append(i)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=args.betas, eps=args.eps)
    scheduler_f = lrs.MultiStepLR(optimizer, milestones, args.gamma)
    log = pd.DataFrame(index=[],
                       columns=['epoch',

                                'loss',
                                'loss_ssim_ir',
                                'loss_ssim_vi',
                                'loss_sf_ir',
                                'loss_sf_vi'
                                ])

    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch+1, args.epochs))

        train_log = train(args, train_loader_ir, model, criterion_ssim_ir,  criterion_ssim_vi, criterion_sf_ir,  criterion_sf_vi, optimizer, epoch)     # 训练集

        print('loss: %.4f - loss_ssim_ir: %.4f - loss_ssim_vi: %.4f - loss_sf_ir: %.4f- loss_sf_vi: %.4f'
              % (train_log['loss'],
                 train_log['loss_ssim_ir'],
                 train_log['loss_ssim_vi'],
                 train_log['loss_sf_ir'],
                 train_log['loss_sf_vi']
                 ))

        tmp = pd.Series([
            epoch + 1,

            train_log['loss'],
            train_log['loss_ssim_ir'],
            train_log['loss_ssim_vi'],
            train_log['loss_sf_ir'],
            train_log['loss_sf_vi']
        ], index=['epoch', 'loss', 'loss_ssim_ir', 'loss_ssim_vi', 'loss_sf_ir', 'loss_sf_vi'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' %args.name, index=False)

        scheduler_f.step()
        if (epoch+1) % 1 == 0:
            test(scheduler_f, model, test_loader_ir)
            torch.save(model.state_dict(), 'models/%s/model_{}.pth'.format(epoch+1) %args.name)


if __name__ == '__main__':
    ckt = {'epoch':0, 'psnr':0.0} 
    main()


