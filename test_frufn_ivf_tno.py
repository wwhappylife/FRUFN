from PIL import Image
import numpy as np
import os
import torch

import glob
import time
import imageio

import torchvision.transforms as transforms
from thop import clever_format
from thop import profile
from torch.utils.data import DataLoader, Dataset

from math import exp
import torch.nn.functional as F
from torch.autograd import Variable
from ema_model.ema import ModelEMA
from ufnet_image import UF_Net as Unfolding_Net
from tqdm import tqdm
import argparse
import cv2
from utils import print_network,ins_norm,out_norm

device = torch.device('cuda:0')


def parse_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    return args

class GetDataset(Dataset):
    def __init__(self, training_dir_ir, training_dir_vi, traing_dir_gt, ir_name_list, vi_name_list, gt_name_list, transform=None):
        super(GetDataset, self).__init__()
        ir_name_list.sort()
        vi_name_list.sort()
        gt_name_list.sort()
        self.training_dir_ir = training_dir_ir
        self.training_dir_vi = training_dir_vi
        self.training_dir_gt = training_dir_gt
        self.ir_name_list = ir_name_list
        self.vi_name_list = vi_name_list
        self.gt_name_list = gt_name_list
        self.transform = transform

    def __getitem__(self, index):
        ir = cv2.imread(self.training_dir_ir + self.ir_name_list[index],cv2.IMREAD_GRAYSCALE)
        vi = cv2.imread(self.training_dir_vi + self.vi_name_list[index])
        vi = cv2.cvtColor(vi, cv2.COLOR_BGR2YCrCb)

        # ------------------To tensor------------------#
        if self.transform is not None:
            tran = self.transform
            ir = tran(ir)
            vi = tran(vi)

            return ir,vi, self.ir_name_list[index]

    def __len__(self):
        return len(self.ir_name_list)
    
args = parse_args()
training_dir_ir = "/home/wangwu1/general_image_fusion/Test_TNO/ir/"
#training_dir_ir = "/home/wangwu/MIF/Harvard_med/CT-MRI/test/ct/" # for mri-ct task
#training_dir_ir = "/home/wangwu/MIF/Harvard_med/PET-MRI/test/mri/" # for mri-pet task
#training_dir_ir = "/home/wangwu/MIF/Harvard_med/SPECT-MRI/test/mri/" # for mri-spect task
ir_name_list = os.listdir(training_dir_ir) 

training_dir_vi = "/home/wangwu1/general_image_fusion/Test_TNO/vi/"
#training_dir_vi = "/home/wangwu/MIF/Harvard_med/CT-MRI/test/mri/"
#training_dir_vi = "/home/wangwu/MIF/Harvard_med/PET-MRI/test/pet/"
#training_dir_vi = "/home/wangwu/MIF/Harvard_med/SPECT-MRI/test/spect/"
vi_name_list = os.listdir(training_dir_vi)

transform_train = transforms.Compose([transforms.ToTensor(),
                                          ])
dataset_test_dir = GetDataset(training_dir_ir, training_dir_vi, ir_name_list, vi_name_list, gt_name_list,
                                                  transform=transform_train)
test_loader = DataLoader(dataset_test_dir,
                              shuffle=False,
                              batch_size=1)

# test ufnet
model = Unfolding_Net(base_filter=32, num_spectral=1).to(device)
model_path = "./models/model/ufnet_deeper_ivf_sd.pth"
model.load_state_dict(torch.load(model_path))
def fusion():
    
    fl = 0.0
    pa = 0.0
    tic = time.time()
    for i, (ir,vi, name)  in tqdm(enumerate(test_loader), total=len(test_loader)):
        

        ir = ir.to(device)
        vi = vi.to(device)
        factor = 8
        if ir.shape[-2]%factor != 0:
            new_h = ir.shape[-2] - ir.shape[-2]%factor
            ir = ir[:,:,:new_h,:]
            vi = vi[:,:,:new_h,:]
        if ir.shape[-1]%factor != 0:
            new_w = ir.shape[-1] - ir.shape[-1]%factor
            ir = ir[:,:,:,:new_w]
            vi = vi[:,:,:,:new_w]
            
        viy = vi[:,0:1, :, :]
        vicr = vi[:,1:2, :, :]
        vicb = vi[:,2:3, :, :]

        with torch.no_grad():
            model.eval()
            out,_,_,mask = model.forward(viy, ir)
            out = torch.clamp(out,0,1)
            out = np.squeeze(out.detach().permute(0,2,3,1).cpu().numpy())
            result = out * 255
            result = np.clip(result,0,255)
            result = result.astype(np.uint8)

            print(name[0])
            cv2.imwrite('./tno_result/'+name[0], result)
           
    toc = time.time()
    print('end {}{}'.format(i // 10, i % 10), ', time:{}'.format(toc - tic))
    fl, pa = clever_format([fl,pa],"%.3f")
    print(fl, pa)



if __name__ == '__main__':

    fusion()
