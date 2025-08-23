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
from ufformer_deq import UFormer_DEQ as Unfolding_Net
from tqdm import tqdm
import argparse
import cv2
from utils import print_network,ins_norm,out_norm

device = torch.device('cuda:0')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='model_inn', help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--gamma', default=0.5, type=int)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--ema_decay', default=0.999, type=float)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--wnorm', action='store_true', help="use weight normalization")
    parser.add_argument('--f_solver', default='anderson', type=str, choices=['anderson', 'broyden', 'naive_solver'],
                        help='forward solver to use (only anderson and broyden supported now)')
    parser.add_argument('--b_solver', default='broyden', type=str, choices=['anderson', 'broyden', 'naive_solver'],
                        help='backward solver to use')
    parser.add_argument('--f_thres', type=int, default=5, help='forward pass solver threshold')
    parser.add_argument('--b_thres', type=int, default=10, help='backward pass solver threshold')
    parser.add_argument('--f_eps', type=float, default=1e-3, help='forward pass solver stopping criterion')
    parser.add_argument('--b_eps', type=float, default=1e-3, help='backward pass solver stopping criterion')
    parser.add_argument('--f_stop_mode', type=str, default="abs", help="forward pass fixed-point convergence stop mode")
    parser.add_argument('--b_stop_mode', type=str, default="abs", help="backward pass fixed-point convergence stop mode")
    parser.add_argument('--eval_factor', type=float, default=1, help="factor to scale up the f_thres at test for better convergence.")
    parser.add_argument('--eval_f_thres', type=int, default=0, help="directly set the f_thres at test.")

    parser.add_argument('--indexing_core', action='store_true', help="use the indexing core implementation.")
    parser.add_argument('--ift', action='store_true', help="use implicit differentiation.")
    parser.add_argument('--safe_ift', action='store_true', help="use a safer function for IFT to avoid potential segment fault in older pytorch versions.")
    parser.add_argument('--n_losses', type=int, default=2, help="number of loss terms (uniform spaced, 1 + fixed point correction).")
    parser.add_argument('--indexing', type=int, nargs='+', default=[], help="indexing for fixed point correction.")
    parser.add_argument('--phantom_grad', type=int, nargs='+', default=[1], help="steps of Phantom Grad")
    parser.add_argument('--tau', type=float, default=1, help="damping factor for unrolled Phantom Grad")
    parser.add_argument('--sup_all', action='store_true', help="supervise all the trajectories by Phantom Grad.")

    args = parser.parse_args()

    return args

class GetDataset(Dataset):
    def __init__(self, training_dir_ir, training_dir_vi, ir_name_list, vi_name_list, gt_name_list, transform=None):
        super(GetDataset, self).__init__()
        ir_name_list.sort()
        vi_name_list.sort()
        self.training_dir_ir = training_dir_ir
        self.training_dir_vi = training_dir_vi
        self.ir_name_list = ir_name_list
        self.vi_name_list = vi_name_list
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
vi_name_list = os.listdir(training_dir_vi)

transform_train = transforms.Compose([transforms.ToTensor(),
                                          ])

dataset_test_dir = GetDataset(training_dir_ir, training_dir_vi, ir_name_list, vi_name_list, gt_name_list,
                                                  transform=transform_train)
test_loader = DataLoader(dataset_test_dir,
                              shuffle=False,
                              batch_size=1)

# test ufnet
model = Unfolding_Net(args, num_channel=1, base_filter=32, num_spectral=1).to(device)
model_path = "./models/model/ufnet_deq_deeper_ivf.pth"
model.load_state_dict(torch.load(model_path))


def fusion():
    
    fl = 0.0
    pa = 0.0
    tic = time.time()
    for i, (ir,vi name)  in tqdm(enumerate(test_loader), total=len(test_loader)):
        

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
            out,mask = model.forward(viy, ir, ir, ir)
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
