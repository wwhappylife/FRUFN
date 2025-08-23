import os
import argparse
from tqdm import tqdm
import torch
import joblib
import torch.backends.cudnn as cudnn
# optim三
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

import torchvision.transforms as transforms
# dataloader
from torch.utils.data import DataLoader, Dataset
from dataset import Get_MEF_Dataset
# model
from ufet_image import UF_Net as Unfolding_Net
from uf_former_4070_new import Color_Net
# loss
from losses import ssim_loss_vi 
from loss_vif import fusion_loss_med as fusion_loss_vif
from loss_vif import L_GT_Grad
from utils import cal_psnr, compute_ssim_rgb
import cv2
import numpy as np


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

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='model', help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=320, type=int)
    parser.add_argument('--ema_decay', default=0.999, type=float)
    parser.add_argument('--use_ema', action='store_false', default=True, help='use EMA model')
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--gamma', default=0.5, type=int)
    parser.add_argument('--ubatch_size', default=4, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float)
    parser.add_argument('--weight', default=[1,1,0.0001, 0.0002], type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)

    args = parser.parse_args()

    return args


class GetDataset(Dataset):
    def __init__(self, training_dir_ir, training_dir_vi, training_dir_gt, ir_name_list, vi_name_list, gt_name_list, transform=None):
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
        ir = cv2.imread(self.training_dir_ir + self.ir_name_list[index])
        ir = cv2.cvtColor(ir, cv2.COLOR_BGR2YCrCb)

        vi = cv2.imread(self.training_dir_vi + self.vi_name_list[index])
        vi = cv2.cvtColor(vi, cv2.COLOR_BGR2YCrCb)
        gt = cv2.imread(self.training_dir_gt + self.ir_name_list[index])

        # ------------------To tensor------------------#
        if self.transform is not None:
            tran = self.transform
            ir = tran(ir)
            vi = tran(vi)
            gt = tran(gt)

            return ir,vi,gt

    def __len__(self):
        return len(self.ir_name_list)

def train(train_loader_mef,model,c_model, optimizer,epoch,scheduler=None):
    
    model.train()
    for i, (under,over,gt)  in tqdm(enumerate(train_loader_mef), total=len(train_loader_mef)):
        
        under = under.to(device)
        undery = under[:,0:1,:,:]
        undercrcb = under[:,-2:,:]
        over = over.to(device)
        overcrcb = over[:,-2:,:]
        overy = over[:,0:1,:,:]
        gt = gt.to(device)
        gty = gt[:,0:1,:,:]
        gtcrcb = gt[:,-2:,:]
        
        optimizer.zero_grad()
        
        outy,out_undery,out_overy,_ = model(undery, overy)
        outy = outy.detach()
        outcrcb = c_model(torch.cat((outy,undercrcb,overcrcb),dim=1))
    
        mef_loss = ssim_loss_vi(outy,gty)
        color_loss = l1_loss(gtcrcb,outcrcb)
        total_loss = color_loss
        print("Total Loss: {:.2e}".format(total_loss.item()))

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e-4, norm_type=2)
        optimizer.step()
        model.zero_grad()
        #c_model.zero_grad()
        #if i == 1:
        #    break

def test(scheduler_f, model,c_model, test_loader_ir):
    avg_psnr = 0.0
    avg_ssim = 0.0
    torch.set_grad_enabled(False)
    epoch = scheduler_f.last_epoch
    model.eval()
    print('\nEvaluation:')
     
    for i, (under,over,gt)  in tqdm(enumerate(test_loader_ir), total=len(test_loader_ir)):
        with torch.no_grad():
            under = under.to(device)
            over = over.to(device)
            gt = gt.to(device)
            factor = 32
            if under.shape[-2]%factor != 0:
                new_h = under.shape[-2] - under.shape[-2]%factor
                under = under[:,:,:new_h,:]
                over = over[:,:,:new_h,:]
                gt = gt[:,:,:new_h,:]
            if under.shape[-1]%factor != 0:
                new_w = under.shape[-1] - under.shape[-1]%factor
                under = under[:,:,:,:new_w]
                over = over[:,:,:,:new_w]
                gt = gt[:,:,:,:new_w]
            undery = under[:,0:1, :, :]
            undercr = under[:,1:2, :, :]
            undercb = under[:,2:3, :, :]
            overy = over[:,0:1, :,:]
            overcr = over[:,1:2, :,:]
            overcb = over[:,2:3, :,:]
            gty = gt[:,0:1, :, :]

            cr = torch.cat((overcr,undercr),dim=0)
            cb = torch.cat((undercb,overcb),dim=0)
       
            EPS = 1e-6
            w_cr = (torch.abs(cr) + EPS) / torch.sum(torch.abs(cr) + EPS, dim=0)
            w_cb = (torch.abs(cb) + EPS) / torch.sum(torch.abs(cb) + EPS, dim=0)
            fcr = torch.sum(w_cr * cr, dim=0, keepdim=True).clamp(-1, 1)
            fcb = torch.sum(w_cb * cb, dim=0, keepdim=True).clamp(-1, 1)

            with torch.no_grad():
                
                outy,_,_,_ = model(undery,overy)
                outcrcb = c_model(torch.cat((outy,undercr,undercb,overcr,overcb),dim=1))
                out = torch.cat((outy,outcrcb),dim=1)
                out = torch.clamp(out,0,1)
        
                out = np.squeeze(out.detach().permute(0,2,3,1).cpu().numpy())
                gt = gt.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
                
                avg_psnr += cal_psnr(out,gt)
                avg_ssim += compute_ssim_rgb(out,gt)
                
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

   # supervised train_data 2
    train_dir_f = "/home/wangwu1/hybird_mfi/new_mef_dataset/source/" # forground
    train_dir_g = "/home/wangwu1/hybird_mfi/new_mef_dataset/label/" # gt

    transform_train = transforms.Compose([transforms.ToTensor(),
                                          ])
    dataset_train_ir1 = Get_MEF_Dataset(train_dir_f,train_dir_g,
                                                  transform=transform_train,patch_size=96)

    train_loader_mef = DataLoader(dataset_train_ir1,
                              shuffle=True,
                              batch_size=4)
    # test data
    test_dir_under = "/home/wangwu1/hybird_mfi/meif_dataset/test/source1/" # under
    under_name_list = os.listdir(test_dir_under) 
    test_dir_over = "/home/wangwu1/hybird_mfi/meif_dataset/test/source2/" # over
    over_name_list = os.listdir(test_dir_over) 
    test_dir_gt = "/home/wangwu1/hybird_mfi/meif_dataset/test/label/" # gt
    gt_name_list = os.listdir(test_dir_gt)
    dataset_test_ir = GetDataset(test_dir_under, test_dir_over, test_dir_gt, under_name_list, over_name_list, gt_name_list,
                                                  transform=transform_train)
    test_loader_ir = DataLoader(dataset_test_ir,
                              shuffle=True,
                              batch_size=1)
    

    model = Unfolding_Net(base_filter=32, num_spectral=1).to(device)
    model_path = "./models/model/ufnet_meif_deeper.pth"
    model.load_state_dict(torch.load(model_path),strict=True)
    c_model = Color_Net(num_spectral=2, base_filter=48).to(device)

    milestones = []
    for i in range(1, args.epochs+1):
        if i == 200:
            milestones.append(i)
        if i == 300:
            milestones.append(i)
        if i == 400:
            milestones.append(i)
        if i == 500:
            milestones.append(i)

    optimizer = optim.Adam(c_model.parameters(), lr=args.lr,
                           betas=args.betas, eps=args.eps)
    
    scheduler_f = lrs.MultiStepLR(optimizer, milestones, args.gamma)

    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch+1, args.epochs))
        model.zero_grad()
        c_model.zero_grad()
        train_log = train(train_loader_mef,model,c_model, optimizer,epoch)

        scheduler_f.step()
        if (epoch+1) % 1 == 0:
            test(scheduler_f, model,c_model, test_loader_ir)
            torch.save(c_model.state_dict(), 'models/%s/model_{}.pth'.format(epoch+1) %args.name)

if __name__ == '__main__':
    ckt = {'psnr_epoch':0, 'ssim_epoch':0, 'ssim':0.0, 'psnr':0.0} 
    i_iter = 0
    main()
    

