"""
使用15block
不在使用BN
"""
import sys
sys.path.append('../')
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from lib.model import PatchDiscriminator
from lib.model import PathSRGAN
from lib.loss import PerceptionLoss as Ploss
from lib.dataset import TripleDataSet
from torch.utils.tensorboard import SummaryWriter
from lib.dataset import wrap_image_plus
from lib import utilis
from utils import epoch_test
import cv2
import torch
torch.manual_seed(0)
device_id = '8,9'
os.environ['CUDA_VISIBLE_DEVICES'] = device_id
device_num = len(device_id.split(','))
device = torch.device('cuda')

#================================
#  config path
#================================
method = 'PathSRGAN'
title = 'bightness_-30'

img_4x_path = '/mnt/diskarray/srdata/new/4x/'
img_10x_path = '/mnt/diskarray/srdata/new/10x/'
img_20x_path = '/mnt/diskarray/srdata/new/20x/'
train_names_log = '/mnt/diskarray/srdata/new/train.txt'
test_names_log = '/mnt/diskarray/srdata/new/test.txt'

checkpoints_path = f'/mnt/diskarray/mjb/SRLog_new/{method}/checkpoints/{title}/'
tensorboard_log = f'/mnt/diskarray/mjb/SRLog_new/{method}/tensorboard/{title}/'
train_img_log_path = f'/mnt/diskarray/mjb/SRLog_new/{method}/train_img/{title}/'
test_img_log_path = f'/mnt/diskarray/mjb/SRLog_new/{method}/test_img/{title}/'
# for pretrain model
D1_path = '/mnt/diskarray/mjb/SRLog/our_420_outside_15layer/checkpoints/continue_1/netD1_epoch_5_2.pth'
G_path = '/mnt/diskarray/mjb/SRLog/our_420_outside_15layer/checkpoints/continue_1/netG_epoch_7_5.pth'
D2_path = '/mnt/diskarray/mjb/SRLog/our_420_outside_15layer/checkpoints/continue_1/netD2_epoch_7_5.pth'

#=================================
#       model config 
#=================================
adver_decay = 1e-3
p_decay = 6e-3
stage1_epoch = 6
stage2_epoch = 6
batch_size = 20
img_save_step_ratio = 0.02
model_save_step_ratio = 0.2
NUM_EPOCHS = stage2_epoch+stage1_epoch
lr = 8e-5
batch_size *= device_num    # batch size  
num_workers = device_num*3

pretrain_flag = False
# no need to modify,this is designed to save appropriate log
sample_nums = utilis.get_sample_numbers(train_names_log)
batch_nums = sample_nums//batch_size
model_save_step = int(batch_nums*model_save_step_ratio)
img_log_step = int(batch_nums*img_save_step_ratio)


#check path
utilis.path_checker(checkpoints_path)
utilis.path_checker(train_img_log_path)

train_set = TripleDataSet(img_4x_path,img_10x_path,img_20x_path,train_names_log,brightness=-30)
test_set = TripleDataSet(img_4x_path,img_10x_path,img_20x_path,test_names_log,brightness=-30)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True,num_workers=num_workers)
test_loader = DataLoader(dataset = test_set,batch_size = batch_size,shuffle=False,num_workers = num_workers)

#load model
G = PathSRGAN(1).to(device)
D1 = PatchDiscriminator().to(device)
D2 = PatchDiscriminator().to(device)
ploss = Ploss(23).to(device)

#set cost function

bce_loss = torch.nn.BCELoss()
mse_loss = torch.nn.MSELoss()
# load pretrained model
if pretrain_flag:
    G.load_state_dict(torch.load(G_path)) 
    D2.load_state_dict(torch.load(D2_path))
    D1.load_state_dict(torch.load(D1_path))
    
#set mode
if device_num > 1:
    G = torch.nn.DataParallel(G).to(device)
    D1 = torch.nn.DataParallel(D1).to(device)
    D2 = torch.nn.DataParallel(D2).to(device)
    ploss = torch.nn.DataParallel(ploss).to(device)

D1.train()
D2.train()
G.train()

Writer= SummaryWriter(tensorboard_log)

########################
#1.train stage1
########################
image_save_counter = 0
optimizerG = optim.Adam(G.parameters(),lr = lr)
optimizerD1 = optim.Adam(D1.parameters(),lr = lr)

if __name__ == '__main__':
    for epoch in range(stage1_epoch):
        for i,(img_4x,img_10x,img_20x) in enumerate(train_loader):
            print('just come in ')
            ############################
            # (1) Update G
            ###########################
            img_4x = img_4x.to(device)
            img_10x = img_10x.to(device)
            gen10= G(img_4x,stage1 = True)
            print('gen10 is over')
            G.zero_grad()
            #10x监督loss
            out_labels = D1(gen10)
            valid = torch.cuda.FloatTensor(out_labels.size()).fill_(1.0)
            invalid = torch.cuda.FloatTensor(out_labels.size()).fill_(0.0)
            image_mse2 = mse_loss(gen10,img_10x)
            adver_loss2 = bce_loss(out_labels,valid)
            perception_loss2 = ploss(gen10,img_10x).mean()


            adver_loss2 = adver_decay*adver_loss2
            perception_loss2 = p_decay*perception_loss2

            g_loss = image_mse2 + adver_loss2 + perception_loss2
            g_loss.backward(retain_graph = True)
            optimizerG.step()
            ######################
            # (2) Updata D
            #####################
            #优化4x---->20x 的鉴别器 netD
            D1.zero_grad()
            real_loss = bce_loss(D1(img_10x),valid)
            fake_loss = bce_loss(out_labels,invalid)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizerD1.step()
            print('all done')
            if i % img_log_step ==0:
                tensor_list = [img_4x[0,:,:,:],gen10[0,:,:,:],img_10x[0,:,:,:],gen10[0,:,:,:],img_20x[0,:,:,:]]
                warped_image = wrap_image_plus(tensor_list)
                #G loss situation
                Writer.add_scalar('stage2/scalar/adversarial_loss2',adver_loss2, image_save_counter)
                Writer.add_scalar('stage2/scalar/mse_loss2',image_mse2, image_save_counter)
                Writer.add_scalar('stage2/scalar/perception_loss2',perception_loss2,image_save_counter)

                Writer.add_scalar('stage2/scalar/D1_loss',d_loss, image_save_counter)

                #save image log
                image_log_name = train_img_log_path+'stage2_'+str(epoch)+'_'+str(i//img_log_step)+'.tif'
                cv2.imwrite(image_log_name,warped_image)
                image_save_counter += 1

            sys.stdout.write("\r[Epoch {}/{}] [Batch {}/{}] [D1: {:.4f}] [G:{:.4f}] [adver: {:.4f}] [mse:{:.4f}] [ploss:{:.4f}]".format(epoch,NUM_EPOCHS, i, len(train_loader),d_loss.item(),
                            g_loss.item(), adver_loss2.item(), image_mse2.item(), perception_loss2.item() ))
            sys.stdout.flush()
            if i%model_save_step == 0 and i!= 0:
                torch.save(G.state_dict(), checkpoints_path+'G_%d_%d.pth' % (epoch,i // model_save_step))
                torch.save(D1.state_dict(), checkpoints_path+'D1_%d_%d.pth' % (epoch,i // model_save_step))
        lr = lr*0.6
        for param in optimizerD1.param_groups:
            param['lr'] = lr
        for param in optimizerG.param_groups:
            param['lr'] = lr
    ########################
    #2.train stage2
    ########################
    batch_size = 2
    train_set = TripleDataSet(img_4x_path,img_10x_path,img_20x_path,train_names_log,brightness=-30)
    test_set = TripleDataSet(img_4x_path,img_10x_path,img_20x_path,test_names_log,brightness=-30)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    test_loader = DataLoader(dataset = test_set,batch_size = batch_size,shuffle=False,num_workers = num_workers)
    image_save_counter = 0
    lr = 1e-4
    netG_params = utilis.need_grad(False,G,'stage1')
    optimizerG = optim.Adam(netG_params,lr = lr)
    optimizerD2 = optim.Adam(D2.parameters(),lr = lr)
    for epoch in range(stage1_epoch,stage2_epoch+stage1_epoch):
        for i,(img_4x,img_10x,img_20x) in enumerate(train_loader):
            ############################
            # (1) Update G
            ###########################
            img_4x = img_4x.to(device)
            img_10x = img_10x.to(device)
            img_20x = img_20x.to(device)
            gen10,gen20 = G(img_4x)
            G.zero_grad()
            #10x监督loss
            out_labels = D2(gen20)
            valid = torch.cuda.FloatTensor(out_labels.size()).fill_(1.0)
            invalid = torch.cuda.FloatTensor(out_labels.size()).fill_(0.0)

            image_mse2 = mse_loss(gen20,img_20x)
            adver_loss2 = bce_loss(out_labels,valid)
            perception_loss2 = ploss(gen20,img_20x).mean()


            adver_loss2 = adver_decay*adver_loss2
            perception_loss2 = p_decay*perception_loss2

            g_loss = image_mse2 + adver_loss2 + perception_loss2
            g_loss.backward(retain_graph = True)
            optimizerG.step()
            ######################
            # (2) Updata D
            #####################
            #优化4x---->20x 的鉴别器 netD
            D2.zero_grad()
            real_loss = bce_loss(D2(img_20x),valid)
            fake_loss = bce_loss(out_labels,invalid)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizerD2.step()

            if i % img_log_step ==0:
                tensor_list = [img_4x[0,:,:,:],gen10[0,:,:,:],img_10x[0,:,:,:],gen20[0,:,:,:],img_20x[0,:,:,:]]
                warped_image = wrap_image_plus(tensor_list)
                #G loss situation
                Writer.add_scalar('stage2/scalar/adversarial_loss2',adver_loss2, image_save_counter)
                Writer.add_scalar('stage2/scalar/mse_loss2',image_mse2, image_save_counter)
                Writer.add_scalar('stage2/scalar/perception_loss2',perception_loss2,image_save_counter)

                Writer.add_scalar('stage2/scalar/D2_loss',d_loss, image_save_counter)

                #save image log
                image_log_name = train_img_log_path+'stage2_'+str(epoch)+'_'+str(i//img_log_step)+'.tif'
                cv2.imwrite(image_log_name,warped_image)
                image_save_counter += 1

            sys.stdout.write("\r[Epoch {}/{}] [Batch {}/{}] [D2 loss: {:.4f}] [G loss: {:.4f}]".format(epoch,NUM_EPOCHS, i, len(train_loader),d_loss.item(), g_loss.item()))
            sys.stdout.flush()
            if i%model_save_step == 0 and i!= 0:
                torch.save(G.state_dict(), checkpoints_path+'G_%d_%d.pth' % (epoch,i // model_save_step))
                torch.save(D2.state_dict(), checkpoints_path+'D2_%d_%d.pth' % (epoch,i // model_save_step))
        lr = lr*0.5
        for param in optimizerD2.param_groups:
            param['lr'] = lr
        for param in optimizerG.param_groups:
            param['lr'] = lr

