"""
author:majiabo
time:2019.1.18
f:utils for sr

"""
import cv2
from skimage import measure
import os
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from lib.dataset import wrap_image_plus
def model_eval_our(G,test_set,batch_size,nums_to_test,img_save_path,times):
    print('Eval model...')
    root_path = os.path.join(img_save_path, str(times))
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    loader = DataLoader(test_set,batch_size=batch_size,shuffle=True,num_workers=4)
    ssims = list()
    psnrs = list()
    img_save_counter = 0
    for index,(lr_img, mr_img, hr_img) in enumerate(loader):
        lr_img = lr_img.cuda()
        hr_img = hr_img.cuda()
        _, gen = G(lr_img)
        for i in range(gen.shape[0]):
            tensor_list = [lr_img[i,:,:,:],gen[i,:,:,:],hr_img[i,:,:,:]]
            wraped = wrap_image_plus(tensor_list)
            h = wraped.shape[0]
            gen_ = wraped[:,h:2*h,:]
            hr = wraped[:,2*h:,:]
            ssim = measure.compare_ssim(gen_,hr,multichannel=True)
            psnr = measure.compare_psnr(hr,gen_)
            ssims.append(ssim)
            psnrs.append(psnr)
            img_path = os.path.join(img_save_path,str(times), f'{img_save_counter}.png')
            cv2.imwrite(img_path,wraped)
            img_save_counter += 1
        if index >= nums_to_test:
            break
    avg_ssim = sum(ssims)/len(ssims)
    avg_psnr = sum(psnrs)/len(psnrs)
    return [avg_ssim,avg_psnr]
def load_multi_model(model,path):
    """
    add support for single gpu model
    :param model: loaded model
    :param path: weights path
    :return: a model loaded weights
    """
    multi_weights = torch.load(path)
    temp_state = OrderedDict()
    single_flag = True
    for k,v in multi_weights.items():
        if 'module' == k[:6]:
            single_flag = False
        new_k = k[7:]
        temp_state[new_k] = v
    # load model
    if single_flag:
        model.load_state_dict(multi_weights)
    else:
        model.load_state_dict(temp_state)
    return model

def need_grad(status,model,fix_name = 'stage1'):
    '''
    锁定模型部分权值不更新
    '''
    for name,value in model.named_parameters():
        if fix_name in name:
            value.requires_grad = status
            if status:
                print(name,':has been released...')
            else:
                print(name,':has been fixed...')
    model_params = filter(lambda p: p.requires_grad,model.parameters() )
    return model_params

def wrap_image(tensor_list):
    """
    将tensor拼成横着的图片
    tensor_list: list，内在元素是 pytorch tensor [c,h,w]
    for convience
    4x,gen10x,10x,gen20x,20x

    """
    # 讲tensor转为numpy 并转换通道
    assert len(tensor_list[0].shape) == 3 ,"tensor 必须 为 CHW"
    img_list = []
    for tensor in tensor_list:
        img_x = tensor.cpu().detach().numpy()
        img_x = np.transpose(img_x,axes=(1,2,0))
        img_x = cv2.cvtColor(img_x,cv2.COLOR_RGB2BGR)
        img_list.append(img_x)
    h,w,_ = img_list[-1].shape
    for i,img in enumerate(img_list[:-1]):
        img = cv2.resize(img,(h,w))
        img_list[i] = img
    assemble_img = np.concatenate(img_list,axis = 1)
    assemble_img = np.uint8(assemble_img*255)
    return assemble_img

def record_argument(file_name,**kargs):
    import time
    localtime = time.asctime()
    with open(file_name,'w') as f:
        f.write('Record time:')
        f.write(localtime)
        f.write('\n\n')
        for key,value in kargs.items():
            content = key+':'+str(value)+'\n'
            f.write(content)
        print('Argument recorded..')

def check_regis_acc(lr,hr,resize_ratio = 5,ratio = 2,max_shift = 10):
    """
    计算已近配准后的的图像的配准精度
    resize_ratio: 图像放大倍率
    max_shift:两幅图像允许最大的偏移量，超出后无法给出结果
    ratio：高倍图像和低倍图像的倍率关系

    PS:低倍图像会被首先插值到高倍图像，然后
    """
    #将图像放大到需要的倍数
    hr= cv2.resize(hr,(0,0),fx=resize_ratio,fy=resize_ratio)
    lr = cv2.resize(lr,(0,0),fx=resize_ratio,fy=resize_ratio)
    #将偏移同步
    max_shift = int(resize_ratio*max_shift)
    result = image_shift(lr,hr,max_shift=max_shift,ratio=ratio)
    return result
    
def image_shift(lr,hr,max_shift=10,ratio =2):
    """
    lr:低分辨率图像
    hr:高分辨率图像
    计算图像内容一致的两幅图像的相对偏移量，总是以hr为基准
    
    适用于偏移在10像素以内。
    保证输入图像是8的倍数，减小误差
    只能处理像素级别误差
    """
    h,w = lr.shape[0:2]
    sh = int(h/4)
    sw = int(w/4)
    bh = int(h/2)
    bw = int(w/2)

    x_ = [0.125,0.625,0.375,0.125,0.625]
    y_ = [0.125,0.125,0.375,0.625,0.625]
    x_list = [int(w*x) for x in x_]
    y_list = [int(h*y) for y in y_]
    h_shift = []
    w_shift = []
    for x,y in zip(x_list,y_list):
        lr_candidate = cv2.resize(lr[y:y+sh,x:x+sw],(bh,bw))
        bx,by = int(x*2),int(y*2)
        hr_candidate = hr[by-max_shift:by+bh+max_shift,bx-max_shift:bx+bw+max_shift]
        result = cv2.matchTemplate(hr_candidate,lr_candidate,method=cv2.TM_CCOEFF_NORMED)
        _, maxVal,_, maxLoc = cv2.minMaxLoc(result)
        #negative 为向左，上偏移
        h_shift.append(maxLoc[0]-max_shift)
        w_shift.append(maxLoc[1]-max_shift)
    h_mean = sum(h_shift)/len(h_shift)
    w_mean = sum(w_shift)/len(w_shift)
    return [h_mean,w_mean,h_shift,w_shift]

def cal_avg_ssim(name_log,lr_path,hr_path):
    """
    针对当前数据组织格式
    name_log: 存放图像名字的文件
    root_path1,root_path2: 要比对的图像的目录
    计算当前数据的平均相似度
    reutrn :返回两个值，第一个为当前数据的平均ssim
            第二个为单个图像的ssim，与name_log记录的顺序一致
    """
    name_list = read_namelog(name_log)
    ssim_list = []
    for name in name_list:
        hr_img_path = os.path.join(hr_path,name)
        lr_img_path = os.path.join(lr_path,name)

        hr = cv2.imread(hr_img_path)
        lr = cv2.imread(lr_img_path)
        #考虑到通道不影响计算相似度，故省略之
        shape = tuple(hr.shape[0:2])
        lr = cv2.resize(lr,shape)
        ssim = measure.compare_ssim(lr,hr,multichannel=True)
        ssim_list.append(ssim)
    ssim_mean = sum(ssim_list)/len(ssim_list)
    return [ssim_mean,ssim_list]

def split_img(img):
    """
    将生成的长条状的log图像分开，按序返回
    """
    h,w = img.shape[0:2]
    if w/h == 3:
        #三个图拼成一个
        lr = img[:,:h,:]
        gen = img[:,h:h*2,:]
        hr = img[:,h*2:,:]
        return [lr,gen,hr]
    elif w/h == 5:
        #五个图拼成一个
        lr = img[:,:h,:]
        gen_lr = img[:,h:h*2,:]
        mr = img[:,h*2:h*3,:]
        gen_hr = img[:,h*3:h*4,:]
        hr = img[:,h*4:,:]
        return[lr,gen_lr,mr,gen_hr,hr]
    else:
        # Not support yet
        return None

def read_namelog(name):
    """
    读取namefile 文件，解析文件名返回列表
    """
    name_list = []
    with open(name,'r') as f:
        for line in f:
            line = line.strip()
            name_list.append(line)
    return name_list


