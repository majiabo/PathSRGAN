import torch.utils.data as data
import random
import torch
import cv2
import os
import numpy as np
#from lib import data_augment
from lib import data_enhancement as Enhancement

#from tqdm import tqdm
class Fusion20xSingleLayersTripleDataSet(data.Dataset):
    """
        use single layer generate fusion
        example : layers = [-2 , 0 , 2] 
    """
    def __init__(self , img_20x_data_path , img_20x_label_path , name_log_path , layers,sample = None):
        self.image_20x_data_path = img_20x_data_path
        self.image_20x_label_path = img_20x_label_path
        self.name_log_path = name_log_path
        self.layers = layers
        self.image_name_list = []
        self.__read_name_log()
        if sample:
            import random
            random.seed(0)
            random.shuffle(self.image_name_list)
            self.image_name_list = self.image_name_list[:int(len(self.image_name_list)*sample)]
        print('使用数据的数量：',len(self.image_name_list))
    def __read_name_log(self):
        with open(self.name_log_path , 'r') as f:
            for line in f:
                name = line.strip()
                self.image_name_list.append(name)

    def __len__(self):
        return len(self.image_name_list)
    
    def __getitem__(self , id):
        img_name = self.image_name_list[id][0 : self.image_name_list[id].find('.tif')]
        r = random.randint(0 , len(self.layers) - 1)

        img_20x_data_path = os.path.join(self.image_20x_data_path , img_name + '_' + str(self.layers[r]) + '.tif')
        img_20x_label_path = os.path.join(self.image_20x_label_path , img_name + '.tif')
        
        img_20x_label = cv2.imread(img_20x_label_path)
        img_20x_data = cv2.imread(img_20x_data_path)

        # BGR to RGB
        img_20x_label = cv2.cvtColor(img_20x_label,cv2.COLOR_BGR2RGB)
        img_20x_data = cv2.cvtColor(img_20x_data , cv2.COLOR_BGR2RGB)
        # H*W*C to C*H*W
        img_20x_label = np.transpose(img_20x_label, axes= (2,0,1)).astype(np.float32)/255.
        img_20x_data = np.transpose(img_20x_data , axes = (2 , 0 , 1)).astype(np.float32) / 255.
        # numpy array to torch tensor
        img_20x_data = torch.from_numpy(img_20x_data)
        img_20x_label = torch.from_numpy(img_20x_label)

        return [img_20x_data , img_20x_label] #返回4x，10x，20x

class ClassifierDataset(data.Dataset):
    r'''
    Arguments:
        image_path
        name_log_path
        Upasample_flag = False
    out:
        [img_20x,labels]
    note:
        仅用于分类器的数据输入
    '''
    def __init__(self,image_path,name_log_path,
            Downsample_flag= False,pos_flag = True,
            Upsample_flagC = False,Upsample_flagG = False):
        self.image_path = image_path
        self.name_log_path = name_log_path
        self.image_name_list = []
        self.Downsample_flag = Downsample_flag #20倍到10倍
        self.Upsample_flag_C = Upsample_flagC #4倍256分类
        self.Upsample_flag_G = Upsample_flagG #4倍到128生成256
        self.pos_flag = pos_flag
        # default 阳性,10倍不降采样，图256
        self.__read_name_log()
    def __read_name_log(self):
        with open (self.name_log_path) as f:
            for line in f:
                name = line.strip()
                # for positive
                if self.pos_flag is True: 
                    if 'neg' not in name:
                        self.image_name_list.append(name)
                else: 
                    if 'neg' in name:
                        self.image_name_list.append(name)

    def __len__(self):
        return len(self.image_name_list)
    def __getitem__(self,id):

        img_path = os.path.join(self.image_path+self.image_name_list[id])
        img = cv2.imread(img_path)
        if self.Downsample_flag is True:
            img = cv2.resize(img, (256,256))
        if self.Upsample_flag_C is True:
            img = cv2.resize(img, (256,256))
        if self.Upsample_flag_G is True:
            img = cv2.resize(img,(128,128))
        #BGR 2 RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #归一化，HWC to CHW
        img = np.transpose(img, axes= (2,0,1)).astype(np.float32)/255.
        #numpy 2 torch
        img = torch.from_numpy(img)

        if 'neg' in self.image_name_list[id]:
            label = np.zeros(1).astype(np.float32)
        else:
            label = np.ones(1).astype(np.float32)


        return [img,label]

class DoubleDataset(data.Dataset):
    """
    超分辨配对数据集，用来测试VDSR，SRCNN，NatrueGAN...

    """
    def __init__(self,lr_path,hr_path,name_path,resize_ratio = 4,crop_flag = False,ycbcr = False,sample = None):
        """
        lr_path:存放低分辨图像的路径
        hr_path:存放高分辨图像的路径
        name_path:存放需要读取的图像的文件名的txt文件
        resize: scale to fit model
        """
        self.lr_path = lr_path
        self.hr_path = hr_path
        self.crop_flag = crop_flag
        self.name_path = name_path
        self.resize_ratio = resize_ratio
        self.sample = sample
        self.ycbcr = ycbcr

        # init some function
        self._read_names()

    def _read_names(self):
        with open(self.name_path,'r') as f:
            self.image_name_list = [line.strip() for line in f]
        if self.sample:
            random.seed(0)
            random.shuffle(self.image_name_list)
            length = int(len(self.image_name_list) * self.sample)
            self.image_name_list  = self.image_name_list[:length]
    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self,id):
        lr_path = os.path.join(self.lr_path,self.image_name_list[id])
        hr_path = os.path.join(self.hr_path,self.image_name_list[id])
        lr_img = cv2.imread(lr_path).astype(np.float32)/255.
        hr_img = cv2.imread(hr_path).astype(np.float32)/255.
        lr_img = cv2.resize(lr_img,(0,0),fx=self.resize_ratio,fy=self.resize_ratio)
        if self.ycbcr:
            lr_img = cv2.cvtColor(lr_img,cv2.COLOR_BGR2YCrCb)
            hr_img = cv2.cvtColor(hr_img,cv2.COLOR_BGR2YCrCb)
        else:
            lr_img = cv2.cvtColor(lr_img,cv2.COLOR_BGR2RGB)
            hr_img = cv2.cvtColor(hr_img,cv2.COLOR_BGR2RGB)
        if self.crop_flag:
            #hr_img = hr_img[32:96,32:96,:]
            #lr_img = lr_img[32:96,32:96,:]
            hr_img = cv2.resize(hr_img,(64,64))
            lr_img = cv2.resize(lr_img,(64,64))

        lr_img = np.transpose(lr_img,axes=(2,0,1))
        hr_img = np.transpose(hr_img,axes=(2,0,1))
        lr_img = torch.from_numpy(lr_img)
        hr_img = torch.from_numpy(hr_img)
        #for debug
        return [lr_img,hr_img]

class DoubleDatasetHRdown(data.Dataset):
    """
    HR down sample to supervised lr
    """
    def __init__(self,lr_path,hr_path,name_path,sample = False):
        """
        lr_path:存放低分辨图像的路径
        hr_path:存放高分辨图像的路径
        name_path:存放需要读取的图像的文件名的txt文件
        resize: scale to fit model
        """
        self.lr_path = lr_path
        self.hr_path = hr_path
        self.name_path = name_path
        # init some function
        self._read_names()
        if sample:
            import random
            random.seed(0)
            random.shuffle(self.image_name_list)
            self.image_name_list = self.image_name_list[:len(self.image_name_list)//10]
    def _read_names(self):
        with open(self.name_path,'r') as f:
            self.image_name_list = [line.strip() for line in f]
    
    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self,id):
        lr_path = os.path.join(self.lr_path,self.image_name_list[id])
        hr_path = os.path.join(self.hr_path,self.image_name_list[id])
        lr_img = cv2.imread(lr_path)
        hr_img = cv2.imread(hr_path)
        hr_img = cv2.resize(hr_img,(0,0),fx = 0.5,fy = 0.5)
        lr_img = cv2.cvtColor(lr_img,cv2.COLOR_BGR2RGB)
        hr_img = cv2.cvtColor(hr_img,cv2.COLOR_BGR2RGB)

        lr_img = np.transpose(lr_img,axes=(2,0,1)).astype(np.float32)/255.
        hr_img = np.transpose(hr_img,axes=(2,0,1)).astype(np.float32)/255.
        lr_img = torch.from_numpy(lr_img)
        hr_img = torch.from_numpy(hr_img)
        #for debug
        return [lr_img,hr_img]

class TripleDataSetHRdown(data.Dataset):
    """
    适用于4x，10x,20x超分辨，一次性读取对应的三组图
    """
    def __init__(self,image_4x_path,image_20x_path,name_log_path,real_4x = False):
        self.image_4x_path = image_4x_path
        self.image_20x_path = image_20x_path
        self.name_log_path = name_log_path
        self.real_4x = False
        self.image_name_list = []

        # init some utils function
        self.__read_name_log()
    def __read_name_log(self):
        with open(self.name_log_path,'r') as f:
            for line in f:
                name = line.strip()
                self.image_name_list.append(name)

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self,id):
        img_4x_path = os.path.join(self.image_4x_path,self.image_name_list[id])
        img_20x_path = os.path.join(self.image_20x_path,self.image_name_list[id])

        img_4x = cv2.imread(img_4x_path)
        img_20x = cv2.imread(img_20x_path)
        img_10x = cv2.resize(img_20x,(0,0),fx=0.5,fy=0.5)

        # BGR to RGB
        img_4x = cv2.cvtColor(img_4x,cv2.COLOR_BGR2RGB)
        img_10x = cv2.cvtColor(img_10x,cv2.COLOR_BGR2RGB)
        img_20x = cv2.cvtColor(img_20x,cv2.COLOR_BGR2RGB)
        # H*W*C to C*H*W
        img_4x = np.transpose(img_4x, axes = (2,0,1)).astype(np.float32)/255.
        img_10x = np.transpose(img_10x, axes= (2,0,1)).astype(np.float32)/255.
        img_20x = np.transpose(img_20x, axes= (2,0,1)).astype(np.float32)/255.
        # numpy array to torch tensor
        img_4x = torch.from_numpy(img_4x)
        img_10x = torch.from_numpy(img_10x)
        img_20x = torch.from_numpy(img_20x)

        return [img_4x,img_10x,img_20x]

class DatasetWithoutUpsample(data.Dataset):
    def __init__(self,lr_path,hr_path,name_path,sample = None,brightness = None,datatype=None,exclude = None):
        """
        lr_path:存放低分辨图像的路径
        hr_path:存放高分辨图像的路径
        name_path:存放需要读取的图像的文件名的txt文件
        resize: scale to fit model
        """
        self.lr_path = lr_path
        self.hr_path = hr_path
        self.sample = sample
        self.name_path = name_path
        self.brightness = brightness
        self.datatype = datatype
        self.exclude = exclude
        # init some function
        self._read_names()
        if sample:
            random.seed(0)
            random.shuffle(self.image_name_list)
            self.image_name_list = self.image_name_list[:int(len(self.image_name_list)*sample)]
        print('使用数据的数量：',len(self.image_name_list))
    def _read_names(self):
        with open(self.name_path,'r') as f:
            if self.datatype is None:
                self.image_name_list = [line.strip() for line in f]
            else:
                self.image_name_list = [line.strip() for line in f if self.datatype in line]
            if self.exclude:
                new_list = []
                for slide in self.exclude:
                    for name in self.image_name_list:
                        if slide not in name:
                            new_list.append(name)
                self.image_name_list = new_list

            
            

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self,id):
        lr_path = os.path.join(self.lr_path,self.image_name_list[id])
        hr_path = os.path.join(self.hr_path,self.image_name_list[id])
        lr_img = cv2.imread(lr_path)
        hr_img = cv2.imread(hr_path)
        lr_img = cv2.cvtColor(lr_img,cv2.COLOR_BGR2RGB)
        hr_img = cv2.cvtColor(hr_img,cv2.COLOR_BGR2RGB)
        lr_img = np.transpose(lr_img,axes=(2,0,1)).astype(np.float32)/255.
        hr_img = np.transpose(hr_img,axes=(2,0,1)).astype(np.float32)/255.
        lr_img = torch.from_numpy(lr_img)
        hr_img = torch.from_numpy(hr_img)
        #for debug
        return [lr_img,hr_img]

class AugmentDouble(data.Dataset):
    """
    使用数据增强，读取一对图像的dataset
    """
    def __init__(self,lr_path,hr_path,name_path,sample = None,datatype=None,exclude = None,augment_flag=True):
        """
        lr_path:存放低分辨图像的路径
        hr_path:存放高分辨图像的路径
        name_path:存放需要读取的图像的文件名的txt文件
        resize: scale to fit model
        sample:从训练集采集数据的比例
        datatype:读取数据的类型，posi，neg,None都读
        exclude:排除某些片子
        augment:增强的方式
                1. contrast
                2. gamma
                3. noise
                4. hsv 
                5. linear
        """
        self.lr_path = lr_path
        self.hr_path = hr_path
        self.sample = sample
        self.name_path = name_path
        self.datatype = datatype
        self.exclude = exclude
        self.augment_flag = augment_flag
        # init some function
        self._read_names()
        #print(len(self.image_name_list))
        if sample:
            import random
            random.seed(0)
            random.shuffle(self.image_name_list)
            self.image_name_list = self.image_name_list[:int(len(self.image_name_list)*sample)]
        print('使用数据的数量：',len(self.image_name_list))
        if augment_flag:
            print('使用了数据增强')

    def _read_names(self):
        with open(self.name_path,'r') as f:
            if self.datatype is None:
                self.image_name_list = [line.strip() for line in f]
                #print(self.image_name_list)
            else:
                self.image_name_list = [line.strip() for line in f if self.datatype in line]
            if self.exclude is not None:
                new_list = []
                for slide in self.exclude:
                    for name in self.image_name_list:
                        if slide not in name:
                            new_list.append(name)
                self.image_name_list = new_list

    def __len__(self):
        return len(self.image_name_list)
    def __dataenhancement(self,img):
        '''
        增强封装
        '''
        #===初始化随机一个10位二进制数===
        choose_enhance = []
        func = [Enhancement.linear_trans,Enhancement.contrast,
                Enhancement.gamma_trans,Enhancement.HSV_trans,
                Enhancement.Sharp,Enhancement.Gauss,Enhancement.img_noise]
        random.shuffle(func)
        img = func[0](img)
        if random.random() > 0.5:
            img = func[1](img)
        return img
    def __getitem__(self,id):
        lr_path = os.path.join(self.lr_path,self.image_name_list[id])
        hr_path = os.path.join(self.hr_path,self.image_name_list[id])
        lr_img = cv2.imread(lr_path)
        hr_img = cv2.imread(hr_path)
        lr_img = cv2.cvtColor(lr_img,cv2.COLOR_BGR2RGB)
        hr_img = cv2.cvtColor(hr_img,cv2.COLOR_BGR2RGB)
        # for data augment
        if self.augment_flag:
            if random.random() > 0.5:
                lr_img = self.__dataenhancement(lr_img)

        lr_img = np.transpose(lr_img,axes=(2,0,1)).astype(np.float32)/255.
        hr_img = np.transpose(hr_img,axes=(2,0,1)).astype(np.float32)/255.
        lr_img = torch.from_numpy(lr_img)
        hr_img = torch.from_numpy(hr_img)
        #for debug
        return [lr_img,hr_img]

class TaskDataset(data.Dataset):
    r"""
    Arguments:
        image_4x_path
        image_10x_path
        name_log_path
    out:
        [img_4x,img10_x,label]
    note:
        适用于4x，10x，一次性读取对应的两组图
        图片二分类，文件名含有neg为neg，其余为pos
    
    """
    def __init__(self,image_4x_path,image_10x_path,name_log_path):
        self.image_4x_path = image_4x_path
        self.image_10x_path = image_10x_path
        self.name_log_path = name_log_path
        self.image_name_list = []

        # init some utils function
        self.__read_name_log()
        
    def __read_name_log(self):
        with open(self.name_log_path,'r') as f:
            self.image_name_list = [line.strip() for line in f]

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self,id):
        img_4x_path = os.path.join(self.image_4x_path,self.image_name_list[id])
        img_10x_path = os.path.join(self.image_10x_path,self.image_name_list[id])

        img_4x = cv2.imread(img_4x_path)
        img_10x = cv2.imread(img_10x_path)

        # BGR to RGB
        img_4x = cv2.cvtColor(img_4x,cv2.COLOR_BGR2RGB)
        img_10x = cv2.cvtColor(img_10x,cv2.COLOR_BGR2RGB)
        # H*W*C to C*H*W 
        img_4x = np.transpose(img_4x, axes = (2,0,1)).astype(np.float32)/255.
        img_10x = np.transpose(img_10x, axes= (2,0,1)).astype(np.float32)/255.
        # numpy array to torch tensor
        img_4x = torch.from_numpy(img_4x)
        img_10x = torch.from_numpy(img_10x)
        if 'neg' in self.image_name_list[id]:
            label = np.zeros(1).astype(np.float32)
        else:
            label = np.ones(1).astype(np.float32)
        return [img_4x,img_10x,label]

class WeightDataSet(data.Dataset):
    """
    适用于4x，10x,20x超分辨，一次性读取对应的三组图
    ps:新加入weight map，所以，一次读四组图。
    """
    def __init__(self,image_4x_path,image_10x_path,image_20x_path,weight_map_path,name_log_path):
        self.image_4x_path = image_4x_path
        self.image_10x_path = image_10x_path
        self.image_20x_path = image_20x_path
        self.weight_map_path = weight_map_path
        self.name_log_path = name_log_path
        self.image_name_list = []

        # init some utils function
        self.__read_name_log()
    def __read_name_log(self):
        with open(self.name_log_path,'r') as f:
            for line in f:
                name = line.strip()
                self.image_name_list.append(name)

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self,id):
        img_4x_path = os.path.join(self.image_4x_path,self.image_name_list[id])
        img_10x_path = os.path.join(self.image_10x_path,self.image_name_list[id])
        img_20x_path = os.path.join(self.image_20x_path,self.image_name_list[id])
        weight_map_path = os.path.join(self.weight_map_path,self.image_name_list[id])

        img_4x = cv2.imread(img_4x_path)
        img_10x = cv2.imread(img_10x_path)
        img_20x = cv2.imread(img_20x_path)
        weight_map_20 = cv2.imread(weight_map_path,cv2.IMREAD_GRAYSCALE)/255.
        weight_map_10 = cv2.resize(weight_map_20,tuple(img_10x.shape[0:2]))
        weight_map_20 = weight_map_20.astype(np.float32)
        weight_map_10 = weight_map_10.astype(np.float32)
        weight_map_20 = np.expand_dims(weight_map_20,0)
        weight_map_10 = np.expand_dims(weight_map_10,0)
        #weight_map = np.expand_dims(weight_map,0)
        # BGR to RGB
        img_4x = cv2.cvtColor(img_4x,cv2.COLOR_BGR2RGB)
        img_10x = cv2.cvtColor(img_10x,cv2.COLOR_BGR2RGB)
        img_20x = cv2.cvtColor(img_20x,cv2.COLOR_BGR2RGB)
        # H*W*C to C*H*W
        img_4x = np.transpose(img_4x, axes = (2,0,1)).astype(np.float32)/255.
        img_10x = np.transpose(img_10x, axes= (2,0,1)).astype(np.float32)/255.
        img_20x = np.transpose(img_20x, axes= (2,0,1)).astype(np.float32)/255.
        # numpy array to torch tensor
        img_4x = torch.from_numpy(img_4x)
        img_10x = torch.from_numpy(img_10x)
        img_20x = torch.from_numpy(img_20x)
        weight_map_20 =torch.from_numpy(weight_map_20)
        weight_map_10 = torch.from_numpy(weight_map_10)
        return [img_4x,img_10x,img_20x,weight_map_10,weight_map_20]

class TripleDataSet(data.Dataset):
    """
    适用于4x，10x,20x超分辨，一次性读取对应的三组图
    """
    def __init__(self,image_4x_path,image_10x_path,image_20x_path,name_log_path,crop = False,sample=None, brightness=None):
        self.image_4x_path = image_4x_path
        self.image_10x_path = image_10x_path
        self.image_20x_path = image_20x_path
        self.name_log_path = name_log_path
        self.crop = crop
        self.brightness = brightness
        self.image_name_list = []

        # init some utils function
        self.__read_name_log()
        if sample:
            random.seed(0)
            random.shuffle(self.image_name_list)
            self.image_name_list = self.image_name_list[:int(len(self.image_name_list)*sample)]
    def __read_name_log(self):
        with open(self.name_log_path,'r') as f:
            for line in f:
                name = line.strip()
                self.image_name_list.append(name)

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self,id):
        img_4x_path = os.path.join(self.image_4x_path,self.image_name_list[id])
        img_10x_path = os.path.join(self.image_10x_path,self.image_name_list[id])
        img_20x_path = os.path.join(self.image_20x_path,self.image_name_list[id])

        img_4x = cv2.imread(img_4x_path)
        img_10x = cv2.imread(img_10x_path)
        img_20x = cv2.imread(img_20x_path)

        # BGR to RGB
        img_4x = cv2.cvtColor(img_4x,cv2.COLOR_BGR2RGB)
        img_10x = cv2.cvtColor(img_10x,cv2.COLOR_BGR2RGB)
        img_20x = cv2.cvtColor(img_20x,cv2.COLOR_BGR2RGB)
        if self.crop:
            #because input 4x is 256
            img_4x = img_4x[64:192,64:192,:]
            img_10x = img_10x[128:384,128:384,:]
            img_20x = img_20x[256:768,256:768,:]
        if self.brightness is not None:
            img_4x = cv2.cvtColor(img_4x, cv2.COLOR_RGB2HSV)
            img_10x = cv2.cvtColor(img_10x, cv2.COLOR_RGB2HSV)
            img_20x = cv2.cvtColor(img_20x, cv2.COLOR_RGB2HSV)

            img_4x[..., 2] = img_4x[..., 2] + self.brightness
            img_10x[..., 2] = img_10x[..., 2] + self.brightness
            img_20x[..., 2] = img_20x[..., 2] + self.brightness

            img_4x = cv2.cvtColor(img_4x, cv2.COLOR_HSV2RGB)
            img_10x = cv2.cvtColor(img_10x, cv2.COLOR_HSV2RGB)
            img_20x = cv2.cvtColor(img_20x, cv2.COLOR_HSV2RGB)
        #     print('4x',img_4x.shape)
        #     print('10x',img_10x.shape)
        #     print('20x',img_20x.shape)
        # # H*W*C to C*H*W
        img_4x = np.transpose(img_4x, axes = (2,0,1)).astype(np.float32)/255.
        img_10x = np.transpose(img_10x, axes= (2,0,1)).astype(np.float32)/255.
        img_20x = np.transpose(img_20x, axes= (2,0,1)).astype(np.float32)/255.
        # numpy array to torch tensor

        img_4x = torch.from_numpy(img_4x)
        img_10x = torch.from_numpy(img_10x)
        img_20x = torch.from_numpy(img_20x)

        return img_4x,img_10x,img_20x

class TestDataSet(data.Dataset):
    """
    适用于4x，10x,20x超分辨，一次性读取对应的三组图
    """
    def __init__(self,image_4x_path,image_10x_path,image_20x_path,name_log_path):
        self.image_4x_path = image_4x_path
        self.image_10x_path = image_10x_path
        self.image_20x_path = image_20x_path
        self.name_log_path = name_log_path
        self.image_name_list = []

        # init some utils function
        self.__read_name_log()
    def __read_name_log(self):
        with open(self.name_log_path,'r') as f:
            for line in f:
                name = line.strip()
                self.image_name_list.append(name)

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self,id):
        img_4x_path = os.path.join(self.image_4x_path,self.image_name_list[id])
        img_10x_path = os.path.join(self.image_10x_path,self.image_name_list[id])
        img_20x_path = os.path.join(self.image_20x_path,self.image_name_list[id])
 
        img_4x = cv2.imread(img_4x_path)
        img_10x = cv2.imread(img_10x_path)
        img_20x = cv2.imread(img_20x_path)

        # BGR to RGB
        img_4x = cv2.cvtColor(img_4x,cv2.COLOR_BGR2RGB)
        img_10x = cv2.cvtColor(img_10x,cv2.COLOR_BGR2RGB)
        img_20x = cv2.cvtColor(img_20x,cv2.COLOR_BGR2RGB)
        # H*W*C to C*H*W
        img_4x = np.transpose(img_4x, axes = (2,0,1)).astype(np.float32)/255.
        img_10x = np.transpose(img_10x, axes= (2,0,1)).astype(np.float32)/255.
        img_20x = np.transpose(img_20x, axes= (2,0,1)).astype(np.float32)/255.
        # numpy array to torch tensor
        img_4x = torch.from_numpy(img_4x)
        img_10x = torch.from_numpy(img_10x)
        img_20x = torch.from_numpy(img_20x)

        return [img_4x,img_10x,img_20x]

def tensor2image (tensor):
    '''
    tensor_list: list，内在元素是 pytorch tensor [c,h,w]
    变化成 image 0-255 hwc BGR

    '''

    imgx = tensor.cpu().detach().numpy()        
    imgx = np.transpose(imgx, axes = (1,2,0))
    imgx = cv2.cvtColor(imgx,cv2.COLOR_RGB2BGR)
    imgx = np.uint(imgx*255)
    return imgx
    
def wrap_image_plus(tensor_list,ycbcr2bgr = False):
    """
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
        if ycbcr2bgr:
            img_x = cv2.cvtColor(img_x,cv2.COLOR_YCrCb2BGR)
        else:
            img_x = cv2.cvtColor(img_x,cv2.COLOR_RGB2BGR)
        img_list.append(img_x)
    h,w,_ = img_list[-1].shape
    for i,img in enumerate(img_list[:-1]):
        img = cv2.resize(img,(h,w))
        img_list[i] = img
    assemble_img = np.concatenate(img_list,axis = 1)
    assemble_img = np.uint8(assemble_img*255) #注意
    return assemble_img

def wrap_multi_channel_img(tensor_list):
    """
    tensor_list: list，内在元素是 pytorch tensor [c,h,w]
    其中，若c是多通道图，则排列之
    """
    img_list = []
    for tensor in tensor_list:
        img = tensor.cpu().detach().numpy()
        img = np.transpose(img,axes = (1,2,0))
        if img.shape[-1] >3:
            sub_list = []
            num = img.shape[-1]//3
            for i in range(num):
                tmp = img[:,:,3*i:3*(i+1)]
                sub_list.append(tmp)
            joint_img = np.concatenate(sub_list,axis = 1)
            img_list.append(joint_img)
        else:
            img_list.append(img)
    assemble_img = np.concatenate(img_list,axis = 0)
    assemble_img = np.uint8(assemble_img*255)
    assemble_img = cv2.cvtColor(assemble_img,cv2.COLOR_RGB2BGR)
    return assemble_img

def get_images_name(path,name_log_path,image_type = 'tif',split = True):
    """
    get image name from a dir 
    split: Ture,将其拆分为训练集和验证
    """
    import random
    from glob import glob
    path_name = os.path.join(path,'*'+image_type)
    name_list = glob(path_name)
    random.shuffle(name_list)
    if not split:
        with open(name_log_path,'w') as f:
            for name in name_list:
                name = os.path.split(name)[1]
                f.write(name)
                f.write('\n')
    else:
        path_head,file_name = os.path.split(name_log_path)
        train_path = os.path.join(path_head,'train.txt')
        test_path = os.path.join(path_head,'test.txt')
        test_list = []
        train_list = []
        for index,name in enumerate(name_list):
            if index<700:
                test_list.append(name)
            else:
                train_list.append(name)
        with open(test_path,'w') as f:
            for name in test_list:
                name = os.path.split(name)[1]
                f.write(name)
                f.write('\n')
        with open(train_path,'w') as f:
            for name in train_list:
                name = os.path.split(name)[1]
                f.write(name)
                f.write('\n')

def cal_acc(predict,gt):

    """

    predict:预测的lables

    gt：真值

    """
    predict = predict.cpu().numpy()
    gt = gt.cpu().numpy()
    g = predict.shape[0]
    right_labels = np.sum(np.abs(predict-gt)<0.5)
    acc = right_labels/g
    return acc

if __name__ == "__main__":
    img_4x_path = '/mnt/diskarray/srdata/new/4x/'
    img_10x_path = '/mnt/diskarray/srdata/new/10x/'
    img_20x_path = '/mnt/diskarray/srdata/new/20x/'
    train_names_log = '/mnt/diskarray/srdata/new/train.txt'
    test_names_log = '/mnt/diskarray/srdata/new/test.txt'

    dataset = TripleDataSet(img_4x_path, img_10x_path, img_20x_path, train_names_log, bightness = None )