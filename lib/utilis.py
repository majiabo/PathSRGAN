import cv2
from skimage.measure import compare_psnr, compare_ssim
from skimage import morphology
import os
import openslide
import numpy as np
import time
import csv




#2019.3.8 版本
#相对稳定版
#从lr配hr与从hr配lr均已测试完成

#增加SURF法粗配（测试版）




"""
请注意！！！！
将图片看成数组时，H在前，W在后
如：
sample_10x_result = sample_10x[maxloc[1]:maxloc[1]+256,maxloc[0]:maxloc[0]+256,:]
maxloc为模板匹配后得到的坐标，W在前H在后
而利用该坐标进行剪切时，看做array的图像三个维度分别为H，W，C
"""

def get_coors_from_csv(csv_path,slide_id,target_20x = 768,CLASS = ['LSIL','HSIL','ASCUS']):
    """
    lisubo所写，majiabo改写
    从csv文件中获取坐标信息
    csv_path:存放csv所有子文件的目录
    slide_id:slide的片号
    """  
    fileCsv1 = csv_path + slide_id + '/file1.csv'
    fileCsv2 = csv_path + slide_id + '/file2.csv'
    csv1 = open(fileCsv1, "r").readlines()
    csv2 = open(fileCsv2, "r").readlines()
    if csv1==[]:
        print('Empty csv file, skip.')
        return None
    # 获取所需类别的序号
    info = {}
    listIndex = []

    for c in CLASS:
            listIndex += [csv2.index(line) for line in csv2 if c in line]
            info[c] = []
    listIndex = list(set(listIndex))  # 序号列表   
    for i in listIndex:
        line = csv2[i]
        elems = line.strip().split(',')

        label = elems[0]  # 标签
        shape = elems[1].strip().split(' ')[0]  # 标注形状

        # 获取外接矩形右上点坐标和偏移量
        index_Y = int(float(elems[2]))
        index_X = int(float(elems[3]))
        dy = int(float(elems[4]))
        dx = int(float(elems[5]))
        center_y = index_Y + dy//2
        center_x = index_X + dx//2
        left_y = center_y - target_20x//2
        left_x = center_x - target_20x//2
        info[label].append([slide_id,left_y,left_x,target_20x,target_20x])
    return  info


def need_grad(status,model,fix_name):
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

def get_sample_numbers(filename):
    """
    从存放样本的txt文件中获取样本数量
    """
    with open(filename) as f:
        lines = [line.strip() for line in f]
        num = len(lines)
    return num

def get_slide_name(file):
    """
    读取slide的file_name以及label
    slide_name格式:
    neg_1149300.svs or posi_1149300.svs
    """
    label=[]
    slide_name_result=[]
    with open(file,'r') as f:
        slide_name = f.readlines()
        for i in slide_name:
            label.append(i.split('_')[0])
            i=i.split('_')[1]
            slide_name_result.append(i.rstrip('\n'))
        return slide_name_result, label
    
def path_checker(path):
    """
    检查目录是否存在，不存在，则创建
    """
    if not os.path.isdir(path):
        os.makedirs(path)
        print('目录不存在，已创建...')
    else:
        print('目录已存在')

def evaluate(lr,hr):
    """
    计算SSIM和PSNR
    return：[ssim,psnr]
    """
    ssim = compare_ssim(lr,hr,multichannel = True)
    psnr = compare_psnr(lr,hr)
    return ssim,psnr

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

def regis_coarse(low_path, high_path, slide_ratio, max_shift = 4000, points = 2):
    """
    粗配函数
    path: 
        图片路径
    slide_ratio：输入一个list存放原图倍数[low_slide, high_slide]
    max_shift: 模板匹配冗余量，尺度为大图上的尺度
    points：匹配时采样的点数
    返回：平均坐标偏移平均值及各样本值
    """
    
    i=0
    match_flag = True
    ratio = slide_ratio[1]/slide_ratio[0]
    dW = []
    dH = []
    lsize = 2000

    lr = openslide.OpenSlide(low_path)
    hr = openslide.OpenSlide(high_path)

    w,h = lr.dimensions
    safelength =int( w//3)
    ran_xs = np.random.randint(safelength,high=w - safelength,size = 20)
    ran_ys = np.random.randint(safelength,high=h - safelength,size = 20)
    for lr_x,lr_y in zip(ran_xs,ran_ys):
        i+=1
        print('No. {}'.format(i))
        lr_sample = np.array(lr.read_region((lr_x,lr_y),level=0,size=(lsize,lsize)).convert('RGB'))
        lr_sample = cv2.resize(lr_sample,(0,0),fx=ratio,fy=ratio)
        #将低倍图坐标映射到高倍图
        hr_x = int(lr_x*ratio-max_shift)
        hr_y = int(lr_y*ratio-max_shift)
        hsize = int(2000*ratio+max_shift*2)
        
        try:
            hr_sample = np.array(hr.read_region((hr_x,hr_y),level = 0,size=(hsize,hsize)).convert('RGB'))
        except:
            return None, None, None, None, False

        result = cv2.matchTemplate(hr_sample,lr_sample,cv2.TM_CCOEFF_NORMED)
        _, val, _, maxloc = cv2.minMaxLoc(result)
        if val <0.6:
            max_shift += 1000
            print('max_shift:',max_shift)
        else:
            dW.append(maxloc[0] - max_shift)
            dH.append(maxloc[1] - max_shift)
        if max_shift>9000:
            match_flag = False
            print('match failed...')
            break
        if len(dW)>=points:
            break
        print('maxVal:',val)
    try:
        ex = int(sum(dW)/len(dW))
        ey = int(sum(dH)/len(dH))
    except ZeroDivisionError:
        return None, None, None, None, False
    
    
    return ex, ey, dW, dH, match_flag


def write_regis_result(slide_name, label, slide_ratio, ex, ey, dW, dH, output_file = './SVS/coarse_regis_log.txt', save_mode = 'a'):
    """
    用来写入粗配结果
    slide_name: slide文件名
    label：slide label
    参数含义与粗配输入、输出相同
    """
    with open(output_file,save_mode) as f:
        f.write('---------------------\n')
        f.write('label:'+label+'\n')
        f.write('slide_name:'+slide_name+'\n')
        f.write('ratio:'+str(slide_ratio[0])+'\t'+str(slide_ratio[1])+'\n')
        f.write('ex:'+str(ex)+'\n')
        f.write('ey:'+str(ey)+'\n')
        f.write('x_value:'+str(dW)+'\n')
        f.write('y_value:'+str(dH)+'\n')

def get_regis_result(slide_name, slide_ratio, file='./SVS/coarse_regis_log.txt'):
    """
    从文件中读取粗配得到的偏移坐标
    file: 读取的log文件路径
    slide_name: 片子的文件名
    slide_ratio：输入一个list存放原图倍数[low_slide, high_slide]
    """
    with open(file,'r') as f:
        m = f.readlines()
        for i in range(len(m)):
            if (str(slide_name) in m[i]) and (str(slide_ratio[0]) in m[i+1]) and (str(slide_ratio[1]) in m[i+1]):
                ex = int(m[i+2].lstrip('ex:').rstrip('\n'))
                ey = int(m[i+3].lstrip('ey:').rstrip('\n'))
                return ex, ey
        return None, None


def read_coarse_regis_infos(info_path, mode):
    """
    info_path:粗配结果文件
    mode:映射模式{
        2.5: 4x to 10x
        2: 10x to 20x
        5: 4x to 20x
    }
    return:
    [[label,...,],[slide_id,...,],[ex,...,],[ey,...,]]
    注意slide_id返回为str形式
    """
    label_pool=[]
    slide_id_pool=[]
    ex_pool=[]
    ey_pool=[]
    label=''
    slide_id=''
    ex=0
    ey=0
    with open(info_path,'r') as f:
        m = f.readlines()
        for i in range(len(m)):
            flag=False
            if (mode==2.5) and ('4\t10' in m[i]):
                flag=True
                label = m[i-2].lstrip('label:').strip()
                slide_id = m[i-1].lstrip('slide_name:').rstrip('.svs\n')
                ex = int(m[i+1].lstrip('ex:').rstrip('\n'))
                ey = int(m[i+2].lstrip('ey:').rstrip('\n'))
            elif (mode==2) and ('10\t20' in m[i]):
                flag=True
                label = m[i-2].lstrip('label:').strip()
                slide_id = m[i-1].lstrip('slide_name:').rstrip('.svs\n')
                ex = int(m[i+1].lstrip('ex:').rstrip('\n'))
                ey = int(m[i+2].lstrip('ey:').rstrip('\n'))
            elif (mode==5) and ('4\t20' in m[i]):
                flag=True
                label = m[i-2].lstrip('label:').strip()
                slide_id = m[i-1].lstrip('slide_name:').rstrip('.svs\n')
                ex = int(m[i+1].lstrip('ex:').rstrip('\n'))
                ey = int(m[i+2].lstrip('ey:').rstrip('\n'))
            if flag:
                label_pool.append(label)
                slide_id_pool.append(slide_id)
                ex_pool.append(ex)
                ey_pool.append(ey)
        if len(label_pool)==0:
            return [None,None,None,None]
        return [label_pool,slide_id_pool,ex_pool,ey_pool]
    
    
    
##############    
def generate_4to20 (file):
    """
    """
    a=read_coarse_regis_infos(file ,2.5)
    b=read_coarse_regis_infos(file ,2)
    c=[[],[],[],[]]
    if a[1]==b[1]:
        c[0]=a[0]
        c[1]=a[1]
        for i in range(len(a[0])):
            c[2].append(2*a[2][i]+b[2][i])
            c[3].append(2*a[3][i]+b[3][i])
        
        with open(file ,'a') as f:
            for i in range(len(a[0])):
                f.write('---------------------\n')
                f.write('label:'+c[0][i]+'\n')
                f.write('slide_name:'+c[1][i]+'.svs\n')
                f.write('ratio:4\t20\n')
                f.write('ex:'+str(c[2][i])+'\n')
                f.write('ey:'+str(c[3][i])+'\n')
    else:
        print('generate error')


def _BinarySP(img, threColor=35, threVol=2000, Blocksize=101, C=10):
        '''
        执行图像分割，分割背景与前景
        '''
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        th1 = cv2.adaptiveThreshold(gray[50:462,50:462],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,Blocksize,C)
        a = th1
        a = a<=0 #the foreground is 1,and background is 0,so inverse the color
        dst=morphology.remove_small_objects(a, min_size=threVol, connectivity=1)
        imgBin = dst > 0
        s = imgBin.sum()
        if s >1:
            flag = True
        else:
            flag = False
        return [flag,imgBin]


def regis_careful_version1(slide_name, path_4x, path_10x, path_20x, shift, pic_num=1000, redun_20x = 5000, label='neg'):
    """
    细配函数version_1
    从20x图中随机取有含有细胞的点，降采样后分别在10x、4x图上滑动给出结果
    输出三张图片分别为128*128,256*256,512*512
    slide_name: 图片文件名
    path: 图片路径
    pic_num：取图的数量
    shift：给出一个列表表示粗配结果[4to20shift_x, 4to20shift_y, 10to20shift_x, 10to20shift_y]
    redun_20x: 在随机取点时作用在20x上的冗余，防止取到白边
    label: 表明当前片子的label用于命名
    """
    
    i = 0
    sample_size_4x = 200
    sample_size_10x = 400
    size_20x = 512
    img_4x = openslide.OpenSlide(path_4x)
    img_10x = openslide.OpenSlide(path_10x)
    img_20x = openslide.OpenSlide(path_20x)
    
    w,h = img_20x.dimensions
    ran_xs = np.random.randint(redun_20x,high=w-size_20x-redun_20x,size = pic_num)
    ran_ys = np.random.randint(redun_20x,high=h-size_20x-redun_20x,size = pic_num)
    while i != pic_num:
        x_20x=ran_xs[i]
        y_20x=ran_ys[i]
        sample_20x = np.array(img_20x.read_region((x_20x,y_20x),level=0,size=(size_20x,size_20x)).convert('RGB'))
        sample_20x = cv2.cvtColor(sample_20x,cv2.COLOR_BGR2RGB)
        [cell_flag,_] = _BinarySP(sample_20x)
        
        if not(cell_flag):
            ran_xs = np.delete(ran_xs,i)
            ran_ys = np.delete(ran_ys,i)
            ran_xs = np.hstack((ran_xs,np.random.randint(redun_20x,high=w-size_20x-redun_20x,size = 1)))
            ran_ys = np.hstack((ran_ys,np.random.randint(redun_20x,high=h-size_20x-redun_20x,size = 1)))
            continue
        else:
            
            #在4x上细配
            sample_20x_102 = cv2.resize(sample_20x,(102,102))
            safe_belt_4x = (sample_size_4x-102)//2
            x_4x = int((x_20x-shift[0])//5-safe_belt_4x)
            y_4x = int((y_20x-shift[1])//5-safe_belt_4x)
            sample_4x = np.array(img_4x.read_region((x_4x,y_4x),level = 0,size=(sample_size_4x,sample_size_4x)).convert('RGB'))
            sample_4x = cv2.cvtColor(sample_4x,cv2.COLOR_BGR2RGB)
            
            result = cv2.matchTemplate(sample_4x,sample_20x_102,cv2.TM_CCOEFF)
            _, maxval, _, maxloc = cv2.minMaxLoc(result)

            sample_4x_result = sample_4x[maxloc[1]:maxloc[1]+102,maxloc[0]:maxloc[0]+102,:]
            sample_4x_result = cv2.resize(sample_4x_result,(128,128))
            '''sample_4x_result = np.hstack((sample_4x_result, sample_20x_102))'''#测试用拼图
            
            #测试用
            '''if not(os.path.exists('./4x_1')):
                os.mkdir('./4x_1')
            cv2.rectangle(sample_4x, maxloc, (maxloc[0]+102,maxloc[1]+102), [0, 0, 0])
            cv2.imwrite('./4x_1/{}-{}.tif'.format(int(slide_id),i),sample_4x)'''
            #
            
            #在10x上细配
            sample_20x_256 = cv2.resize(sample_20x,(256,256))
            safe_belt_10x = (sample_size_10x-256)//2
            x_10x = int((x_20x-shift[2])//2-safe_belt_10x)
            y_10x = int((y_20x-shift[3])//2-safe_belt_10x)
            sample_10x = np.array(img_10x.read_region((x_10x,y_10x),level = 0,size=(sample_size_10x,sample_size_10x)).convert('RGB'))
            sample_10x = cv2.cvtColor(sample_10x,cv2.COLOR_BGR2RGB)
            
            result = cv2.matchTemplate(sample_10x,sample_20x_256,cv2.TM_CCOEFF)
            _, maxval, _, maxloc = cv2.minMaxLoc(result)

            sample_10x_result = sample_10x[maxloc[1]:maxloc[1]+256,maxloc[0]:maxloc[0]+256,:]
            '''sample_10x_result = np.hstack((sample_10x_result, sample_20x_256))'''#测试用拼图
            
            #测试用
            '''if not(os.path.exists('./10x_1')):
                os.mkdir('./10x_1')
            cv2.rectangle(sample_10x, maxloc, (maxloc[0]+256,maxloc[1]+256), [0, 0, 0])
            cv2.imwrite('./10x_1/{}-{}.tif'.format(int(slide_id),i),sample_10x)
            
            cv2.imwrite('./result/{}-{}.tif'.format(int(slide_id),i),pingtu(sample_4x_result,sample_4x,sample_10x_result,sample_10x,sample_20x))'''
            #
            
            #输出部分
            if not(os.path.exists('./regis_tif/'+label)):
                os.mkdir('./regis_tif/'+label)
            path = os.path.join('./regis_tif/',label)
            if not(os.path.exists(path+'/4x')):
                os.mkdir(path+'/4x')
            cv2.imwrite(path+'/4x/{}_{}.tif'.format(slide_name.rstrip('.svs.'),i),sample_4x_result)
            if not(os.path.exists(path+'/10x')):
                os.mkdir(path+'/10x')
            cv2.imwrite(path+'/10x/{}_{}.tif'.format(slide_name.rstrip('.svs.'),i),sample_10x_result)
            if not(os.path.exists(path+'/20x')):
                os.mkdir(path+'/20x')
            cv2.imwrite(path+'/20x/{}_{}.tif'.format(slide_name.rstrip('.svs.'),i),sample_20x)
            i+=1
            
            
            
    return ran_xs,ran_ys#返回生成随机点坐标用于调试
            
            
            
def regis_careful_version2(slide_name, path_4x, path_10x, path_20x, shift, pic_num=1000, redun_4x = 1000, label = 'neg'):
    """
    细配函数version_2
    从4x图中随机取有含有细胞的点，上采样后分别在10x、20x图上滑动给出结果
    输出三张图片分别为128*128,256*256,512*512
    slide_name: 图片文件名
    path: 图片路径
    pic_num：取图的数量
    shift：给出一个列表表示粗配结果[4to10shift_x, 4to10shift_y, 4to20shift_x, 4to20shift_y]
    redun_4x: 在随机取点时作用在4x上的冗余，防止取到白边
    label: 表明当前片子的label用于命名
    """
    
    i = 0
    sample_size_20x = 1000
    sample_size_10x = 400
    size_4x = 102
    img_4x = openslide.OpenSlide(path_4x)
    img_10x = openslide.OpenSlide(path_10x)
    img_20x = openslide.OpenSlide(path_20x)
    
    w,h = img_4x.dimensions
    ran_xs = np.random.randint(redun_4x,high=w-size_4x-redun_4x,size = pic_num)
    ran_ys = np.random.randint(redun_4x,high=h-size_4x-redun_4x,size = pic_num)
    while i != pic_num:
        x_4x=ran_xs[i]
        y_4x=ran_ys[i]
        sample_4x = np.array(img_4x.read_region((x_4x,y_4x),level=0,size=(size_4x,size_4x)).convert('RGB'))
        sample_4x = cv2.cvtColor(sample_4x,cv2.COLOR_BGR2RGB)
        [cell_flag,_] = _BinarySP(cv2.resize(sample_4x,(512,512)))
        
        if not(cell_flag):
            ran_xs = np.delete(ran_xs,i)
            ran_ys = np.delete(ran_ys,i)
            ran_xs = np.hstack((ran_xs,np.random.randint(redun_4x,high=w-size_4x-redun_4x,size = 1)))
            ran_ys = np.hstack((ran_ys,np.random.randint(redun_4x,high=h-size_4x-redun_4x,size = 1)))
            continue
        else:
            #在10x上细配
            sample_4x_256 = cv2.resize(sample_4x,(256,256))
            safe_belt_10x = (sample_size_10x-256)//2
            x_10x = int(2.5*x_4x+shift[0]-safe_belt_10x)
            y_10x = int(2.5*y_4x+shift[1]-safe_belt_10x)
            sample_10x = np.array(img_10x.read_region((x_10x,y_10x),level = 0,size=(sample_size_10x,sample_size_10x)).convert('RGB'))
            sample_10x = cv2.cvtColor(sample_10x,cv2.COLOR_BGR2RGB)
            
            result = cv2.matchTemplate(sample_10x,sample_4x_256,cv2.TM_CCOEFF)
            _, maxval, _, maxloc = cv2.minMaxLoc(result)

            sample_10x_result = sample_10x[maxloc[1]:maxloc[1]+256,maxloc[0]:maxloc[0]+256,:]
            #测试用
            '''if not(os.path.exists('./10x_1')):
                os.mkdir('./10x_1')
            cv2.rectangle(sample_10x, maxloc, (maxloc[0]+256,maxloc[1]+256), [0, 0, 0])
            cv2.imwrite('./10x_1/{}_{}.tif'.format(int(slide_id),i),sample_10x)'''
            #
            
            #在20x上细配
            sample_4x_512 = cv2.resize(sample_4x,(512,512))
            safe_belt_20x = (sample_size_20x-512)//2
            x_20x = int(5*x_4x+shift[2]-safe_belt_20x)
            y_20x = int(5*y_4x+shift[3]-safe_belt_20x)
            sample_20x = np.array(img_20x.read_region((x_20x,y_20x),level = 0,size=(sample_size_20x,sample_size_20x)).convert('RGB'))
            sample_20x = cv2.cvtColor(sample_20x,cv2.COLOR_BGR2RGB)
            
            result = cv2.matchTemplate(sample_20x,sample_4x_512,cv2.TM_CCOEFF)
            _, maxval, _, maxloc = cv2.minMaxLoc(result)

            sample_20x_result = sample_20x[maxloc[1]:maxloc[1]+512,maxloc[0]:maxloc[0]+512,:]
            #测试用
            '''if not(os.path.exists('./20x_1')):
                os.mkdir('./20x_1')
            cv2.rectangle(sample_20x, maxloc, (maxloc[0]+512,maxloc[1]+512), [0, 0, 0])
            cv2.imwrite('./20x_1/{}_{}.tif'.format(int(slide_id),i),sample_20x)'''
            #
            
            #输出部分
            if not(os.path.exists('./regis_tif/'+label)):
                os.mkdir('./regis_tif/'+label)
            path = os.path.join('./regis_tif/',label)
            if not(os.path.exists(path+'/4x')):
                os.mkdir(path+'/4x')
            cv2.imwrite(path+'/4x/{}_{}.tif'.format(slide_name.rstrip('.svs.'),i),cv2.resize(sample_4x,(128,128)))
            if not(os.path.exists(path+'/10x')):
                os.mkdir(path+'/10x')
            cv2.imwrite(path+'/10x/{}_{}.tif'.format(slide_name.rstrip('.svs.'),i),sample_10x_result)
            if not(os.path.exists(path+'/20x')):
                os.mkdir(path+'/20x')
            cv2.imwrite(path+'/20x/{}_{}.tif'.format(slide_name.rstrip('.svs.'),i),sample_20x_result)
            i+=1
            
            
    return ran_xs,ran_ys#返回生成随机点坐标用于调试


def regis_careful_version3(slide_name, path_4x, path_10x, path_20x, shift, pic_num=1000, redun_4x = 2000, label = 'neg'):
    """
    细配函数version_3
    从4x图中随机取有含有细胞的点，上采样后分别在10x图上滑动给出结果,然后再根据10x图上采样后再在20x图上滑动给出结果
    输出三张图片分别为128*128,256*256,512*512
    slide_name: 图片文件名
    path: 图片路径
    pic_num：取图的数量
    shift：给出一个列表表示粗配结果[4to10shift_x, 4to10shift_y, 10to20shift_x, 10to20shift_y]
    redun_4x: 在随机取点时作用在4x上的冗余，防止取到白边
    label: 表明当前片子的label用于命名
    """
    
    i = 0
    sample_size_20x = 1200
    sample_size_10x = 400
    size_4x = 102
    img_4x = openslide.OpenSlide(path_4x)
    img_10x = openslide.OpenSlide(path_10x)
    img_20x = openslide.OpenSlide(path_20x)
    
    w,h = img_4x.dimensions
    ran_xs = np.random.randint(redun_4x,high=w-size_4x-redun_4x,size = pic_num)
    ran_ys = np.random.randint(redun_4x,high=h-size_4x-redun_4x,size = pic_num)
    while i != pic_num:
        x_4x=ran_xs[i]
        y_4x=ran_ys[i]
        sample_4x = np.array(img_4x.read_region((x_4x,y_4x),level=0,size=(size_4x,size_4x)).convert('RGB'))
        sample_4x = cv2.cvtColor(sample_4x,cv2.COLOR_BGR2RGB)
        [cell_flag,_] = _BinarySP(cv2.resize(sample_4x,(512,512)))
        
        if not(cell_flag):
            ran_xs = np.delete(ran_xs,i)
            ran_ys = np.delete(ran_ys,i)
            ran_xs = np.hstack((ran_xs,np.random.randint(redun_4x,high=w-size_4x-redun_4x,size = 1)))
            ran_ys = np.hstack((ran_ys,np.random.randint(redun_4x,high=h-size_4x-redun_4x,size = 1)))
            continue
        else:
            #在10x上细配
            sample_4x_256 = cv2.resize(sample_4x,(256,256))
            safe_belt_10x = (sample_size_10x-256)//2
            x_10x = int(2.5*x_4x+shift[0]-safe_belt_10x)
            y_10x = int(2.5*y_4x+shift[1]-safe_belt_10x)
            sample_10x = np.array(img_10x.read_region((x_10x,y_10x),level = 0,size=(sample_size_10x,sample_size_10x)).convert('RGB'))
            sample_10x = cv2.cvtColor(sample_10x,cv2.COLOR_BGR2RGB)
            
            result = cv2.matchTemplate(sample_10x,sample_4x_256,cv2.TM_CCOEFF)
            _, maxval, _, maxloc = cv2.minMaxLoc(result)

            sample_10x_result = sample_10x[maxloc[1]:maxloc[1]+256,maxloc[0]:maxloc[0]+256,:]
            #测试用
            '''if not(os.path.exists('./10x_1')):
                os.mkdir('./10x_1')
            cv2.rectangle(sample_10x, maxloc, (maxloc[0]+256,maxloc[1]+256), [0, 0, 0])
            cv2.imwrite('./10x_1/{}_{}.tif'.format(int(slide_id),i),sample_10x)'''
            #
            
            #在20x上细配
            sample_10x_512 = cv2.resize(sample_10x_result,(512,512))
            safe_belt_20x = (sample_size_20x-512)//2
            x_20x = int(2*x_10x+shift[2]-safe_belt_20x)
            y_20x = int(2*y_10x+shift[3]-safe_belt_20x)
            sample_20x = np.array(img_20x.read_region((x_20x,y_20x),level = 0,size=(sample_size_20x,sample_size_20x)).convert('RGB'))
            sample_20x = cv2.cvtColor(sample_20x,cv2.COLOR_BGR2RGB)
            
            result = cv2.matchTemplate(sample_20x,sample_10x_512,cv2.TM_CCOEFF)
            _, maxval, _, maxloc = cv2.minMaxLoc(result)

            sample_20x_result = sample_20x[maxloc[1]:maxloc[1]+512,maxloc[0]:maxloc[0]+512,:]
            #测试用
            '''if not(os.path.exists('./20x_1')):
                os.mkdir('./20x_1')
            cv2.rectangle(sample_20x, maxloc, (maxloc[0]+512,maxloc[1]+512), [0, 0, 0])
            cv2.imwrite('./20x_1/{}_{}.tif'.format(int(slide_id),i),sample_20x)'''
            #
            
            if not(os.path.exists('./regis_tif/'+label)):
                os.mkdir('./regis_tif/'+label)
            path = os.path.join('./regis_tif/',label)
            if not(os.path.exists(path+'/4x')):
                os.mkdir(path+'/4x')
            cv2.imwrite(path+'/4x/{}_{}.tif'.format(slide_name.rstrip('.svs.'),i),cv2.resize(sample_4x,(128,128)))
            if not(os.path.exists(path+'/10x')):
                os.mkdir(path+'/10x')
            cv2.imwrite(path+'/10x/{}_{}.tif'.format(slide_name.rstrip('.svs.'),i),sample_10x_result)
            if not(os.path.exists(path+'/20x')):
                os.mkdir(path+'/20x')
            cv2.imwrite(path+'/20x/{}_{}.tif'.format(slide_name.rstrip('.svs.'),i),sample_20x_result)
            i+=1
            
    return ran_xs,ran_ys#返回生成随机点坐标用于调试

def map_coors(inx,iny,mode,shift):
    """
    inx,iny:需要映射的x,y
    mode:映射模式{
        2.5+:从4x到10x
        2.5-:从10x到4x
        2+:从10x到20x
        2-:从20x到10x
        5+:从4x到20x
        5-:从20x到4x
    }
    shift:偏移量

    return：映射后的坐标(outx，outy)
    """
    if mode[-1]=='+':
        ratio = float(mode.rstrip('+'))
        outx=int(ratio*inx+shift[0])
        outy=int(ratio*iny+shift[1])
        
    elif mode[-1]=='-':
        ratio = float(mode.rstrip('-'))
        outx=int((inx-shift[0])//ratio)
        outy=int((iny-shift[1])//ratio)
        
    return (outx,outy)

def regis_patch_hrtolr(coors,size,shift,lr_path,hr_path,ratio,redundant = 100):
    """
    coors：所需要裁剪的patch的左上角坐标点(x,y)，对应高倍坐标
    size:所需要的patch的大小(w,h),对应高倍
    shift:两张片子之间的偏移量(sx,sy)
    lr_path,hr_path:存片路径
    redundant:由粗配误差决定,精配的冗余量

    由高倍向低倍映射
    return :[regied_lr_patch,hr_patch,maxval]
    """    

    lr_handle = openslide.OpenSlide(lr_path)
    hr_handle = openslide.OpenSlide(hr_path)
    
    x,y = coors
    w,h = size
    lw,lh = [int(i/ratio) for i in size]
    #利用粗配结果进行坐标映射
    lx,ly = map_coors(x,y,str(ratio)+'-',shift)
    #裁剪patch
    hr_patch = np.array(hr_handle.read_region((x,y),level=0,size=(w,h)).convert('RGB'))
    #resize到lr的尺度
    sample_hr_patch = cv2.resize(hr_patch,(lw,lh))
    #考虑冗余作用
    lx = lx - redundant
    ly = ly - redundant
    rlw = int(lw + redundant*2)
    rlh = int(lh + redundant*2)
    #裁剪小patch
    lr_patch = np.array(lr_handle.read_region((lx,ly),0,(rlw,rlh)).convert('RGB'))
    #模板匹配
    result = cv2.matchTemplate(lr_patch,sample_hr_patch,cv2.TM_CCOEFF_NORMED)
    _, maxval, _, maxloc = cv2.minMaxLoc(result)
    
    #裁剪结果
    regised_lr_patch = lr_patch[maxloc[1]:maxloc[1]+lh,maxloc[0]:maxloc[0]+lw,:]

    return [regised_lr_patch,hr_patch,maxval]



def regis_patch_lrtohr(coors,size,shift,lr_path,hr_path,ratio,redundant = 500):
    """
    coors：所需要裁剪的patch的左上角坐标点(x,y)，对应低倍坐标
    size:所需要的patch的大小(w,h),对应低倍
    shift:两张片子之间的偏移量(sx,sy)
    lr_path,hr_path:存片路径
    redundant:由粗配误差决定,精配的冗余量

    由高倍向低倍映射
    return :[regied_lr_patch,hr_patch,maxval]
    """    

    lr_handle = openslide.OpenSlide(lr_path)
    hr_handle = openslide.OpenSlide(hr_path)
    
    x,y = coors
    w,h = size
    if size==(102,102) and ratio==2.5:
        hw,hh = [256,256]
    elif size==(102,102) and ratio==5:
        hw,hh = [512,512]
    else:
        hw,hh = [int(i*ratio) for i in size]#考虑到102*2.5或*5不是256与512，这一步与hrtolr相比进行改变
    #利用粗配结果进行坐标映射
    hx,hy = map_coors(x,y,str(ratio)+'+',shift)
    #裁剪patch
    lr_patch = np.array(lr_handle.read_region((x,y),level=0,size=(w,h)).convert('RGB'))
    #resize到hr的尺度
    sample_lr_patch = cv2.resize(lr_patch,(hw,hh))
    #考虑冗余作用
    hx = hx - redundant
    hy = hy - redundant
    rhw = int(hw + redundant*2)
    rhh = int(hh + redundant*2)
    #裁剪小patch
    hr_patch = np.array(hr_handle.read_region((hx,hy),0,(rhw,rhh)).convert('RGB'))
    #模板匹配


    
    result = cv2.matchTemplate(hr_patch,sample_lr_patch,cv2.TM_CCOEFF_NORMED)
    _, maxval, _, maxloc = cv2.minMaxLoc(result)

    #裁剪结果
    regised_hr_patch = hr_patch[maxloc[1]:maxloc[1]+hh,maxloc[0]:maxloc[0]+hw,:]

    return [regised_hr_patch,lr_patch,maxval]
            


############
def regis_coarse_SURF(low_path, high_path, slide_ratio):
    """
    利用SURF进行粗配
    简单测试版
    
    改进意见：将20倍换用更小尺寸图进行特征点寻取
    """
    ratio = slide_ratio[1]/slide_ratio[0]

    lr = openslide.OpenSlide(low_path)
    hr = openslide.OpenSlide(high_path)

    lw,lh = lr.level_dimensions[1]
    lr_ratio = lr.level_downsamples[1]
    lr_patch = np.array(lr.read_region((0,0),level=1,size=(lw,lh)))
    lr_patch_gray = cv2.cvtColor(lr_patch, cv2.COLOR_BGR2GRAY)
    
    hw,hh = hr.level_dimensions[2]
    hr_ratio = hr.level_downsamples[2]
    hr_patch = np.array(hr.read_region((0,0),level=2,size=(hw,hh)))
    hr_patch_gray = cv2.cvtColor(hr_patch, cv2.COLOR_BGR2GRAY)
    
    surf = cv2.xfeatures2d.SURF_create(5000)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    
    kp_lr, des_lr = surf.detectAndCompute(lr_patch_gray,None)#des是描述子
    kp_hr, des_hr = surf.detectAndCompute(hr_patch_gray,None)
    
    
    """img1 = cv2.drawKeypoints(lr_patch_gray, kp_lr, None, (255,0,0), 4)
    img2 = cv2.drawKeypoints(hr_patch_gray, kp_hr, None, (255,0,0), 4)"""
    
    
    matches = flann.knnMatch(des_lr,des_hr,k=2)
    
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append([m])
    xs=[]
    ys=[]
    for i in good:
        queryIdx=i[0].queryIdx
        trainIdx=i[0].trainIdx
        x_lr,y_lr = kp_lr[queryIdx].pt
        x_hr,y_hr = kp_hr[trainIdx].pt
        x_lr *= lr_ratio
        y_lr *= lr_ratio
        x_hr *= hr_ratio
        y_hr *= hr_ratio
        xs.append(x_hr - x_lr*ratio)
        ys.append(y_hr - y_lr*ratio)
    std_x=np.std(xs)
    std_y=np.std(ys)
    ex=int(np.average(xs))
    ey=int(np.average(ys))
    return ex,ey,std_x,std_y
