"""
time:2019.3.7
author:majiabo
func:get date with labels
"""

import utilis
import os
from skimage import io
import cv2
import numpy as np
import openslide
import multiprocessing

def get_slide_name(file):
    """
    读取slide的file_name以及label
    slide_name格式:
    neg/1149300.svs or posi/1149300.svs
    """
    label=[]
    slide_name_result=[]
    with open(file,'r') as f:
        slide_name = f.readlines()
        for i in slide_name:
            label.append(i.split('/')[0])
            i=i.split('/')[1]
            slide_name_result.append(i.rstrip('.svs\n'))
        return slide_name_result, label


#labelCSV_path = '/home/fj/SVS/labelCSV/'
slide_ids_path = './id_lost.txt'
svs_path = '/mnt/diskarray/slides_data/SVS/'
svs_10x_path = '/mnt/diskarray/slides_data/SVS_new/'
store_lr_path = '/mnt/diskarray/srdata/new_add/4x/'
store_mr_path = '/mnt/diskarray/srdata/new_add/10x/'
store_hr_path = '/mnt/diskarray/srdata/new_add/20x/'
coarse_match_log = './coarse_regis_log_3.txt'
#检查路径
utilis.path_checker(store_lr_path)
utilis.path_checker(store_mr_path)
utilis.path_checker(store_hr_path)
utilis.path_checker(store_lr_path+'posi/')
utilis.path_checker(store_mr_path+'posi/')
utilis.path_checker(store_hr_path+'posi/')
utilis.path_checker(store_lr_path+'neg/')
utilis.path_checker(store_mr_path+'neg/')
utilis.path_checker(store_hr_path+'neg/')

thread_num = 5
label_pool=[]
slide_pool=[]
ex4to20_pool=[]
ey4to20_pool=[]
ex10to20_pool=[]
ey10to20_pool=[]

#获取文件名列表与label列表
labels_4to20,slide_ids_4to20,exs_4to20,eys_4to20 = utilis.read_coarse_regis_infos(coarse_match_log,5)
labels_10to20,slide_ids_10to20,exs_10to20,eys_10to20 = utilis.read_coarse_regis_infos(coarse_match_log,2)
if labels_4to20 == labels_10to20 and slide_ids_4to20 == slide_ids_10to20:
    labels = labels_10to20
    slide_ids = slide_ids_10to20

tmp=[]
for i in range(len(slide_ids)):
    if slide_ids[i] in get_slide_name(slide_ids_path)[0]:
        tmp.append(i)

print('共需处理:',len(tmp))

l = len(tmp)
m = l//thread_num
n = l%thread_num
for i in range(thread_num):
    slide_pool.append([])
    label_pool.append([])
    ex4to20_pool.append([])
    ey4to20_pool.append([])
    ex10to20_pool.append([])
    ey10to20_pool.append([])
    for j in range(m):
        index = tmp[i*m+j]
        slide_pool[i].append(slide_ids[index])
        label_pool[i].append(labels[index])
        ex4to20_pool[i].append(exs_4to20[index])
        ey4to20_pool[i].append(eys_4to20[index])
        ex10to20_pool[i].append(exs_10to20[index])
        ey10to20_pool[i].append(eys_10to20[index])

if n != 0:
    for i in range(n):
        index = tmp[thread_num*m+i]
        slide_pool[i].append(slide_ids[index])
        label_pool[i].append(labels[index])
        ex4to20_pool[i].append(exs_4to20[index])
        ey4to20_pool[i].append(eys_4to20[index])
        ex10to20_pool[i].append(exs_10to20[index])
        ey10to20_pool[i].append(eys_10to20[index])

redun = 5000#取在20倍图上、用来防止随机点取到白边的冗余
pic_num = 2000
target_size = 512

def gen(thread,labels,slide_ids,exs_4to20,eys_4to20,exs_10to20,eys_10to20):
    try:
        for i in range(len(slide_ids)):
            label = labels[i]
            slide_id = slide_ids[i]
            ex_4to20 = exs_4to20[i]
            ey_4to20 = eys_4to20[i]
            ex_10to20 = exs_10to20[i]
            ey_10to20 = eys_10to20[i]
            print('Thread {}, No.{}, {}, {}'.format(thread,i+1,slide_id,label))
            """print('X_shift:',ex)
            print('Y_shift:',ey)"""
            #test 

            lr_path = os.path.join(svs_path,'4x',label,slide_id+'.svs')
            mr_path = os.path.join(svs_10x_path,'10x',label,slide_id+'.svs')
            hr_path = os.path.join(svs_path,'20x',label,slide_id+'.svs')
            #获取坐标
            
            
            try:
                hr_handle = openslide.OpenSlide(hr_path)
            except:
                with open('./wrong_log_1.txt','a') as f:
                    f.write(slide_id+'\n')
                continue
            hr_w,hr_h = hr_handle.dimensions
            xs = np.random.randint(redun,high=hr_w-target_size-redun,size = pic_num*10)
            ys = np.random.randint(redun,high=hr_h-target_size-redun,size = pic_num*10)
            counter=1
            for x,y in zip(xs,ys):
                #[lr_result, hr_result,maxval_1] = utilis.regis_patch_hrtolr((x,y),(target_size,target_size),(ex_4to20,ey_4to20),lr_path,hr_path,ratio=5,redundant = 800)
                [mr_result, hr_result,maxval_2, mr, mr_rec] = utilis.regis_patch_hrtolr((x,y),(target_size,target_size),(ex_10to20,ey_10to20),mr_path,hr_path,ratio=2,redundant = 1000)
                #移动
            
            
                #if (maxval_1<0.8) or (maxval_2<0.8) or not(utilis._BinarySP(cv2.resize(hr_result,(512,512)))[0]):
                    #print(counter, maxval_1, maxval_2)
                    #continue
                #lr_result = cv2.resize(lr_result,(target_size//4,target_size//4))
                #io.imsave(store_lr_path+'{}/{}_{}.tif'.format(label,slide_id,counter),lr_result)
                #io.imsave(store_mr_path+'{}/{}_{}.tif'.format(label,slide_id,counter),mr_result)
                #io.imsave(store_hr_path+'{}/{}_{}.tif'.format(label,slide_id,counter),hr_result)
                io.imsave('./lr_result.tif',mr_result)
                io.imsave('./hr_result.tif',hr_result)
                io.imsave('./lr_total.tif',mr)
                io.imsave('./lr_rec.tif',mr_rec)
                print(counter)
                counter+=1
                if counter%200==0:
                    print(thread,counter)
                if counter==pic_num:
                    break
    
            print('-'*10)  
            print('Thread {}, No.{}, {} Over'.format(thread,i+1,slide_id))
            print('-'*10)
    except Exception as ex:
        print(ex)

p= multiprocessing.Pool(processes=thread_num)
for i in range(thread_num):
    p.apply_async(gen, (i, label_pool[i], slide_pool[i], ex4to20_pool[i],ey4to20_pool[i],ex10to20_pool[i],ey10to20_pool[i],   ))
p.close()
p.join()

