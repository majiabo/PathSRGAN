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

labelCSV_path = '/home/fj/SVS/labelCSV/'
slide_ids_path = './slide_info.txt'
svs_path = '/home/fj/SVS/'
store_lr_path = '/home/fj/Data/70k_2/4x/'
store_hr_path = '/home/fj/Data/70k_2/20x/'
coarse_match_log = '/home/fj/SR/full_match.txt'
#检查路径
utilis.path_checker(store_lr_path)
utilis.path_checker(store_hr_path)
utilis.path_checker(store_lr_path+'neg/')
utilis.path_checker(store_hr_path+'neg/')


#获取文件名列表与label列表
labels,slide_ids,exs,eys = utilis.read_coarse_regis_infos(coarse_match_log,5)
print('共需处理:',len(slide_ids))
target_size = 102
redun = 5000#取在4倍图上、用来防止随机点取到白边的冗余
pic_num = 1000
for label,slide_id,ex,ey in zip(labels,slide_ids,exs,eys):
    print('Processing:',slide_id)
    print('status:',label)
    print('X_shift:',ex)
    print('Y_shift:',ey)
    #test 

    lr_path = os.path.join(svs_path,'4x',label,slide_id+'.svs')
    hr_path = os.path.join(svs_path,'20x',label,slide_id+'.svs')
    #获取坐标
    if 'posi' in label:
        try:
            infos = utilis.get_coors_from_csv(labelCSV_path,slide_id,target_20x=512)
        except:
            with open('./wrong_log_1.txt','a') as f:
                f.write(slide_id+'\n')
            continue
        for category,shift_list in infos.items():
            counter=0
            utilis.path_checker(store_lr_path+category+'/')
            utilis.path_checker(store_hr_path+category+'/')
            for _,x,y,w,h in shift_list:
            
                try:
                    [lr_result,hr_result,maxval] = utilis.regis_patch_hrtolr((x,y),(w,h),(ex,ey),lr_path,hr_path,ratio=5,redundant = 400)
                    if (maxval<0.9):
                        continue
                    lr_result = cv2.resize(lr_result,(128,128))
                    io.imsave(store_lr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),lr_result)
                    io.imsave(store_hr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),hr_result)
                    counter+=1
                
                    #数据增强
                    [lr_result,hr_result,maxval] = utilis.regis_patch_hrtolr((x+50,y),(w,h),(ex,ey),lr_path,hr_path,ratio=5,redundant = 400)
                    if (maxval<0.9):
                        continue
                    lr_result = cv2.resize(lr_result,(128,128))
                    io.imsave(store_lr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),lr_result)
                    io.imsave(store_hr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),hr_result)
                    counter+=1
                
                    [lr_result,hr_result,maxval] = utilis.regis_patch_hrtolr((x-50,y),(w,h),(ex,ey),lr_path,hr_path,ratio=5,redundant = 400)
                    if (maxval<0.9):
                        continue
                    lr_result = cv2.resize(lr_result,(128,128))
                    io.imsave(store_lr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),lr_result)
                    io.imsave(store_hr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),hr_result)
                    counter+=1
                
                    [lr_result,hr_result,maxval] = utilis.regis_patch_hrtolr((x,y+50),(w,h),(ex,ey),lr_path,hr_path,ratio=5,redundant = 400)
                    if (maxval<0.9):
                        continue
                    lr_result = cv2.resize(lr_result,(128,128))
                    io.imsave(store_lr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),lr_result)
                    io.imsave(store_hr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),hr_result)
                    counter+=1
                
                    [lr_result,hr_result,maxval] = utilis.regis_patch_hrtolr((x,y-50),(w,h),(ex,ey),lr_path,hr_path,ratio=5,redundant = 400)
                    if (maxval<0.9):
                        continue
                    lr_result = cv2.resize(lr_result,(128,128))
                    io.imsave(store_lr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),lr_result)
                    io.imsave(store_hr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),hr_result)
                    counter+=1

                    [lr_result,hr_result,maxval] = utilis.regis_patch_hrtolr((x-50,y-50),(w,h),(ex,ey),lr_path,hr_path,ratio=5,redundant = 400)
                    if (maxval<0.9):
                        continue
                    lr_result = cv2.resize(lr_result,(128,128))
                    io.imsave(store_lr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),lr_result)
                    io.imsave(store_hr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),hr_result)
                    counter+=1

                    [lr_result,hr_result,maxval] = utilis.regis_patch_hrtolr((x+50,y-50),(w,h),(ex,ey),lr_path,hr_path,ratio=5,redundant = 400)
                    if (maxval<0.9):
                        continue
                    lr_result = cv2.resize(lr_result,(128,128))
                    io.imsave(store_lr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),lr_result)
                    io.imsave(store_hr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),hr_result)
                    counter+=1

                    [lr_result,hr_result,maxval] = utilis.regis_patch_hrtolr((x-50,y+50),(w,h),(ex,ey),lr_path,hr_path,ratio=5,redundant = 400)
                    if (maxval<0.9):
                        continue
                    lr_result = cv2.resize(lr_result,(128,128))
                    io.imsave(store_lr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),lr_result)
                    io.imsave(store_hr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),hr_result)
                    counter+=1

                    [lr_result,hr_result,maxval] = utilis.regis_patch_hrtolr((x+50,y+50),(w,h),(ex,ey),lr_path,hr_path,ratio=5,redundant = 400)
                    if (maxval<0.9):
                        continue
                    lr_result = cv2.resize(lr_result,(128,128))
                    io.imsave(store_lr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),lr_result)
                    io.imsave(store_hr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),hr_result)
                    counter+=1
                except:
                    with open('./wrong_log_1.txt','a') as f:
                        f.write(slide_id+'\n')
                    continue
        print('{} over'.format(slide_id))
        print('-'*10)
            
    elif 'neg' in label:
        try:
            hr_handle = openslide.OpenSlide(hr_path)
        except:
            with open('./wrong_log_1.txt','a') as f:
                f.write(slide_id+'\n')
            continue
        hr_w,hr_h = hr_handle.dimensions
        xs = np.random.randint(redun,high=hr_w-target_size-redun,size = pic_num*10)
        ys = np.random.randint(redun,high=hr_h-target_size-redun,size = pic_num*10)
        counter=0
        for x,y in zip(xs,ys):
            [lr_result, hr_result,maxval] = utilis.regis_patch_hrtolr((x,y),(512,512),(ex,ey),lr_path,hr_path,ratio=5,redundant = 100)
            #移动
        
        
            if (maxval<0.9) or not(utilis._BinarySP(hr_result)[0]):
                continue
            lr_result = cv2.resize(lr_result,(128,128))
            io.imsave(store_lr_path+'{}/neg_{}_{}.tif'.format(label,slide_id,counter),lr_result)
            io.imsave(store_hr_path+'{}/neg_{}_{}.tif'.format(label,slide_id,counter),hr_result)
            counter+=1
            if counter==pic_num:
                break
            
        print('{} over'.format(slide_id))
        print('-'*10)
