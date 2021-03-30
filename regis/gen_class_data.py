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

labelCSV_path = '/mnt/diskarray/slides_data/SVS/labelCSV/'
"""slide_ids_path = '/mnt/diskarray/classdata/coarse_regis_log.txt'"""
svs_path = '/mnt/diskarray/slides_data/SVS/'
svs_10x_path = '/mnt/diskarray/slides_data/SVS_new/'
store_lr_path = './4x/'
store_mr_path = './10x/'
store_hr_path = './20x/'
coarse_match_log = './coarse_regis_log.txt'
#检查路径
utilis.path_checker(store_lr_path)
utilis.path_checker(store_mr_path)
utilis.path_checker(store_hr_path)
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
print('共需处理:',len(slide_ids))

l = len(slide_ids)
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
        slide_pool[i].append(slide_ids[i*m+j])
        label_pool[i].append(labels[i*m+j])
        ex4to20_pool[i].append(exs_4to20[i*m+j])
        ey4to20_pool[i].append(eys_4to20[i*m+j])
        ex10to20_pool[i].append(exs_10to20[i*m+j])
        ey10to20_pool[i].append(eys_10to20[i*m+j])
if n != 0:
    
    for i in range(n):
        slide_pool[i].append(slide_ids[thread_num*m+i])
        label_pool[i].append(labels[thread_num*m+i])
        ex4to20_pool[i].append(exs_4to20[thread_num*m+i])
        ey4to20_pool[i].append(eys_4to20[thread_num*m+i])
        ex10to20_pool[i].append(exs_10to20[thread_num*m+i])
        ey10to20_pool[i].append(eys_10to20[thread_num*m+i])

redun = 5000#取在20倍图上、用来防止随机点取到白边的冗余
pic_num = 1000
target_size = 1024

def gen(thread,labels,slide_ids,exs_4to20,eys_4to20,exs_10to20,eys_10to20):
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
        if 'posi' in label:
            try:
                infos = utilis.get_coors_from_csv(labelCSV_path,slide_id,target_20x=target_size)
            except Exception as ex:
                print(ex)
                with open('./wrong_log_1.txt','a') as f:
                    f.write(slide_id+'\n')
                    f.write('csv'+'\n')
                continue
            for category,shift_list in infos.items():
                counter=0
                if len(shift_list)==0:
                    continue
                utilis.path_checker(store_lr_path+category+'/')
                utilis.path_checker(store_mr_path+category+'/')
                utilis.path_checker(store_hr_path+category+'/')
                for _,x,y,w,h in shift_list:
                
                    try:
                        [lr_result,hr_result,maxval_1] = utilis.regis_patch_hrtolr((x,y),(target_size,target_size),(ex_4to20,ey_4to20),lr_path,hr_path,ratio=5,redundant = 400)
                        [mr_result,_,maxval_2] = utilis.regis_patch_hrtolr((x,y),(target_size,target_size),(ex_10to20,ey_10to20),mr_path,hr_path,ratio=2,redundant = 400)
                        '''if (maxval_1<0.5) and (maxval_2>0.8):
                            print(counter)
                            io.imsave('./tmp/'+slide_id+'_'+category+'_4x_'+str(counter)+'.tif',lr_result)
                            io.imsave('./tmp/'+slide_id+'_'+category+'_10x_'+str(counter)+'.tif',mr_result)
                            io.imsave('./tmp/'+slide_id+'_'+category+'_20x_'+str(counter)+'.tif',hr_result)
                            counter+=1'''


                        if (maxval_1<0.8) or (maxval_2<0.8):
                            print(slide_id,category,maxval_1,maxval_2)
                        else:
                            lr_result = cv2.resize(lr_result,(target_size//4,target_size//4))
                            io.imsave(store_lr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),lr_result)
                            io.imsave(store_mr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),mr_result)
                            io.imsave(store_hr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),hr_result)
                            counter+=1
                    
                        #数据增强
                        [lr_result,hr_result,maxval_1] = utilis.regis_patch_hrtolr((x+50,y),(target_size,target_size),(ex_4to20,ey_4to20),lr_path,hr_path,ratio=5,redundant = 400)
                        [mr_result,_,maxval_2] = utilis.regis_patch_hrtolr((x+50,y),(target_size,target_size),(ex_10to20,ey_10to20),mr_path,hr_path,ratio=2,redundant = 400)
                        if (maxval_1<0.8) or (maxval_2<0.8):
                            print(slide_id,category,maxval_1,maxval_2)
                        else:    
                            lr_result = cv2.resize(lr_result,(target_size//4,target_size//4))
                            io.imsave(store_lr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),lr_result)
                            io.imsave(store_mr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),mr_result)
                            io.imsave(store_hr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),hr_result)
                            counter+=1
                    
                        [lr_result,hr_result,maxval_1] = utilis.regis_patch_hrtolr((x-50,y),(target_size,target_size),(ex_4to20,ey_4to20),lr_path,hr_path,ratio=5,redundant = 400)
                        [mr_result,_,maxval_2] = utilis.regis_patch_hrtolr((x-50,y),(target_size,target_size),(ex_10to20,ey_10to20),mr_path,hr_path,ratio=2,redundant = 400)
                        if (maxval_1<0.8) or (maxval_2<0.8):
                            print(slide_id,category,maxval_1,maxval_2)
                        else:
                            lr_result = cv2.resize(lr_result,(target_size//4,target_size//4))
                            io.imsave(store_lr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),lr_result)
                            io.imsave(store_mr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),mr_result)
                            io.imsave(store_hr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),hr_result)
                            counter+=1
                    
                        [lr_result,hr_result,maxval_1] = utilis.regis_patch_hrtolr((x,y+50),(target_size,target_size),(ex_4to20,ey_4to20),lr_path,hr_path,ratio=5,redundant = 400)
                        [mr_result,_,maxval_2] = utilis.regis_patch_hrtolr((x,y+50),(target_size,target_size),(ex_10to20,ey_10to20),mr_path,hr_path,ratio=2,redundant = 400)
                        if (maxval_1<0.8) or (maxval_2<0.8):
                            print(slide_id,category,maxval_1,maxval_2)
                        else:
                            lr_result = cv2.resize(lr_result,(target_size//4,target_size//4))
                            io.imsave(store_lr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),lr_result)
                            io.imsave(store_mr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),mr_result)
                            io.imsave(store_hr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),hr_result)
                            counter+=1
                    
                        [lr_result,hr_result,maxval_1] = utilis.regis_patch_hrtolr((x,y-50),(target_size,target_size),(ex_4to20,ey_4to20),lr_path,hr_path,ratio=5,redundant = 400)
                        [mr_result,_,maxval_2] = utilis.regis_patch_hrtolr((x,y-50),(target_size,target_size),(ex_10to20,ey_10to20),mr_path,hr_path,ratio=2,redundant = 400)
                        if (maxval_1<0.8) or (maxval_2<0.8):
                            print(slide_id,category,maxval_1,maxval_2)
                        else:
                            lr_result = cv2.resize(lr_result,(target_size//4,target_size//4))
                            io.imsave(store_lr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),lr_result)
                            io.imsave(store_mr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),mr_result)
                            io.imsave(store_hr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),hr_result)
                            counter+=1

                        [lr_result,hr_result,maxval_1] = utilis.regis_patch_hrtolr((x-50,y-50),(target_size,target_size),(ex_4to20,ey_4to20),lr_path,hr_path,ratio=5,redundant = 400)
                        [mr_result,_,maxval_2] = utilis.regis_patch_hrtolr((x-50,y-50),(target_size,target_size),(ex_10to20,ey_10to20),mr_path,hr_path,ratio=2,redundant = 400)
                        if (maxval_1<0.8) or (maxval_2<0.8):
                            print(slide_id,category,maxval_1,maxval_2)
                        else:
                            lr_result = cv2.resize(lr_result,(target_size//4,target_size//4))
                            io.imsave(store_lr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),lr_result)
                            io.imsave(store_mr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),mr_result)
                            io.imsave(store_hr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),hr_result)
                            counter+=1

                        [lr_result,hr_result,maxval_1] = utilis.regis_patch_hrtolr((x+50,y-50),(target_size,target_size),(ex_4to20,ey_4to20),lr_path,hr_path,ratio=5,redundant = 400)
                        [mr_result,_,maxval_2] = utilis.regis_patch_hrtolr((x+50,y-50),(target_size,target_size),(ex_10to20,ey_10to20),mr_path,hr_path,ratio=2,redundant = 400)
                        if (maxval_1<0.8) or (maxval_2<0.8):
                            print(slide_id,category,maxval_1,maxval_2)
                        else:
                            lr_result = cv2.resize(lr_result,(target_size//4,target_size//4))
                            io.imsave(store_lr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),lr_result)
                            io.imsave(store_mr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),mr_result)
                            io.imsave(store_hr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),hr_result)
                            counter+=1

                        [lr_result,hr_result,maxval_1] = utilis.regis_patch_hrtolr((x-50,y+50),(target_size,target_size),(ex_4to20,ey_4to20),lr_path,hr_path,ratio=5,redundant = 400)
                        [mr_result,_,maxval_2] = utilis.regis_patch_hrtolr((x-50,y+50),(target_size,target_size),(ex_10to20,ey_10to20),mr_path,hr_path,ratio=2,redundant = 400)
                        if (maxval_1<0.8) or (maxval_2<0.8):
                            print(slide_id,category,maxval_1,maxval_2)
                        else:

                            lr_result = cv2.resize(lr_result,(target_size//4,target_size//4))
                            io.imsave(store_lr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),lr_result)
                            io.imsave(store_mr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),mr_result)
                            io.imsave(store_hr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),hr_result)
                            counter+=1

                        [lr_result,hr_result,maxval_1] = utilis.regis_patch_hrtolr((x+50,y+50),(target_size,target_size),(ex_4to20,ey_4to20),lr_path,hr_path,ratio=5,redundant = 400)
                        [mr_result,_,maxval_2] = utilis.regis_patch_hrtolr((x+50,y+50),(target_size,target_size),(ex_10to20,ey_10to20),mr_path,hr_path,ratio=2,redundant = 400)
                        if (maxval_1<0.8) or (maxval_2<0.8):
                            print(slide_id,category,maxval_1,maxval_2)
                        else:

                            lr_result = cv2.resize(lr_result,(target_size//4,target_size//4))
                            io.imsave(store_lr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),lr_result)
                            io.imsave(store_mr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),mr_result)
                            io.imsave(store_hr_path+'{}/posi_{}_{}.tif'.format(category,slide_id,counter),hr_result)
                            counter+=1
                    except Exception as ex:
                        print(ex)
                        with open('./wrong_log_1.txt','a') as f:
                            f.write(slide_id+'\n')
                            f.write('regis'+'\n')
                        continue
            '''counter = 0
            try:
                xs,ys=utilis.Generate_negative_sample_our(hr_path, '/mnt/hdd1/SVS/imgBIN/', '/mnt/hdd1/SVS/labelMAP', slide_id, level=2, numSample=100, size=1024)    
                for x,y in zip(xs,ys):
                    [lr_result,hr_result,maxval_1] = utilis.regis_patch_hrtolr((x,y),(1024,1024),(ex_4to20,ey_4to20),lr_path,hr_path,ratio=5,redundant = 400)
                    [mr_result,_,maxval_2] = utilis.regis_patch_hrtolr((x,y),(1024,1024),(ex_10to20,ey_10to20),mr_path,hr_path,ratio=2,redundant = 400)
                    if (maxval_1<0.8) or (maxval_2<0.8):
                        print('neg',maxval_1,maxval_2)
                    else:
                        lr_result = cv2.resize(lr_result,(target_size//4,target_size//4))
                        io.imsave(store_lr_path+'neg/neg_{}_{}.tif'.format(slide_id,counter),lr_result)
                        io.imsave(store_mr_path+'neg/neg_{}_{}.tif'.format(slide_id,counter),mr_result)
                        io.imsave(store_hr_path+'neg/neg_{}_{}.tif'.format(slide_id,counter),hr_result)
                        counter+=1
            except:
                with open('./wrong_log_1.txt','a') as f:
                    f.write(slide_id+'\n')
                    f.write('regis'+'\n')
                continue'''
            print('-'*10)
            print('Thread {}, No.{}, {} Over'.format(thread,i+1,slide_id))
            print('-'*10)
                
        '''elif 'neg' in label:
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
                [lr_result, hr_result,maxval_1] = utilis.regis_patch_hrtolr((x,y),(target_size,target_size),(ex_4to20,ey_4to20),lr_path,hr_path,ratio=5,redundant = 800)
                [mr_result, _,maxval_2] = utilis.regis_patch_hrtolr((x,y),(target_size,target_size),(ex_10to20,ey_10to20),mr_path,hr_path,ratio=2,redundant = 1000)
                #移动
            
            
                if (maxval_1<0.8) or (maxval_2<0.8) or not(utilis._BinarySP(hr_result)[0]):
                    print(slide_id,counter, maxval_1, maxval_2)
                    continue
                lr_result = cv2.resize(lr_result,(target_size//4,target_size//4))
                io.imsave(store_lr_path+'{}/neg_{}_{}.tif'.format(label,slide_id,counter),lr_result)
                io.imsave(store_mr_path+'{}/neg_{}_{}.tif'.format(label,slide_id,counter),mr_result)
                io.imsave(store_hr_path+'{}/neg_{}_{}.tif'.format(label,slide_id,counter),hr_result)
                counter+=1
                if counter==pic_num:
                    break

            print('-'*10)  
            print('Thread {}, No.{}, {} Over'.format(thread,i+1,slide_id))
            print('-'*10)'''

p= multiprocessing.Pool(processes=thread_num)
for i in range(thread_num):
    p.apply_async(gen, (i, label_pool[i], slide_pool[i], ex4to20_pool[i],ey4to20_pool[i],ex10to20_pool[i],ey10to20_pool[i],   ))

p.close()
p.join()

