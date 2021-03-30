#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:23:17 2019

@author: fj
"""
import sys
sys.path.append('./')
import utilis
import os
import multiprocessing

path = '/mnt/diskarray/slides_data/SVS/'
path_10x = '/mnt/diskarray/slides_data/SVS_new/'
slide_ids_path = '/mnt/diskarray/slides_data/SVS/slide_info.txt'
#获取slide文件名列表与label列表
slide, label = utilis.get_slide_name(slide_ids_path,split = '_')
thread_num = 5
l = len(slide)
m = l//thread_num
n = l%thread_num
slide_pool = []
label_pool = []
for i in range(thread_num):
    slide_pool.append([])
    label_pool.append([])
    for j in range(m):
        slide_pool[i].append(slide[i*m+j])
        label_pool[i].append(label[i*m+j])
if n != 0:
    for i in range(n):
        slide_pool[i].append(slide[thread_num*m+i])
        label_pool[i].append(label[thread_num*m+i])
        

        
 


def regis(pairs):
    for pair in pairs:
        slide_name=pair[0]
        lab=pair[1]
        print('-'*20)
        print('-'*20)
        print(slide_name)
        print('Regis_coarse begin')
    
        #4x-10x粗配
        print('4\t10')
        low_path = os.path.join(path, '4x', lab,  slide_name)
        #path_4x = low_path
        high_path = os.path.join(path_10x, '10x', lab,  slide_name)
        #path_10x = high_path
        try:
            ex1,ey1,dW1,dH1,flag = utilis.regis_coarse(low_path, high_path, [4,10],4000)
        except Exception as ex:
            print(ex)
        if not flag:
            with open('./wrong_match.txt','a') as f:
                f.write(slide_name+'\n')
            continue
            
    
        
        #10x-20x粗配
        print('10\t20')
        low_path = os.path.join(path_10x, '10x', lab,  slide_name)
        high_path = os.path.join(path, '20x', lab,  slide_name)
        #path_20x = high_path
        try:
            ex2,ey2,dW2,dH2,flag = utilis.regis_coarse(low_path, high_path, [10,20],4000)
        except Exception as ex:
            print(ex)
        if not flag:
            with open('./wrong_match.txt','a') as f:
                f.write(slide_name+'\n')
            continue

        utilis.write_regis_result(slide_name, lab, [4,10], ex1, ey1, dW1, dH1, output_file = './coarse_regis_log.txt')
        utilis.write_regis_result(slide_name, lab, [10,20], ex2, ey2, dW2, dH2, output_file = './coarse_regis_log.txt')
        utilis.write_regis_result(slide_name, lab, [4,20], int(2*ex1+ex2), int(2*ey1+ey2), [], [], output_file = './coarse_regis_log.txt')
        print('Regis_coarse over')
    

p= multiprocessing.Pool(processes=thread_num)
for i in range(thread_num):
    pairs=list(zip(slide_pool[i],label_pool[i])) 
    p.apply_async(regis, (pairs, ))

p.close()
p.join()

    
