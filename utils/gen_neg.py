import openslide
import cv2
import utilis
import numpy as np
import sys
slide_root = '/mnt/hdd1/SVS/4x/neg/'
save_root = '/mnt/sdb/neg_zx/'
slide_log = '/mnt/disk1/zhixing/neg_list.txt'
coarse_match_log = '/home/fj/SR/full_match.txt'
with open(slide_log) as f:
    slides_path = [slide_root+line.strip()+'.svs' for line in f]
utilis.path_checker(save_root)

img_perslide = 500
for slide_counter,path in enumerate(slides_path):
    print("Processing:[{}/{}]".format(slide_counter,len(slides_path)))
    img_counter = 0
    slide_id = path.split('/')[-1].split('.')[0]
    slide_handle = openslide.OpenSlide(path)
    w,h = slide_handle.dimensions
    safelength =int( w//3)
    ran_xs = np.random.randint(safelength,high=w - safelength,size = img_perslide*15)
    ran_ys = np.random.randint(safelength,high=h - safelength,size = img_perslide*15)
    for x,y in zip(ran_xs,ran_ys):
        lr_sample = np.array(slide_handle.read_region((x,y),level=0,size=(256,256)).convert('RGB'))
        flag,_ = utilis._BinarySP(lr_sample)
        if flag:
            img_path = save_root+slide_id+'_'+str(img_counter)+'.tif'
            img = cv2.cvtColor(lr_sample,cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path,img)
            img_counter += 1
            if img_counter>500:
                break
            sys.stdout.write('{}:[{}/{}]'.format(slide_id,img_counter,img_perslide))
            sys.stdout.flush()
