import sys
import glob
import os
from PIL import Image
import numpy as np
from cStringIO import StringIO
import pyDarknet
import base64
import cPickle
import time

src_path = sys.argv[1]
out_path = sys.argv[2]
if len(sys.argv) > 3:
    gpu = int(sys.argv[3])
else:
    gpu = 0

parts = glob.glob(os.path.join(src_path,'part-*'))

cnt = 0

#init detector
pyDarknet.ObjectDetector.set_device(gpu)
detector = pyDarknet.ObjectDetector('cfg/yolo.cfg','/media/ssd/models/yolo/yolo.weights')


for part in parts:
    rst_list = []
    start = time.time()
    for line in open(part):
        im_id = line.strip().split('\t')[0]
        im_data  = line.strip().split('\t')[-1]
        if im_data == 'N/A':
            continue
        try:
            im = Image.open(StringIO(base64.b64decode(im_data)))
        except:
            continue
        try:
            im.load()
        except IOError:
            pass

        if im.mode == 'L' or im.mode == '1' or im.mode == 'P':
            new_arr = np.empty((im.size[1], im.size[0],3), dtype=np.uint8)
            new_arr[:,:,:] = np.array(im)[:,:,np.newaxis]
            im = Image.fromarray(new_arr)

        rst, rt = detector.detect_object(im)

        rst_list.append((im_id, rst))
        cnt += 1

        if cnt % 500 == 0:
            cur_time = time.time()
          
            print cnt, int(cur_time - start)
    
    part_name = part.split('/')[-1]+'.pc'
    out_name = os.path.join(out_path, part_name)
    cPickle.dump(rst_list, open(out_name,'wb'), cPickle.HIGHEST_PROTOCOL)
    print '{} finished'.format(part)




