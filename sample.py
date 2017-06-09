from PIL import Image
import PIL
import numpy as np
import sys, os
import tensorflow as tf
from multiprocessing import Pool
import glob
import random

normal_paths = glob.glob('/home/mediwhale/data/eye/cropped_image/normal/' + '*.npy')
cataract_paths = glob.glob('/home/mediwhale/data/eye/cropped_image/abnormal/cataract/' + '*.npy')
retina_paths = glob.glob('/home/mediwhale/data/eye/cropped_image/abnormal/retina/' + '*.npy')
glaucoma_paths = glob.glob('/home/mediwhale/data/eye/cropped_image/abnormal/glaucoma/' + '*.npy')
cata_glau_paths = glob.glob('/home/mediwhale/data/eye/cropped_image/abnormal/cataract_glaucoma/' + '*.npy')
retina_cata = glob.glob('/home/mediwhale/data/eye/cropped_image/abnormal/retina_cataract/' + '*.npy')
retina_glau = glob.glob('/home/mediwhale/data/eye/cropped_image/abnormal/retina_glaucoma/' + '*.npy')
abnormal_paths = []
abnormal_paths.extend(cataract_paths)
abnormal_paths.extend(retina_paths)
abnormal_paths.extend(glaucoma_paths)
abnormal_paths.extend(cata_glau_paths)
abnormal_paths.extend(retina_cata)
abnormal_paths.extend(retina_glau)
print len(abnormal_paths)
normal_labels = np.ones([len(normal_paths)])
abnormal_labels = np.zeros([len(abnormal_paths)])

print len(normal_labels)
print normal_labels
print len(abnormal_labels)
print abnormal_labels
random.shuffle(normal_paths)
random.shuffle(abnormal_paths)

tfrecord_path='./sample.tfrecord'
labels=normal_labels
src_paths=normal_paths

labels=labels.astype(np.int64)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

writer = tf.python_io.TFRecordWriter(tfrecord_path)
all_paths_labels = zip(src_paths[:100], labels[:100])
f = open(tfrecord_path.replace('.tfrecord', '_path.txt') , 'w')
#e.g) ./normal.tfrecord--> normal_path.txt

def image2string(path_label):

    resize_=(288,288)
    path,label=path_label
    if not path.endswith('.npy'):
        np_img = np.asarray(Image.open(path))
    else:
        np_img = np.load(path)

    img = Image.fromarray(np_img)
    img = img.resize(resize_, PIL.Image.ANTIALIAS)
    np_img = np.asarray(img)
    height = np_img.shape[0]
    width = np_img.shape[1]
    raw_img = np_img.tostring()

    return raw_img, height, width ,label , path

pool = Pool()
ind=0
for raw_img, height, width,label,path in pool.imap(image2string ,all_paths_labels ):
    try:
        print height,width,label
        f.write(path.split('/')[-1]+'\n')
        msg = '\r-Progress : {0}'.format(str(ind) + '/' + str(len(all_paths_labels)))
        sys.stdout.write(msg)
        sys.stdout.flush()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'raw_image': _bytes_feature(raw_img),
            'label': _int64_feature(label)}))
        writer.write(example.SerializeToString())
    except IndexError as ie:
        print str(ie)
        print path
        continue
    except IOError as ioe:
        print str(ioe)
        print path
        continue
    except Exception as e:
        print path
        print str(e)
        continue
    ind+=1
writer.close()

