import numpy as np
import matplotlib.pyplot as plt
import PIL
import glob
from PIL import Image
import os, sys
import tensorflow as tf

def make_tfrecord_rawdata(src_paths, labels, tfrecord_path):
    labels=labels.astype(np.int64)
    if os.path.exists(tfrecord_path):
        print tfrecord_path + 'is exists'
        return

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    all_paths_labels = zip(src_paths, labels)
    f = open(tfrecord_path.replace('.tfrecord', '_path.txt') , 'w')
    #e.g) ./normal.tfrecord--> normal_path.txt
    for ind, (path, label) in enumerate(all_paths_labels):
        try:
            f.write(path.split('/')[-1]+'\n')
            msg = '\r-Progress : {0}'.format(str(ind) + '/' + str(len(all_paths_labels)))
            sys.stdout.write(msg)
            sys.stdout.flush()
            if not path.endswith('.npy'):
                np_img = np.asarray(Image.open(path))
            else:
                np_img=np.load(path)
            height = np_img.shape[0]
            width = np_img.shape[1]
            raw_img = np_img.tostring()

            if __debug__ == False:
                print np.shape(np_img)
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
    writer.close()

def reconstruct_tfrecord_rawdata(tfrecord_path):

    print 'now Reconstruct Image Data please wait a second'
    reconstruct_image = []
    # caution record_iter is generator

    record_iter = tf.python_io.tf_record_iterator(path=tfrecord_path)
    n = len(list(record_iter))
    record_iter = tf.python_io.tf_record_iterator(path=tfrecord_path)

    print 'The Number of Data :', n
    ret_img_list = []
    ret_lab_list = []
    for i, str_record in enumerate(record_iter):
        msg = '\r -progress {0}/{1}'.format(i, n)
        sys.stdout.write(msg)
        sys.stdout.flush()

        example = tf.train.Example()
        example.ParseFromString(str_record)

        height = int(example.features.feature['height'].int64_list.value[0])
        width = int(example.features.feature['width'].int64_list.value[0])
        print height ,width
        raw_image = (example.features.feature['raw_image'].bytes_list.value[0])
        label = int(example.features.feature['label'].int64_list.value[0])
        image = np.fromstring(raw_image, dtype=np.uint8)
        image = image.reshape((height, width, -1))
        ret_img_list.append(image)
        ret_lab_list.append(label)
    ret_img = np.asarray(ret_img_list)
    ret_lab = np.asarray(ret_lab_list)
    return ret_img, ret_lab


def get_shuffled_batch(tfrecord_path, batch_size, resize):
    resize_height, resize_width = resize
    filename_queue = tf.train.string_input_producer([tfrecord_path], num_epochs=10)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       # Defaults are not specified since both keys are required.
                                       features={
                                           'height': tf.FixedLenFeature([], tf.int64),
                                           'width': tf.FixedLenFeature([], tf.int64),
                                           'raw_image': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64)
                                       })
    image = tf.decode_raw(features['raw_image'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    label = tf.cast(features['label'], tf.int32)
    image_shape = tf.pack([height, width, 3])
    image_size_const = tf.constant((resize_height, resize_width, 3), dtype=tf.int32)
    image = tf.reshape(image, image_shape)
    image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                   target_height=resize_height,
                                                   target_width=resize_width)
    images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=30, num_threads=3,
                                            min_after_dequeue=10)
    return images, labels


def read_one_example(tfrecord_path, batch_size, resize):
    filename_queue = tf.train.string_input_producer([tfrecord_path], num_epochs=10)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       # Defaults are not specified since both keys are required.
                                       features={
                                           'height': tf.FixedLenFeature([], tf.int64),
                                           'width': tf.FixedLenFeature([], tf.int64),
                                           'raw_image': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64)
                                       })
    image = tf.decode_raw(features['raw_image'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    label = tf.cast(features['label'], tf.int32)
    image_shape = tf.pack([height, width, 3])
    image = tf.reshape(image, image_shape)
    if not resize == None:
        resize_height, resize_width = resize
        image_size_const = tf.constant((resize_height, resize_width, 3), dtype=tf.int32)
        image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                       target_height=resize_height,
                                                       target_width=resize_width)
    # images  = tf.train.shuffle_batch([image ] , batch_size =batch_size  , capacity =30 ,num_threads=3 , min_after_dequeue=10)
    return image, label
