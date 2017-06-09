import batch
import numpy as np
import glob
import random
normal_paths=glob.glob('/home/mediwhale/data/eye/cropped_image/normal/'+'*.npy')
cataract_paths=glob.glob('/home/mediwhale/data/eye/cropped_image/abnormal/cataract/'+'*.npy')
retina_paths=glob.glob('/home/mediwhale/data/eye/cropped_image/abnormal/retina/'+'*.npy')
glaucoma_paths=glob.glob('/home/mediwhale/data/eye/cropped_image/abnormal/glaucoma/'+'*.npy')
cata_glau_paths=glob.glob('/home/mediwhale/data/eye/cropped_image/abnormal/cataract_glaucoma/'+'*.npy')
retina_cata=glob.glob('/home/mediwhale/data/eye/cropped_image/abnormal/retina_cataract/'+'*.npy')
retina_glau=glob.glob('/home/mediwhale/data/eye/cropped_image/abnormal/retina_glaucoma/'+'*.npy')
abnormal_paths=[]
abnormal_paths.extend(cataract_paths)
abnormal_paths.extend(retina_paths)
abnormal_paths.extend(glaucoma_paths)
abnormal_paths.extend(cata_glau_paths)
abnormal_paths.extend(retina_cata)
abnormal_paths.extend(retina_glau)
print len(abnormal_paths)
normal_labels=np.ones([len(normal_paths)])
abnormal_labels=np.zeros([len(abnormal_paths)])

print len(normal_labels)

print normal_labels

print len(abnormal_labels)
print abnormal_labels
random.shuffle(normal_paths)
random.shuffle(abnormal_paths)
batch.make_tfrecord_rawdata(normal_paths[:5000] , normal_labels[:5000] ,'./normal_1.tfrecord')
batch.make_tfrecord_rawdata(abnormal_paths[:5000] , abnormal_labels[:5000],'./abnormal_1.tfrecord')
imgs,labs=batch.reconstruct_tfrecord_rawdata('./sample.tfrecord')
