from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from utils import tfrecord_voc_utils as voc_utils
import tensorflow as tf
import numpy as np
import CenterNet as net
import NiftiDataset
import SimpleITK as sitk
import os,sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

lr = 0.01
batch_size = 1
buffer_size = 256
epochs = 16000
reduce_lr_epoch = []
config = {
    'mode': 'test',                                       # 'train', 'test'
    'input_size': 256,
    'data_format': 'channels_last',                        # 'channels_last' 'channels_first'
    'num_classes': 1,
    'weight_decay': 1e-4,
    'keep_prob': 0.5,                                      # not used
    'batch_size': batch_size,
    #'keypoint_stride': 2.0,
    'score_threshold': 0.1,                                 
    'top_k_results_output': 16,                           


}
testTransforms = [
    NiftiDataset.StatisticalNormalization(5.0, 5.0, nonzero_only=True, zero_floor=True),
    NiftiDataset.Padding((256,256,1)),
    ]


TestDataset = NiftiDataset.NiftiDataset(
    #data_dir = '/data/deasy/DylanHsu/N401_unstripped/subgroup1/training_noaug',
    data_dir = '/data/deasy/DylanHsu/N401_unstripped/subgroup1/testing',
    image_filename = 'img.nii.gz',
    label_filename = 'label_smoothed.nii.gz',
    transforms=testTransforms,
    train=True,
    bounding_boxes=True
    )

testDataset = TestDataset.get_dataset()
testDataset = testDataset.apply(tf.contrib.data.unbatch())
testDataset = testDataset.batch(1)
test_iterator = testDataset.make_initializable_iterator()
test_initializer = test_iterator.make_initializer(testDataset)

centernet = net.CenterNet(config, None)
centernet.load_weight('./centernet/test-49203')
centernet.sess.run(test_initializer)

[image_np, boxlist] = centernet.sess.run(test_iterator.get_next())
result = centernet.test_one_image(image_np)

# id_to_clasname = {k:v for (v,k) in classname_to_ids.items()}
scores = result[0]
bbox = result[1]
class_id = result[2]
print(scores, bbox, class_id)
print(boxlist[:,boxlist[0,:,0]>=0,:])
# plt.figure(1)
# plt.imshow(np.squeeze(img))
# axis = plt.gca()
# for i in range(len(scores)):
#     rect = patches.Rectangle((bbox[i][1],bbox[i][0]), bbox[i][3]-bbox[i][1],bbox[i][2]-bbox[i][0],linewidth=2,edgecolor='b',facecolor='none')
#     axis.add_patch(rect)
#     plt.text(bbox[i][1],bbox[i][0], id_to_clasname[class_id[i]]+str(' ')+str(scores[i]), color='red', fontsize=12)
# plt.show()
