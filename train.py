from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from utils import tfrecord_voc_utils as voc_utils
import tensorflow as tf
import numpy as np
import CenterNet as net
import NiftiBBDataset
#import CenterNet as net
import os
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from skimage import io, transform
# from utils.voc_classname_encoder import classname_to_ids
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
lr = 0.01
batch_size = 15
buffer_size = 256
epochs = 16000
reduce_lr_epoch = []
config = {
    'mode': 'train',                                       # 'train', 'test'
    'input_size': 256,
    'data_format': 'channels_last',                        # 'channels_last' 'channels_first'
    'num_classes': 1,
    'weight_decay': 1e-4,
    'keep_prob': 0.5,                                      # not used
    'batch_size': batch_size,
    'keypoint_stride': 2.0,
    'score_threshold': 0.1,                                 
    'top_k_results_output': 64,                           


}

#data = os.listdir('./voc2007/')
#data = [os.path.join('./voc2007/', name) for name in data]

trainTransforms = [
    NiftiBBDataset.StatisticalNormalization(5.0, 5.0, nonzero_only=True, zero_floor=True),
    NiftiBBDataset.Padding((256,256,1)),
    NiftiBBDataset.RandomNoise(),
    NiftiBBDataset.RandomFlip(0.5, [True,True,True])
    ]

TrainDataset = NiftiBBDataset.NiftiBBDataset(
    data_dir = '/data/deasy/DylanHsu/N401_unstripped/subgroup1/training_noaug',
    image_filename = 'img.nii.gz',
    label_filename = 'label_smoothed.nii.gz',
    transforms=trainTransforms,
    train=True,
    )

trainDataset = TrainDataset.get_dataset()
trainDataset = trainDataset.apply(tf.contrib.data.unbatch())
trainDataset = trainDataset.repeat() 
trainDataset = trainDataset.shuffle(buffer_size)
trainDataset = trainDataset.batch(batch_size)
trainDataset = trainDataset.prefetch(5)
train_iterator = trainDataset.make_initializable_iterator()
train_initializer = train_iterator.make_initializer(trainDataset)
#self.train_initializer, self.train_iterator = self.train_generator
train_gen=[train_initializer, train_iterator]
#train_gen = voc_utils.get_generator(data,batch_size, buffer_size, image_augmentor_config)

#testTransforms = [NiftiBBDataset.Padding((256,256,1))]
#
#testDataset = NiftiBBDataset.NiftiBBDataset(
#    data_dir = '/data/deasy/DylanHsu/N401_unstripped/subgroup1/testing',
#    image_filename = 'img.nii.gz',
#    label_filename = 'label_smoothed.nii.gz',
#    transforms=testTransforms,
#    )

trainset_provider = {
    'data_shape': [256, 256, 1],
    'num_train': 954, # 3.3 lesions avg * 289 training cases
    'num_val': 0,                                         # not used
    'train_generator': train_gen,
    'val_generator': None                                 # not used
}
centernet = net.CenterNet(config, trainset_provider)
# centernet.load_weight('./centernet/test-8350')
# centernet.load_pretrained_weight('./centernet/test-8350')
for i in range(epochs):
    print('-'*25, 'epoch', i, '-'*25)
    if i in reduce_lr_epoch:
        lr = lr/10.
        print('reduce lr, lr=', lr, 'now')
    mean_loss = centernet.train_one_epoch(lr)
    print('>> mean loss', mean_loss)
    centernet.save_weight('latest', './centernet/test')            # 'latest', 'best
# img = io.imread('000026.jpg')
# img = transform.resize(img, [384,384])
# img = np.expand_dims(img, 0)
# result = centernet.test_one_image(img)
# id_to_clasname = {k:v for (v,k) in classname_to_ids.items()}
# scores = result[0]
# bbox = result[1]
# class_id = result[2]
# print(scores, bbox, class_id)
# plt.figure(1)
# plt.imshow(np.squeeze(img))
# axis = plt.gca()
# for i in range(len(scores)):
#     rect = patches.Rectangle((bbox[i][1],bbox[i][0]), bbox[i][3]-bbox[i][1],bbox[i][2]-bbox[i][0],linewidth=2,edgecolor='b',facecolor='none')
#     axis.add_patch(rect)
#     plt.text(bbox[i][1],bbox[i][0], id_to_clasname[class_id[i]]+str(' ')+str(scores[i]), color='red', fontsize=12)
# plt.show()
