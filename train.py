from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from utils import tfrecord_voc_utils as voc_utils
import tensorflow as tf
import numpy as np
import CenterNet as net
import NiftiDataset
#import CenterNet as net
import os

# tensorflow app flags
FLAGS = tf.app.flags.FLAGS
# Hack for Tensorflow 1.14, log_dir already defined in Abseil dependency
for name in list(FLAGS):
    if name=='log_dir':
      delattr(FLAGS,name)

tf.app.flags.DEFINE_string('data_dir', './data',
    """Directory of stored data.""")
tf.app.flags.DEFINE_string('image_filenames','mr1.nii.gz,ct.nii.gz',
    """Image filename""")
tf.app.flags.DEFINE_string('label_filename','label_smoothed.nii.gz',
    """Label filename""")
tf.app.flags.DEFINE_integer('batch_size',15,
    """Size of batch""")               
tf.app.flags.DEFINE_integer('patch_size',256,
    """Size of a data patch""")
tf.app.flags.DEFINE_integer('patch_layer',7,
    """Number of layers in data patch""")
tf.app.flags.DEFINE_integer('epochs',500,
    """Number of epochs for training""")
tf.app.flags.DEFINE_string('log_dir', './tmp/log',
    """Directory where to write training and testing event logs """)
tf.app.flags.DEFINE_float('init_learning_rate',1e-2,
    """Initial learning rate""")
tf.app.flags.DEFINE_string('checkpoint_dir', './tmp/ckpt',
    """Directory where to write checkpoint""")
tf.app.flags.DEFINE_bool('restore_training',True,
    """Restore training from last checkpoint""")
tf.app.flags.DEFINE_integer('shuffle_buffer_size',256,
    """Number of elements used in shuffle buffer""")
#tf.app.flags.DEFINE_string('optimizer','sgd',
#    """Optimization method (sgd, adam, momentum, nesterov_momentum)""")
#tf.app.flags.DEFINE_float('momentum',0.5,
#    """Momentum used in optimization""")
tf.app.flags.DEFINE_boolean('is_batch_job',False,
    """Disable some features if this is a batch job""")
tf.app.flags.DEFINE_string('batch_job_name','',
    """Name the batch job so the checkpoints and tensorboard output are identifiable.""")
tf.app.flags.DEFINE_float('max_ram',15.5,
    """Maximum amount of RAM usable by the CPU in GB.""")
tf.app.flags.DEFINE_float('dropout_keepprob',0.5,
    """probability to randomly keep a parameter for dropout (default 1 = no dropout)""")
tf.app.flags.DEFINE_float('l2_weight',1e-4,
    """Weight for L2 regularization (should be order of 0.0001)""")

tf.app.flags.DEFINE_boolean('use_weighted_dice',False,
    """Use weighted dice""")
tf.app.flags.DEFINE_float('weighted_dice_kD',1.0,
    """Weight for the Diameter term in the Weighted-Dice paradigm.""")
tf.app.flags.DEFINE_float('weighted_dice_kI',5.0,
    """Weight for the Intensity term in the Weighted-Dice paradigm.""")

image_filenames_list = [i.strip() for i in FLAGS.image_filenames.split(',')]
sequences = len(image_filenames_list)
channels = FLAGS.patch_layer * sequences

config = {
    'mode'                : 'train',                                       # 'train', 'test'
    'input_size'          : FLAGS.patch_size,
    'num_channels'        : channels,
    'data_format'         : 'channels_last',                        # 'channels_last' 'channels_first'
    'num_classes'         : 1,
    'weight_decay'        : FLAGS.l2_weight,
    'keep_prob'           : FLAGS.dropout_keepprob,                    # not used
    'batch_size'          : FLAGS.batch_size,
    'score_threshold'     : 0,                                 
    'top_k_results_output': 0,                           
}

# To do: add logic to do padding/cropping based on bb_slice_axis
trainTransforms = [
    NiftiDataset.StatisticalNormalization(0, 5.0, 5.0, nonzero_only=True, zero_floor=True), # Stat.norm. MR1
    NiftiDataset.Padding( (FLAGS.patch_size,FLAGS.patch_size,1) ),
    NiftiDataset.RandomCrop( (FLAGS.patch_size,FLAGS.patch_size,FLAGS.patch_layer), 0, 1),
    NiftiDataset.RandomNoise(0, 0.1), # MR1 noise sigma 0.1
    NiftiDataset.RandomFlip(0.5, [True,True,True])
    ]

TrainDataset = NiftiDataset.NiftiDataset(
    #data_dir = '/data/deasy/DylanHsu/SRS_N401/subgroup1/training_noaug/',
    data_dir = FLAGS.data_dir,
    image_filenames = FLAGS.image_filenames,
    label_filename = FLAGS.label_filename,
    transforms=trainTransforms,
    train=True,
    bounding_boxes=True,
    bb_slice_axis=2,
    bb_slices=1,
    cpu_threads=16
    )

trainDataset = TrainDataset.get_dataset()
trainDataset = trainDataset.apply(tf.contrib.data.unbatch())
trainDataset = trainDataset.repeat() 
trainDataset = trainDataset.shuffle(FLAGS.shuffle_buffer_size)
trainDataset = trainDataset.batch(FLAGS.batch_size)
trainDataset = trainDataset.prefetch(5)
train_iterator = trainDataset.make_initializable_iterator()
train_initializer = train_iterator.make_initializer(trainDataset)
#self.train_initializer, self.train_iterator = self.train_generator
train_gen=[train_initializer, train_iterator]
#train_gen = voc_utils.get_generator(data,batch_size, buffer_size, image_augmentor_config)

#testTransforms = [NiftiDataset.Padding((width,width,1))]
#
#testDataset = NiftiDataset.NiftiDataset(
#    data_dir = '/data/deasy/DylanHsu/N401_unstripped/subgroup1/testing',
#    image_filename = 'img.nii.gz',
#    label_filename = 'label_smoothed.nii.gz',
#    transforms=testTransforms,
#    )

trainset_provider = {
  #'data_shape': [width, width, depth],
  'num_train': 1000, # 3.3 lesions avg * 289 training cases ~ 1000
  'num_val': 0,                                         # not used
  'train_generator': train_gen,
  'val_generator': None                                 # not used
}
centernet = net.CenterNet(config, trainset_provider)
# centernet.load_weight('./centernet/test-8350')
# centernet.load_pretrained_weight('./centernet/test-8350')

checkpoint_slug = "checkpoint"
if FLAGS.is_batch_job and FLAGS.batch_job_name is not '':
  checkpoint_slug = checkpoint_slug + "_" + FLAGS.batch_job_name
checkpoint_prefix = os.path.join(FLAGS.checkpoint_dir, checkpoint_slug)

#if FLAGS.restore_training:
#    # check if checkpoint exists
#    if os.path.exists(checkpoint_prefix+"_latest"):
#        print("{}: Last checkpoint found at {}, loading...".format(datetime.datetime.now(),FLAGS.checkpoint_dir))
#        latest_checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir,latest_filename=latest_filename)
#        saver.restore(sess, latest_checkpoint_path)

lr = FLAGS.init_learning_rate
reduce_lr_epoch = []

for i in range(FLAGS.epochs):
  print('-'*25, 'epoch', i, '-'*25)
  if i in reduce_lr_epoch:
      lr = lr/10.
      print('reduce lr, lr=', lr, 'now')
  mean_loss = centernet.train_one_epoch(lr)
  print('>> mean loss', mean_loss)
  centernet.save_weight('latest', checkpoint_prefix)            # 'latest', 'best
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
