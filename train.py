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

tf.app.flags.DEFINE_integer('scan_axis',2,
    """Which dimension is dropped or 0.5D (default 2=Z)""")

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

assert FLAGS.scan_axis in [0,1,2]
# by 2D image conventions, height comes before width
if FLAGS.scan_axis == 0:
  padding_shape = (1, FLAGS.patch_size, FLAGS.patch_size)
  cropping_shape = (FLAGS.patch_layer, FLAGS.patch_size, FLAGS.patch_size)
elif FLAGS.scan_axis == 1:
  padding_shape = (FLAGS.patch_size, 1, FLAGS.patch_size)
  cropping_shape = (FLAGS.patch_size, FLAGS.patch_layer, FLAGS.patch_size)
elif FLAGS.scan_axis == 2:
  padding_shape = (FLAGS.patch_size, FLAGS.patch_size, 1)
  cropping_shape = (FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer)

trainTransforms = [
    NiftiDataset.StatisticalNormalization(0, 5.0, 5.0, nonzero_only=True, zero_floor=True), # Stat.norm. MR1
    NiftiDataset.ManualNormalization(1, 0, 100.), # use houndfield units [0,100] -> [0,255]
    NiftiDataset.Padding( padding_shape),
    NiftiDataset.RandomCrop( cropping_shape, 0, 1),
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
    bb_slice_axis=FLAGS.scan_axis,
    cpu_threads=4
    )

trainDataset = TrainDataset.get_dataset()
trainDataset = trainDataset.apply(tf.contrib.data.unbatch())
trainDataset = trainDataset.repeat() 
#trainDataset = trainDataset.shuffle(FLAGS.shuffle_buffer_size)
trainDataset = trainDataset.batch(FLAGS.batch_size)
trainDataset = trainDataset.prefetch(5)
train_iterator = trainDataset.make_initializable_iterator()
train_initializer = train_iterator.make_initializer(trainDataset)
train_gen=[train_initializer, train_iterator]

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

