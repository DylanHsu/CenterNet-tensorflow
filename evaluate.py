from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import NiftiDataset
import os
import datetime
import SimpleITK as sitk
import math
import numpy as np
import CenterNet 
from tqdm import tqdm


# tensorflow app flags
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir','./data/evaluate',
    """Directory of evaluation data""")
tf.app.flags.DEFINE_string('image_filenames','mr1.nii.gz,ct.nii.gz',
    """Image filename""")
tf.app.flags.DEFINE_string('label_filename','label_smoothed.nii.gz',
    """Label filename""")
tf.app.flags.DEFINE_string('checkpoint_path','./tmp/ckpt/checkpoint-2240',
    """Directory of saved checkpoints""")
tf.app.flags.DEFINE_integer('patch_size',256,
    """Size of a data patch""")
tf.app.flags.DEFINE_integer('patch_layer',7,
    """Number of layers in data patch""")
tf.app.flags.DEFINE_integer('stride_inplane', 256, 
    """Stride size in 2D plane""")
tf.app.flags.DEFINE_integer('stride_layer',1, 
    """Stride size in layer direction""")
tf.app.flags.DEFINE_integer('batch_size',1,
    """Setting batch size (currently only accept 1)""")
tf.app.flags.DEFINE_string('suffix','',
    """Suffix for saving""")
tf.app.flags.DEFINE_string('case','',
    """Specific case to evaluate""")
tf.app.flags.DEFINE_boolean('is_batch',False,
    """Disable progress bar if this is a batch job""")
tf.app.flags.DEFINE_boolean('use_cpu',False,
    """Decide if you want to use a GPU or CPUs""")
tf.app.flags.DEFINE_integer('scan_axis',2,
    """Which dimension is dropped or 0.5D (default 2=Z)""")

def get_one_batch(image_3d, batch):
  image_batch = []
  for patch in batch:
    image_patch_3d = image_3d[patch[0]:patch[1], patch[2]:patch[3], patch[4]:patch[5], :]
    # Make an arbitrary choice on how to stack the Z values and sequences
    # Last dimension will be (z1c1,z2c1,...,z1c2,z2c2)
    image_patch_2p5d = image_patch_3d.reshape( (image_patch_3d.shape[0], image_patch_3d.shape[1], image_patch_3d.shape[2]*image_patch_3d.shape[3]) , order='F')
    image_batch.append(image_patch_2p5d)
  image_batch = np.asarray(image_batch)
  return image_batch

def evaluate():
  if not FLAGS.use_cpu:
    # select gpu devices
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" # e.g. "0,1,2", "0,2" 
  
  image_filenames_list = [i.strip() for i in FLAGS.image_filenames.split(',')]
  sequences = len(image_filenames_list)
  channels = FLAGS.patch_layer * sequences
  assert FLAGS.scan_axis in [0,1,2]
  if FLAGS.scan_axis == 0:
    padding_shape = (1, FLAGS.patch_size, FLAGS.patch_size)
    cropping_shape = (FLAGS.patch_layer, FLAGS.patch_size, FLAGS.patch_size)
  elif FLAGS.scan_axis == 1:
    padding_shape = (FLAGS.patch_size, 1, FLAGS.patch_size)
    cropping_shape = (FLAGS.patch_size, FLAGS.patch_layer, FLAGS.patch_size)
  elif FLAGS.scan_axis == 2:
    padding_shape = (FLAGS.patch_size, FLAGS.patch_size, 1)
    cropping_shape = (FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer)
 
  config = {
      'mode'                : 'test',                                       # 'train', 'test'
      'input_size'          : FLAGS.patch_size,
      'num_channels'        : channels,
      'data_format'         : 'channels_last',                        # 'channels_last' 'channels_first'
      'num_classes'         : 1,
      'weight_decay'        : 0,
      'keep_prob'           : 0,
      'batch_size'          : FLAGS.batch_size,
      'score_threshold'     : 0.01,                                 
      'top_k_results_output': 200,                           
  }
  
  centernet = CenterNet.CenterNet(config, None)
  centernet.load_weight(FLAGS.checkpoint_path)

  for case in os.listdir(FLAGS.data_dir):
    if FLAGS.case != '' and FLAGS.case != case:
      continue
    image_paths = []
    images = []
    images_tfm = [] 
    for i in range(sequences):
      image_path = os.path.join(FLAGS.data_dir, case, image_filenames_list[i])
      image_paths.append(image_path)
      image = sitk.ReadImage(image_path)
      images.append(image)
      images_tfm.append(image) # maintaining two separate lists just to be safe, I should get better at python
    
    true_label = sitk.ReadImage(os.path.join(FLAGS.data_dir, case, FLAGS.label_filename))
    label_tfm = sitk.Image(true_label.GetSize(), sitk.sitkUInt32)
    label_tfm.SetOrigin(true_label.GetOrigin())
    label_tfm.SetDirection(true_label.GetDirection())
    label_tfm.SetSpacing(true_label.GetSpacing())
    
    transforms = [
      NiftiDataset.StatisticalNormalization(0,5.0,5.0,nonzero_only=True,zero_floor=True),
      NiftiDataset.Padding( padding_shape )
    ]
    
    sample = {'image': images_tfm, 'label': label_tfm}
    true_sample = {'image': [], 'label': true_label}
    for transform in transforms:
        sample = transform(sample)
        true_sample = transform(true_sample)
    images_tfm, label_tfm = sample['image'], sample['label']
    true_label_tfm = true_sample['label']
    
    softmax_tfm = sitk.Image( images_tfm[0].GetSize(), sitk.sitkFloat32)
    softmax_tfm.SetOrigin   ( images_tfm[0].GetOrigin()    )
    softmax_tfm.SetDirection( images_tfm[0].GetDirection() )
    softmax_tfm.SetSpacing  ( images_tfm[0].GetSpacing()   )
    softmax_np = sitk.GetArrayFromImage(softmax_tfm)
    softmax_np = np.asarray(softmax_np, np.float32)
    softmax_np = np.transpose(softmax_np,(1,2,0)) # (Z,Y,X) -> (Y,X,Z)
    
    # a weighting matrix will be used for averaging the overlapped region
    weight_np = np.zeros(softmax_np.shape)
    
    image_np = [] # list of (Z,Y,X) arrays
    for volume in images_tfm:
      image_np += [sitk.GetArrayFromImage(volume)]
    image_3d = np.asarray(image_np, np.float32) # (T,Z,Y,X) array
    if FLAGS.scan_axis == 0:
      transpose = (1,2,3,0) # (T,Z,Y,X) -> (Z,Y,X,T)
    elif FLAGS.scan_axis == 1:
      transpose = (1,3,2,0) # (T,Z,Y,X) -> (Z,X,Y,T)
    elif FLAGS.scan_axis == 2:
      transpose = (2,3,1,0) # (T,Z,Y,X) -> (Y,X,Z,T)
    image_3d = np.transpose(image_3d, transpose)
 
    # prepare image batch indices
    inum = int(math.ceil((image_3d.shape[0]-FLAGS.patch_size)/float(FLAGS.stride_inplane))) + 1 
    jnum = int(math.ceil((image_3d.shape[1]-FLAGS.patch_size)/float(FLAGS.stride_inplane))) + 1
    knum = int(math.ceil((image_3d.shape[2]-FLAGS.patch_layer)/float(FLAGS.stride_layer))) + 1
    patch_total = 0
    ijk_patch_indices = []
    ijk_patch_indicies_tmp = []
    
    assert (image_3d.shape[0]>=FLAGS.patch_size) and (image_3d.shape[1]>=FLAGS.patch_size) and (image_3d.shape[2]>=FLAGS.patch_layer)
    
    for i in range(inum):
      for j in range(jnum):
        for k in range (knum):
          if patch_total % FLAGS.batch_size == 0:
            ijk_patch_indicies_tmp = []
    
          istart = i * FLAGS.stride_inplane
          if istart + FLAGS.patch_size > image_3d.shape[0]: #for last patch
            istart = image_3d.shape[0] - FLAGS.patch_size 
          iend = istart + FLAGS.patch_size
    
          jstart = j * FLAGS.stride_inplane
          if jstart + FLAGS.patch_size > image_3d.shape[1]: #for last patch
            jstart = image_3d.shape[1] - FLAGS.patch_size 
          jend = jstart + FLAGS.patch_size
    
          kstart = k * FLAGS.stride_layer
          if kstart + FLAGS.patch_layer > image_3d.shape[2]: #for last patch
            kstart = image_3d.shape[2] - FLAGS.patch_layer 
          kend = kstart + FLAGS.patch_layer
    
          ijk_patch_indicies_tmp.append([istart, iend, jstart, jend, kstart, kend])
    
          if patch_total % FLAGS.batch_size == 0:
            ijk_patch_indices.append(ijk_patch_indicies_tmp)
    
          patch_total += 1
    
    if not FLAGS.is_batch:
      the_iterations = tqdm(range(len(ijk_patch_indices)))
    else:
      the_iterations = range(len(ijk_patch_indices))
    
    for i in the_iterations:
      # Serve the network a 2.5D slice and evaluate it
      batch = ijk_patch_indices[i]
      batch_image = get_one_batch(image_3d, batch)
      result = centernet.test_one_image(batch_image)
      scores = result[0]
      pred_bboxes = result[1]
      
      # Set up the weighting matrix for overlapping evaluations
      istart = batch[0][0] # Y
      iend   = batch[0][1]
      jstart = batch[0][2] # X
      jend   = batch[0][3]
      kstart = batch[0][4] # Z
      kend   = batch[0][5]
      
      weight_np[istart:iend,jstart:jend,kstart:kend] += 1.0
      for p, pred_bbox in enumerate(pred_bboxes):
        y1,x1,y2,x2 = pred_bbox
        i1 = int(round(y1))
        i2 = int(round(y2))
        j1 = int(round(x1))
        j2 = int(round(x2))
        # probability map is (Y,X,Z) dimension 
        #print('adding %.2f to [%d:%d,%d:%d,:]'%(scores[p],i1,i2,j1,j2))
        softmax_np[i1:i2,j1:j2,kstart:kend] += scores[p]
    softmax_np = softmax_np / np.float32(weight_np)
    if FLAGS.scan_axis == 0:
      inverse_transpose = (0,1,2) # (Z,Y,X) -> (Z,Y,X) special case
    elif FLAGS.scan_axis == 1:
      inverse_transpose = (0,2,1) # (Z,X,Y) -> (Z,Y,X)
    elif FLAGS.scan_axis == 2:
      inverse_transpose = (2,0,1) # (Y,X,Z) -> (Z,Y,X) 
    softmax_np = np.transpose(softmax_np, inverse_transpose) 
    softmax_tfm = sitk.GetImageFromArray(softmax_np)
    softmax_tfm.SetOrigin   ( images_tfm[0].GetOrigin()    )
    softmax_tfm.SetDirection( images_tfm[0].GetDirection() )
    softmax_tfm.SetSpacing  ( images_tfm[0].GetSpacing()   )
    # Get back to the original image size after padding
    roiFilter = sitk.RegionOfInterestImageFilter()
    start_i,start_j,start_k = images_tfm[0].TransformPhysicalPointToIndex(images[0].TransformIndexToPhysicalPoint((0,0,0))) # XYZ indices
    # assume same spacing, direction, origin?
    roiFilter.SetSize(images[0].GetSize())
    roiFilter.SetIndex([start_i,start_j,start_k])
    softmax = roiFilter.Execute(softmax_tfm)
    writer = sitk.ImageFileWriter()
    writer.UseCompressionOn()
    writer.SetFileName(os.path.join(FLAGS.data_dir, case, 'probability_centernet%s.nii.gz'%FLAGS.suffix))
    writer.Execute(softmax)
    #image_paths = [os.path.join(FLAGS.data_dir,case,image_filename) for image_filename in image_filenames_list]
    #true_label_path = os.path.join(FLAGS.data_dir,case,FLAGS.label_filename)
    #
    #[image_np, boxlist] = centernet.sess.run(test_iterator.get_next())
    #
    #class_id = result[2]
    #print(scores, bbox, class_id)
    #print(boxlist[:,boxlist[0,:,0]>=0,:])
def main(argv=None):
    evaluate()

if __name__=='__main__':
    tf.app.run()
