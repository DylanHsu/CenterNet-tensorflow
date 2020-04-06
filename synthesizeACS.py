import os, sys
import SimpleITK as sitk
import math
import numpy as np

for subgroup in [1,2,3,4,5]:
  data_dir = '/data/deasy/DylanHsu/SRS_N401/subgroup%d/testing'%subgroup
  
  axial_path    = 'probability_centernet-7mm-mrctAxial-subgroup%d.nii.gz'%subgroup
  sagittal_path = 'probability_centernet-7mm-mrctSagittal-subgroup%d.nii.gz'%subgroup
  coronal_path  = 'probability_centernet-7mm-mrctCoronal-subgroup%d.nii.gz'%subgroup
  minimal_path  = 'probability_centernet-7mm-mrctACSMinimal-subgroup%d.nii.gz'%subgroup
  average_path  = 'probability_centernet-7mm-mrctACSAverage-subgroup%d.nii.gz'%subgroup
  
  writer = sitk.ImageFileWriter()
  writer.UseCompressionOn()
  minimumImageFilter = sitk.MinimumImageFilter()
  #addImageFilter = sitk.AddImageFilter()
  for case in os.listdir(data_dir):
    axial_pred    = sitk.ReadImage(os.path.join(data_dir,case,axial_path   ))
    sagittal_pred = sitk.ReadImage(os.path.join(data_dir,case,sagittal_path))
    coronal_pred  = sitk.ReadImage(os.path.join(data_dir,case,coronal_path ))
    average_pred = (axial_pred + sagittal_pred + coronal_pred) / (3.0)
    minimal_pred  = minimumImageFilter.Execute(axial_pred, sagittal_pred)
    minimal_pred  = minimumImageFilter.Execute(minimal_pred, coronal_pred)
    writer.SetFileName(os.path.join(data_dir,case,average_path))
    writer.Execute(average_pred)
    writer.SetFileName(os.path.join(data_dir,case,minimal_path))
    writer.Execute(minimal_pred)
