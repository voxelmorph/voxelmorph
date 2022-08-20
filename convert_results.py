import os
import glob
import SimpleITK as sitk
import numpy as np
from array2gif import write_gif


path = '/Users/mona/GitClone/voxelmorph-test/data/data_moved'
out = '/Users/mona/GitClone/voxelmorph-test/data/data_moved_gif'
files = glob.glob(f"{path}/*.nii")

for file in files:
    name = (file.split("/")[-1]).split(".")[0]
    image = sitk.ReadImage(file)
    data = sitk.GetArrayFromImage(image)*255
    # data = np.load(file)
    # data = np.transpose(data, (2,1,0))
    # data = data * 255
    # data = sitk.GetArrayFromImage(image)*255
    # sitk.WriteImage(sitk.GetImageFromArray(data), f"{out}/{name}.nii")
    data_list = [np.stack((data[:,:,i],data[:,:,i],data[:,:,i])) for i in range(data.shape[-1])]
    write_gif(data_list, f"{out}/{name}.gif")
