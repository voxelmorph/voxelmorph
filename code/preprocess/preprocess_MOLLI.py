import os
import shutil
import glob
import SimpleITK as sitk
import numpy as np
import argparse

np.random.seed(821)

def resize(img, new_size, interpolator):
    """Resize the image to the new size using the interpolator with SimpleITK

    Args:
        img (_type_): input SimpleITK image
        new_size (_type_): target size
        interpolator (_type_): Interpolator type

    Returns:
        _type_: Resized SimpleITK image
    """
    dimension = img.GetDimension()

    # Physical image size corresponds to the largest physical size in the training set, or any other arbitrary size.
    reference_physical_size = np.zeros(dimension)

    reference_physical_size[:] = [(sz - 1) * spc if sz * spc > mx else mx for sz, spc, mx in
                                  zip(img.GetSize(), img.GetSpacing(), reference_physical_size)]

    # Create the reference image with a zero origin, identity direction cosine matrix and dimension
    reference_origin = np.zeros(dimension)
    reference_direction = np.identity(dimension).flatten()
    reference_size = new_size
    reference_spacing = [phys_sz / (sz - 1) for sz, phys_sz in zip(reference_size, reference_physical_size)]

    reference_image = sitk.Image(reference_size, img.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    # Always use the TransformContinuousIndexToPhysicalPoint to compute an indexed point's physical coordinates as
    # this takes into account size, spacing and direction cosines. For the vast majority of images the direction
    # cosines are the identity matrix, but when this isn't the case simply multiplying the central index by the
    # spacing will not yield the correct coordinates resulting in a long debugging session.
    reference_center = np.array(
        reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))

    # Transform which maps from the reference_image to the current img with the translation mapping the image
    # origins to each other.
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(img.GetDirection())
    transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)
    # Modify the transformation to align the centers of the original and reference image instead of their origins.
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize()) / 2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))

    # centered_transform = sitk.Transform(transform)
    # centered_transform.AddTransform(centering_transform)

    centered_transform = sitk.CompositeTransform([transform, centering_transform])

    # Using the linear interpolator as these are intensity images, if there is a need to resample a ground truth
    # segmentation then the segmentation image should be resampled using the NearestNeighbor interpolator so that
    # no new labels are introduced.

    return sitk.Resample(img, reference_image, centered_transform, interpolator, 0.0)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='dataset name')
    parser.add_argument('--test_ratio', default=0.1, type=float, help='test_ratio')
    args = parser.parse_args()

    basepath = f"data/{args.dataset}"
    train_output = f"data/{args.dataset}_dataset/train"
    val_output = f"data/{args.dataset}_dataset/test"
    
    input_file_paths = []
    output_file_paths = []
    image_shapes = []
    if os.path.exists(train_output):
        shutil.rmtree(train_output)
    os.makedirs(train_output)
    if os.path.exists(val_output):
        shutil.rmtree(val_output)
    os.makedirs(val_output)

    files = glob.glob(f"{basepath}/*.nii.gz")
    # print(files)
    for file in files:
        input_file_paths.append(file)
        name = file.split("/")[-1][:-7]
        output_file_path = f"{train_output}/{name}"
        output_file_paths.append(output_file_path)

        image = sitk.ReadImage(file)
        image_shapes.append(image.GetSize())
    median_shape = (np.median(np.vstack(image_shapes), 0)).astype(np.int)
    max_shape = np.max(np.vstack(image_shapes), 0)
    min_shape = np.min(np.vstack(image_shapes), 0)

    print(f"Total file number {len(input_file_paths)}")
    print(f"The max shape is {max_shape}, median shape is {median_shape}, min_shape is {min_shape}")

    test_idx = np.random.choice(len(input_file_paths), int(len(input_file_paths)*args.test_ratio))

    min_idx = np.argmin(np.vstack(image_shapes), 0)
    min_dim_0 = image_shapes[min_idx[0]][0]
    min_dim_1 = image_shapes[min_idx[1]][1]
    print(f"Minimal shape {(min_dim_0, min_dim_1)}")

    output_train_filenames = []
    output_test_filenames = []

    for idx, file in enumerate(input_file_paths):
        image = sitk.ReadImage(file)
        original_shape = image_shapes[idx]
        new_shape = (min_dim_0, min_dim_1, original_shape[-1])
        resize_img = resize(image, new_shape, sitk.sitkLinear)
        
        resize_img_array = sitk.GetArrayFromImage(resize_img)
        tmp = (output_file_paths[idx]).split("/")[-1]
        
        if idx in test_idx:
            output_name = f"{val_output}/{tmp}.npy"
            np.save(output_name, resize_img_array)
            output_test_filenames.append(output_name)
        else:
            output_name = f"{train_output}/{tmp}.npy"
            np.save(output_name, resize_img_array)
            output_train_filenames.append(output_name)
    
    print(f"In training set, number of images {len(output_train_filenames)}")
    txt_string = "\n".join(output_train_filenames)
    with open(f'data/{args.dataset}_input.txt', "w") as f:
        f.write(txt_string) 

    print(f"In test set, number of images {len(output_test_filenames)}")
    txt_string = "\n".join(output_test_filenames)
    with open(f'data/{args.dataset}_test.txt', "w") as f:
        f.write(txt_string)     

    