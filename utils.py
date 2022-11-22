import random
import os
import monai
from monai.data import decollate_batch
import torch
import numpy as np
from skimage.measure import (
    label,
    regionprops
)
from skimage import img_as_int
from scipy.ndimage import binary_dilation
from anatomical_structs import anatomical_labels_dict

##-------------------------------------------------------------------------------------------------
## Function: get_files
##-------------------------------------------------------------------------------------------------

def get_files(base_path):
    '''
    This is a function
    '''
    mask_files = []
    image_files = []
    for (dirpath, dirnames, filenames) in os.walk(base_path):
        for file in filenames:
            if file.endswith("Mask.nii"):
                mask_files.append(os.path.join(dirpath, file))
            elif file.endswith(".nii"):
                image_files.append(os.path.join(dirpath, file))

    ## Grab a random set of validation images by sampling 20% of the total indices

    val_indices = random.sample(range(0, len(image_files)), int(0.2 * len(image_files)))

    image_train = [e for i, e in enumerate(image_files) if i not in val_indices]
    mask_train = [e for i, e in enumerate(mask_files) if i not in val_indices]
    image_val = [image_files[i] for i in val_indices]
    mask_val = [mask_files[i] for i in val_indices]

    print("Number of total image files: {}".format(len(image_files)))
    print("Number of total mask files: {}".format(len(mask_files)))
    print("Number of training image files: {}".format(len(image_train)))
    print("Number of training mask files: {}".format(len(mask_train)))
    print("Number of validation image files: {}".format(len(image_val)))
    print("Number of validation mask files: {}".format(len(mask_val)))

    return image_train, mask_train, image_val, mask_val

##-------------------------------------------------------------------------------------------------
## Function: split_images_by_task
##-------------------------------------------------------------------------------------------------

def split_images_by_task(image, label_dict):
    channel_splitter = monai.transforms.SplitDim(dim=1, keepdim=True)
    image_stack = channel_splitter(image)
    shape_channels = []
    point_channels = []
    shape_dict = []
    points_dict = []
    for i, label in enumerate(label_dict):
        if label["type"] == "point":
            point_channels.append(image_stack[i])
            points_dict.append(label)
        else:
            shape_channels.append(image_stack[i])
            shape_dict.append(label)
    return point_channels, shape_channels, points_dict, shape_dict


##-------------------------------------------------------------------------------------------------
## Function: remove_background
##-------------------------------------------------------------------------------------------------

def remove_background(mask, num_classes):
    seg_onehot = monai.networks.utils.one_hot(mask, num_classes=num_classes)
    seg_onehot = seg_onehot[:,1:,:,:]
    return seg_onehot

##-------------------------------------------------------------------------------------------------
## Function: 
##-------------------------------------------------------------------------------------------------

def dilate_points_lines(input_mask, BATCH_SIZE, NUM_CLASSES):
    for i in range(BATCH_SIZE):
        for j in range(NUM_CLASSES):
            if anatomical_labels_dict[j]["type"] in ["point", "line"]:
                dilated_mask = binary_dilation(
                    input=input_mask[i,j,:,:].numpy(),
                    iterations=2
                )
                dilated_mask = torch.from_numpy(dilated_mask)
                
                input_mask[i,j,:,:] = dilated_mask
    return input_mask

##-------------------------------------------------------------------------------------------------
## Function: get_centroids
##-------------------------------------------------------------------------------------------------

def get_centroids(image_stack):
    centroids = []
    for image in decollate_batch(image_stack):
        img_centroid = []
        for channel in image:
            channel_centroids = []
            label_img = label(img_as_int(channel.numpy()))
            label_regionprops = regionprops(np.squeeze(label_img))
            for i in range(2):
                try:
                    channel_centroids.append(label_regionprops[i].centroid)
                except IndexError:
                    channel_centroids.append((0.0,0.0))
            img_centroid.append(channel_centroids)
        centroids.append(img_centroid)
    
    return centroids

##-------------------------------------------------------------------------------------------------
## Function: get_centroid_tensor
##-------------------------------------------------------------------------------------------------

def get_centroid_tensor(centroids):
    '''
    This function turns a list of centroids into a tensor.
    '''
    centroid_stack = []
    for centroid in centroids:
        centroid = torch.tensor(centroid)
        centroid_stack.append(centroid)
    centroid_tensors = torch.stack(centroid_stack)
    return centroid_tensors.type(torch.FloatTensor)

##-------------------------------------------------------------------------------------------------
## Function: scale_centroids
##-------------------------------------------------------------------------------------------------

def scale_centroids(centroids, IMG_HEIGHT, IMG_WIDTH):
    '''
    This function scales the centroids to the 0-1 range
    in proportion to its location along the image axis
    '''
    scale_tensor = torch.tensor([IMG_HEIGHT, IMG_WIDTH])
    return torch.div(centroids, scale_tensor)