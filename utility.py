# IMPORTING LIBRARIES
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import rotate
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from skimage.morphology import label, binary_dilation
from skimage import filters
from skimage.transform import resize
from skimage.segmentation import relabel_sequential
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import cv2

# BUCKETING IMAGES WITH SAME RESOLUTION TOGETHER
def Gen_Buckets(Images, Masks, return_sizes = False):
    Shape = [[image.shape[0], image.shape[1]] for image in Images]
    Unique_Shapes = np.unique(Shape, axis=0)

    Image_Buckets = []
    Mask_Buckets = []
    for shape in Unique_Shapes:
        image_bucket = []
        mask_bucket = []
        for image, mask in zip(Images, Masks):
            if (image.shape[0] ==  shape[0]) &  (image.shape[1] == shape[1]):
                image_bucket.append(image)
                mask_bucket.append(mask)
        Image_Buckets.append(np.array(image_bucket))
        Mask_Buckets.append(np.array(mask_bucket))
    if return_sizes:
        return Image_Buckets, Mask_Buckets, Unique_Shapes
    else:
        return Image_Buckets, Mask_Buckets
    
# Normalizing Images betweeen 0 and 1.
def Normalize_Images(Images, divide_by = 255): 
    if divide_by == 'max':
        return [ image/image.max() for image in Images]
    else:
        return [ image/divide_by for image in Images]

    
# Augmentated Image Generator
def Augment_Images(Images, Masks, fliplr = True, flipud = True, Rot = None, 
                   Rot_Mode = 'edge', Rot90 = True, return_numpy=True):
    # Rot  = Does small angular rotations but keeps the dimensions same.
    # Rot90= Rotates image by multiples of 90 degree and changes the dimensions
    Aug_Images = []
    Aug_Mask =   []
    for image, mask in zip(Images, Masks):
        aug_img = image
        aug_mask = mask
        if fliplr:
            if np.random.uniform(low=0, high=1) > 0.5:
                aug_img = np.fliplr(aug_img)
                aug_mask = np.fliplr(aug_mask)
        if flipud:
            if np.random.uniform(low=0, high=1) > 0.5:
                aug_img = np.flipud(aug_img)
                aug_mask = np.flipud(aug_mask)
        if Rot90:
            if np.random.uniform(low=0, high=1)>0.5:
                k = np.random.randint(low=0, high=4)
                aug_img = np.rot90(aug_img, k = k, axes=(0,1))
                aug_mask = np.rot90(aug_mask, k = k, axes=(0,1))
        if Rot != None:
            if np.random.uniform(low=0, high=1) > 0.5:
                angle = np.random.uniform(low=0, high = Rot)
                aug_img = rotate(aug_img, angle, mode = Rot_Mode)
                aug_mask = rotate(aug_mask, angle, mode = Rot_Mode)
        Aug_Images.append(aug_img)
        Aug_Mask.append(aug_mask)
        
    if return_numpy:
        return np.array(Aug_Images), np.array(Aug_Mask)
    return Aug_Images, Aug_Mask

# Rescaling the Images
def Rescale_Images(Images, Masks, size=(256,256), order_images=3,
                   order_masks = 0, mode='constant'):
    Images_Resized = []; Masks_Resized = [];
    if Masks == None:
        for image in Images:
            img = resize(image, size, order_images, mode, preserve_range = True )
            Images_Resized.append(img)
        return np.array(Images_Resized)
    else:
        for image, mask in zip(Images, Masks):
            img = resize(image, size, order_images, mode, preserve_range = True )
            msk = resize(mask,  size, order_masks, mode, preserve_range = True )

            Images_Resized.append(img)
            Masks_Resized.append(msk)
        
        return np.array(Images_Resized), np.array(Masks_Resized)

def Load_Training_Data(Train_Path, return_mask_report=True):
    Images = []
    Masks = []     # Stores masks by adding them
    Masks_Obj = [] # Stores masks as is, donot add
    Number_Of_Masks = []
    for path in tqdm(glob(Train_Path)):
        image_path = glob(path + '\\images\\*.png')
        assert len(image_path) == 1
        Images.append(imread(image_path[0])[:,:,:3])    
        masks = glob(path + '\\masks\\*.png')
        mask = []
        for m in masks:
            mask.append(imread(m))
        Number_Of_Masks.append(len(mask))
        Masks_Obj.append(np.array(mask))
        mask = np.sum(mask, axis=0)
        mask = np.expand_dims(mask, axis=-1)
        Masks.append(mask)
    if return_mask_report:
        return Images, Masks,Masks_Obj, np.array(Number_Of_Masks)
    else:
        return Images, Masks,Masks_Obj
    
def Load_Test_Data(Test_Path, return_testid=True):
    Test_Images = []
    Test_Id = []
    for path in tqdm(glob(Test_Path)):
        image_path = glob(path + '\\images\\*.png')
        test_id = glob(path + '\\images\\*.png')[0].split('\\')[-1][:-4]
        assert len(image_path) == 1
        Test_Images.append(imread(image_path[0])[:,:,:3])    
        Test_Id.append(test_id)
    if return_testid:
        return Test_Images, Test_Id
    else:
        return Test_Images
    
def KMeans_Clustering(Images, N_Clusters = [2], return_np = True, n_jobs = -1 ):
    Cluster = []
    for image in tqdm(Images):
        cluster = []
        for n_clusters in N_Clusters:
            red, green, blue = image[:,:,0], image[:,:,1],image[:,:,2]
            original_shape = red.shape # so we can reshape the labels later
            samples = np.column_stack([red.flatten(), green.flatten(), blue.flatten()])
            clf = KMeans(n_clusters=n_clusters, n_jobs = n_jobs)
            labels = clf.fit_predict(samples).reshape(original_shape)
            values, counts = np.unique(labels, return_counts =True)

            argsort = counts.argsort()
            argsort = np.flip(argsort, axis=0)
            values_sorted = values[argsort]
            final_labels = np.zeros(labels.shape)
            i = 0
            for value in values_sorted:
                final_labels[labels==value] = i
                i = i+1
            cluster.append(final_labels)
        cluster = np.array(cluster)
        cluster = np.expand_dims(cluster, axis=-1)
        Cluster.append(cluster)
    if return_np:
        return np.array(Cluster)
    else:
        return Cluster


def WaterShed(img):
    # img must be cv2.imread object
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
    ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]
    return markers
