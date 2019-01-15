import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize, rotate
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# function to load training data from file path
def load_train(train_path, size=None, return_mask_report=True):
    images      = []
    masks       = []               # stores masks by adding them
    masks_obj   = []               # stores masks as is, donot add
    no_of_masks = []
    for path in tqdm(glob(train_path+"/*"),desc="loading training images"):
        # loading images
        path_i = glob(path + '/images/*.png')
        assert len(path_i) == 1
        img = imread(path_i[0])[:,:,:3]/255
        if size is not None:
            img = resize(img,size,order=3,mode="constant",preserve_range=True,anti_aliasing=False)
        images.append(img)
        # loading masks
        path_m = glob(path + '/masks/*.png')
        mask = []
        for p in path_m:
            img = imread(p)/255
            if size is not None:
                img = resize(img,size,order=0,mode="constant",preserve_range=True,anti_aliasing=False)            
            mask.append(img)
        no_of_masks.append(len(mask))
        masks_obj.append(np.array(mask))
        mask = np.sum(mask, axis=0)
        mask = np.expand_dims(mask, axis=-1)
        masks.append(mask)
    # removing outliers identified in data exploration notebook
    del_idx = [332, 36] # Also 36 because no such image is present in Test Set.
    for idx in del_idx:
        del images[idx]
        del masks[idx]
        del masks_obj[idx]
#    # converting to numpy array if all images have same sizes
    if size is not None:
        masks  = np.array(masks)
        images = np.array(images)
        
    if return_mask_report:
        return images, masks,masks_obj, np.array(no_of_masks)
    else:
        return images, masks,masks_obj
    
    
    
# a function to load test data
def load_test(test_path, return_testid=True):
    Test_Images = []
    Test_Id = []
    for path in tqdm(glob(test_path)):
        image_path = glob(path + '/images/*.png')
        test_id = glob(path + '/images/*.png')[0].split('/')[-1][:-4]
        assert len(image_path) == 1
        Test_Images.append(imread(image_path[0])[:,:,:3])    
        Test_Id.append(test_id)
    if return_testid:
        return Test_Images, Test_Id
    else:
        return Test_Images


# function to generate augmented images given a batch of images and masks
def augmenter(images, masks, fliplr, flipud, rot, rot_mode, rot90):
    '''
    rot   :  does small angular rotations but keeps the dimensions same.
    rot90 :  rotates image by multiples of 90 degree and changes the dimensions
    '''
    aug_images = []
    aug_masks  =   []
    for image, mask in zip(images, masks):
        aug_img = image
        aug_mask = mask
        if fliplr: # randomly fliplr
            if np.random.uniform(low=0, high=1) > 0.5:
                aug_img = np.fliplr(aug_img)
                aug_mask = np.fliplr(aug_mask)
        if flipud: # randomly flipud
            if np.random.uniform(low=0, high=1) > 0.5:
                aug_img = np.flipud(aug_img)
                aug_mask = np.flipud(aug_mask)
        if rot90: # randomly rotate 90 degrees
            if np.random.uniform(low=0, high=1) > 0.5:
                k = np.random.randint(low=0, high=4)
                aug_img = np.rot90(aug_img, k = k, axes=(0,1))
                aug_mask = np.rot90(aug_mask, k = k, axes=(0,1))
        if rot != None: # randomly rotate if True
            if np.random.uniform(low=0, high=1) > 0.5:
                angle = np.random.uniform(low=0, high = rot)
                aug_img = rotate(aug_img, angle, mode = rot_mode, preserve_range=True)
                aug_mask = rotate(aug_mask, angle, mode = rot_mode, preserve_range=True)
        aug_images.append(aug_img)
        aug_masks.append(aug_mask)        
    return np.array(aug_images), np.array(aug_masks)

# perform k-means segmentation on images
def kmeans_segmentor(images, n_clusters, return_np = True, n_jobs = -1, reduce_size = None, ):
    if reduce_size is not None:
        s = images.shape[1] # orig size 
        images = np.array([resize(img,(reduce_size[0],reduce_size[1],3),mode="constant",preserve_range=True,anti_aliasing=True) for img in images])
    Cluster = []
    for image in tqdm(images, desc="generating cluster maps"):
        cluster = []
        for n in n_clusters:
            red, green, blue = image[:,:,0], image[:,:,1],image[:,:,2]
            original_shape = red.shape # so we can reshape the labels later
            samples = np.column_stack([red.flatten(), green.flatten(), blue.flatten()])
            clf = KMeans(n_clusters=n, n_jobs = n_jobs)
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
            final_labels = final_labels / final_labels.max() # new line added
            cluster.append(final_labels)
        cluster = np.array(cluster).transpose([1,2,0])
#        cluster = np.expand_dims(cluster, axis=-1)
        Cluster.append(cluster)

    if reduce_size is not None:
        Cluster = [resize(c, (s,s), mode="constant",order=0,anti_aliasing=False,preserve_range=True) for c in Cluster]
    if return_np:
        return np.array(Cluster)
    else:
        return Cluster


class DataLoader():
    '''
    A class to load images and masks for train and test folders. Capable of 
    generating (augmented) batches for training.
    '''
    def __init__(self, train_folder, test_folder, size=None):
        self.train_folder  = train_folder
        self.test_folder   = test_folder
        self.size          = size
        self.batch_index   = 0
        self.kmeans        = False
        
        
    def load_training(self, val_size=0.3):
        '''
        A function to load training data and split into train and test set.
        To limit memory overhead during train/val splitting, data is loaded 
        once, and kept as is; train/val indices are generated.
        '''
        # loading entire train set
        self.images, self.masks, self.masks_obj, self.n_masks = load_train(self.train_folder,size=self.size,return_mask_report=True)
        assert len(self.images) == len(self.masks) == len(self.masks_obj)
        
        # splitting training data into training and validation        
        indices = np.arange(0, len(self.images))
        self.idx_train, self.idx_val = train_test_split(indices,test_size=val_size,random_state=42)
        self.n_train = len(self.idx_train)
        self.n_val   = len(self.idx_val)
        
        
    def kmeans_cluster(self, n_clusters=[2,3,5,7], data="train",reduce_size=None):
        '''
        a function to generate cluster maps using k-means.
        '''
        self.kmeans = True
        if data=="train":
            self.cluster_maps = kmeans_segmentor(self.images, n_clusters, reduce_size=reduce_size)
        if data=="test":
            self.cluster_maps_test = kmeans_segmentor(self.test_images, n_clusters, reduce_size=reduce_size)
    
        
    def load_testing(self):
        '''
        a function to load test data
        '''
        self.test_images, self.test_ids = load_test(self.test_folder, return_testid=True)
        assert len(self.test_images) == len(self.test_ids)
        self.n_test = len(self.test_images)
    
    
    def set_augmentor(self,fliplr=True,flipud=True,rot = 3,rot_mode='edge',rot90=True):
        '''
        set augmentor parameters
        '''
        self.fliplr   = fliplr 
        self.flipud   = flipud 
        self.rot      = rot
        self.rot_mode = rot_mode 
        self.rot90    = rot90
        
        
    def reset_batch_iterator(self):
        '''
        set internal batch iterator to zero
        '''
        self.batch_index = 0 
        
        
    def get_batch_train(self, batch_size, augment=True):
        '''
        a function to get batches from training data. Batches will be
        augmented if set to True.
        '''
        start=self.batch_index; end=start+batch_size
        self.batch_index = end
        if end > self.n_train:
            end = self.n_train
            self.batch_index = 0
        if start == end:
            start = 0; end = start + batch_size
            self.batch_index = end
        batch_images       = self.images[self.idx_train][start:end]
        batch_masks        = self.masks[self.idx_train][start:end]
        if self.kmeans:
            batch_cluster_maps = self.cluster_maps[self.idx_train][start:end]
            batch_images       = np.concatenate([batch_images, batch_cluster_maps],axis=-1)
        if augment:
            batch_images, batch_masks = augmenter(batch_images, batch_masks,
                                                  self.fliplr,self.flipud,
                                                  self.rot,self.rot_mode,
                                                  self.rot90)
        batch_masks = np.where(batch_masks > 0.5,1,0)
        return batch_images, batch_masks
        
    
    def get_val(self):
        '''
        a function to return validation data
        '''
        if self.kmeans:
            return np.concatenate([self.images[self.idx_val],self.cluster_maps[self.idx_val]],axis=-1), self.masks[self.idx_val]
        else:
            return self.images[self.idx_val], self.masks[self.idx_val]
        
    
    
if __name__ == "__main__":
    train_folder = './DataSet/stage1_train'
    test_folder  = './DataSet/stage1_test'
    dataloader   = DataLoader(train_folder,test_folder, (32,32) )
    
    dataloader.load_training()
    dataloader.set_augmentor()
    dataloader.kmeans_cluster()
    x,y = dataloader.get_batch_train(batch_size=10,augment=True)
    plt.subplot(1,3,1)
    plt.imshow(x[0,:,:,:3])
    plt.subplot(1,3,2)
    plt.imshow(x[0,:,:,:3]/x[0,:,:,:3].max())
    plt.subplot(1,3,3)
    plt.imshow(y[0,:,:,0])
#    dataloader.reset_batch_iterator()
    print(np.unique(y))