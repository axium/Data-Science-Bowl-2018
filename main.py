import numpy as np
import matplotlib.pyplot as plt
from dataloader import DataLoader
from models import *
from trainer import Trainer

# constants
model_name = "unet" # "unet" or "unet-kmeans"
augmentor  = True
n_clusters = [2,3,5,7]
kmean_reduction_size = (64,64)
mode       = "eval"
name_exp   = "experiment-unet"
lr         = 0.00001
batchsize  = 64
optimizer  = "adam" # or "sgd", "rmsprop", "adadelta"
print_freq = 10
save_freq  = 10
log_dir    = "./saved/"
iterations = 10000
resolution = (256,256)
val_size   = 0.3
save_test  = False
n_show     = 10
threshold  = 0.5

# loading dataset
train_folder = './DataSet/stage1_train'
test_folder  = './DataSet/stage1_test'
dataloader   = DataLoader(train_folder,test_folder, resolution )
dataloader.load_training(val_size=0.3)
if augmentor:
    dataloader.set_augmentor(fliplr=True,flipud=True,rot=3,rot_mode='edge',rot90=True)
if model_name == "unet-kmeans":
    dataloader.kmeans_cluster( n_clusters=n_clusters, data="train", reduce_size=kmean_reduction_size)


# loading model
if model_name == "unet":
    model = unet()
elif model_name == "unet-kmeans":
    model = unet_kmeans()
else:
    raise "model not definied"

# training model
if mode=="train":
    trainer = Trainer(name=name_exp,model=model, lr=lr, batch_size=batchsize,
                  optimizer=optimizer,dataloader=dataloader,print_freq=print_freq, 
                  save_freq=print_freq,log_dir=log_dir,
                  iterations=iterations)
    trainer.train()

if mode=="eval":
    model.load_weights("./weights/%s.h5"%model_name)
    x_val,y_val = dataloader.get_val()
    y_prob      = model.predict(x_val, batch_size=16)
    y_pred      = np.where(y_prob > threshold, 1,0)
    indices     = np.random.permutation(len(y_pred))[:n_show]
    n_rows      = len(indices); n_cols = (x_val.shape[-1] - 2) + 3 
    for i,idx in enumerate(indices):
        j=1
        plt.subplot(n_rows,n_cols,n_cols*i+j); j+=1
        plt.imshow(x_val[idx,:,:,:3]); plt.title("test")
        if model_name == "unet-kmeans":
            for k in range(len(n_clusters)):
                plt.subplot(n_rows,n_cols,n_cols*i+j); j+=1 
                plt.imshow(x_val[idx,:,:,k+3])
                plt.title("kmeans:k=%d"%n_clusters[k])
        plt.subplot(n_rows,n_cols,n_cols*i+j); j+=1
        plt.imshow((y_prob[idx,:,:,0]*255).astype("uint8")); plt.title("probability")
        plt.subplot(n_rows,n_cols,n_cols*i+j); j+=1
        plt.imshow(y_pred[idx,:,:,0]); plt.title("predicted")
        plt.subplot(n_rows,n_cols,n_cols*i+j); j+=1
        plt.imshow(y_val[idx,:,:,0]); plt.title("mask")
    
    
    