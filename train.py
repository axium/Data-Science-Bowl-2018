import numpy as np
import matplotlib.pyplot as plt
from dataloader import DataLoader
from models import *
from trainer import Trainer


# loading dataset
train_folder = './DataSet/stage1_train'
test_folder  = './DataSet/stage1_test'
dataloader   = DataLoader(train_folder,test_folder, (32,32) )

dataloader.load_training()
dataloader.set_augmentor()
#dataloader.kmeans_cluster()


# loading model
model = simple_model()


from trainer import Trainer
trainer = Trainer(name="simplest",model=model, lr=0.00001, batch_size=64,optimizer="adam",
                  dataloader=dataloader, print_freq=10, save_freq=10, log_dir="./saved/",
                  iterations=10000)

trainer.train()


