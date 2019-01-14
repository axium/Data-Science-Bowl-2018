import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam, SGD
import keras.callbacks as CallBacks
from keras.losses import binary_crossentropy
from keras import backend as K
import tensorflow as tf

from dataloader import DataLoader
from models import *



OPT_DICT = {"adam":     tf.train.AdamOptimizer,
            "sgd":      tf.train.GradientDescentOptimizer,
            "rmsprop":  tf.train.RMSPropOptimizer,
            "adadelta": tf.train.AdadeltaOptimizer}



# computes dice loss with binary cross entropy
def dice_plus_xent_loss(ground_truth, prediction):
    """
    BORROWED FROM NIFTYNET SOURCE CODE
    """
    ground_truth = tf.reshape(ground_truth, (tf.shape(ground_truth)[0], -1))
    prediction = tf.reshape(prediction, (tf.shape(prediction)[0], -1))
    prediction = tf.cast(prediction, tf.float32)
    loss_xent = tf.keras.losses.binary_crossentropy(ground_truth, prediction)
    # dice as according to the paper:
    dice_numerator = 2.0 * tf.reduce_sum(prediction * ground_truth, axis=-1)
    dice_denominator = \
        tf.reduce_sum(tf.square(prediction), axis=-1) + \
        tf.reduce_sum(tf.square(ground_truth), axis=-1)

    epsilon = 0.00001

    loss_dice = - (dice_numerator + epsilon) / (dice_denominator + epsilon)
    return tf.reduce_mean(loss_dice) + tf.reduce_mean(loss_xent)

def mse( groundtruth, prediction):
    return tf.reduce_mean(tf.square(groundtruth- prediction))

# computes average iou over range of thresholds from competition
def average_iou(groundtruth, prediction):
    prec = []
    for t in np.arange(0.5,1.0,0.05):
        pred = tf.to_int32(prediction > t)
        score, up_opt = tf.metrics.mean_iou(groundtruth, pred, 2)
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return tf.reduce_mean(tf.stack(prec), axis=0)

def accuracy(ground_truth, prediction):
    ground_truth = tf.reshape(ground_truth, (tf.shape(ground_truth)[0], -1))
    prediction = tf.reshape(prediction, (tf.shape(prediction)[0], -1))
    return tf.reduce_mean(tf.keras.metrics.binary_accuracy(ground_truth, prediction))
    


# Trainer class for Training the model
class Trainer():
    def __init__(self, name, model, lr, iterations, batch_size, optimizer, dataloader,
                 print_freq, save_freq, log_dir):
        self.name       = name
        self.model      = model
        self.lr         = lr
        self.batch_size = batch_size
        self.optimizer  = OPT_DICT[optimizer]
        self.dataloader = dataloader
        self.print_freq = print_freq
        self.save_freq  = save_freq
        self.log_dir    = log_dir# "./saved/"
        self.iterations  = iterations
        
    
    def train(self, resume=False):
        '''
        a method to train the model
        '''
        # constructing graph
        with tf.variable_scope("model",reuse=tf.AUTO_REUSE):
            with tf.name_scope("model"):
                x_tf = tf.placeholder(dtype="float32",shape=(None,)+self.dataloader.size+(3,))
                y_pred = self.model(x_tf)
                y_tf = tf.placeholder(dtype="float32",shape=(None,)+self.dataloader.size+(1,))
        # setting up losses and metrics
        with tf.variable_scope("losses",reuse=tf.AUTO_REUSE):
            with tf.name_scope("losses"):
                loss = dice_plus_xent_loss(y_tf, y_pred) + 0.01*mse(y_tf, y_pred)
                acc  = accuracy(y_tf, y_pred)
                iou  = average_iou(y_tf, y_pred)
        with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
            opt = self.optimizer(self.lr).minimize(loss, var_list=self.model.trainable_weights)
        # setting up session
        self.sess      = tf.Session()
        global_op = tf.global_variables_initializer()
        local_op  = tf.local_variables_initializer()
        self.sess.run([global_op, local_op])
         
        if resume:
            self.model.load_weights(self.log_dir+"%s/model.h5"%self.name)
        # setting up summary writer
        writer_train    = tf.summary.FileWriter(self.log_dir+self.name+"/train", session=self.sess)
        writer_val      = tf.summary.FileWriter(self.log_dir+self.name+"/val", session=self.sess)
        summ_loss       = tf.summary.scalar(name="loss",tensor=loss)
        summ_iou        = tf.summary.scalar(name='iou', tensor=iou)
        summ_acc        = tf.summary.scalar(name='acc', tensor=acc)
        x_summ          = tf.summary.image('x_true', tf.cast(x_tf*255, "uint8"), max_outputs=3)
        y_summ          = tf.summary.image('y_true', tf.cast(y_tf*255, "uint8"), max_outputs=3)
        y_pred_summ     = tf.summary.image('y_pred', tf.cast(y_pred*255, "uint8"), max_outputs=3)

        # starting training
        for i in range(self.iterations):
            # training
            batch_x,batch_y = self.dataloader.get_batch_train(batch_size=self.batch_size,augment=True)
            _,loss_train,iou_train,acc_train = self.sess.run([opt,loss,iou,acc],feed_dict={x_tf:batch_x, y_tf:batch_y})
            loss_summ,iou_summ,acc_summ = self.sess.run([summ_loss,summ_iou,summ_acc],feed_dict={x_tf:batch_x, y_tf:batch_y})
            writer_train.add_summary(loss_summ, i)
            writer_train.add_summary(iou_summ, i)
            writer_train.add_summary(acc_summ, i)
            summ_x,summ_y,summ_pred = self.sess.run([x_summ,y_summ,y_pred_summ],feed_dict={x_tf:batch_x, y_tf:batch_y})
            writer_train.add_summary(summ_x, i)
            writer_train.add_summary(summ_y, i)
            writer_train.add_summary(summ_pred, i)
            
            # validation
            batch_x,batch_y = self.dataloader.get_val()
            loss_val,iou_val,acc_val = self.sess.run([loss,iou,acc],feed_dict={x_tf:batch_x, y_tf:batch_y})
            loss_summ,iou_summ,acc_summ = self.sess.run([summ_loss,summ_iou,summ_acc],feed_dict={x_tf:batch_x, y_tf:batch_y})
            writer_val.add_summary(loss_summ, i)
            writer_val.add_summary(iou_summ, i)
            writer_val.add_summary(acc_summ, i)
            summ_x,summ_y,summ_pred = self.sess.run([x_summ,y_summ,y_pred_summ],feed_dict={x_tf:batch_x, y_tf:batch_y})
            writer_val.add_summary(summ_x, i)
            writer_val.add_summary(summ_y, i)
            writer_val.add_summary(summ_pred, i)
            
            # print
            if i % self.print_freq == 0:
                print("step:%0.4d\tloss=%0.3f/%0.3f...iou=%0.3f/%0.3f...acc=%0.3f/%0.3f"
                      %(i,loss_train,loss_val,iou_train,iou_val,acc_train,acc_val))
        
            if i % self.save_freq == 0:
                self.model.save_weights(self.log_dir+"%s/model.h5"%self.name)