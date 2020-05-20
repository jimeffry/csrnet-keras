import keras.backend as K
import tensorflow as tf
import numpy as np

def MSE_BCE(y_true, y_pred, alpha=1000, beta=10):
    mse = K.mean(K.square(y_true - y_pred), axis=-1)
    bce = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    return alpha * mse + beta * bce

def multiloss(y_true,y_pred):
    '''
    y_true: shape is [batch,1,imgh,imgw]
    y_pred: shape is [batch,imgh,imgw]
    '''
    # outshape = K.int_shape(y_pred)
    # y_pred = K.reshape(y_pred,[outshape[0],outshape[2],outshape[3]])
    # y_pred = K.squeeze(y_pred,axis=1)
    loss_mse = K.mean(K.square(y_true - y_pred),axis=(1,2))
    # Compute max conf across batch for hard negative mining
    pred_num = tf.stop_gradient(K.sum(y_true,axis=(1,2)))
    gt_num = tf.stop_gradient(K.sum(y_pred,axis=(1,2)))
    diff_loss = tf.stop_gradient(K.abs(pred_num - gt_num))
    # diff_loss = tf.stop_gradient(diff_loss / gt_num)
    loss_mse = loss_mse * diff_loss *1000
    # print(loss_l.size())
    return loss_mse

if __name__=='__main__':
    a=K.zeros([2,1,3,3])
    b= K.ones([2,3,3])
    l = multiloss(b,a)
    print("out:",np.array(l))
