
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Conv2D, Input, MaxPooling2D, UpSampling2D,Softmax,Multiply,GlobalAveragePooling2D,Dropout,Flatten,BatchNormalization,Activation,Add
from keras import backend as K  
from keras.models import Model
from keras.initializers import RandomNormal
from keras.engine.topology import Layer

def identity_block(X, f, filters, stage, block):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path (≈3 lines)
    X = Conv2D(filters= F2, kernel_size=(f,f),strides=(1,1),padding='same',name=conv_name_base + '2b')(X)
    X = BatchNormalization(axis=3,name=bn_name_base+'2b')(X)
    X = Activation('relu')(X)
 
    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3,kernel_size=(1,1),strides=(1,1),padding='valid',name=conv_name_base+'2c')(X)
    X = BatchNormalization(axis=3,name=bn_name_base+'2c')(X)
 
    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X
 
def convolutional_block(X, f, filters, stage, block, s = 2):
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X
 
    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s),padding='valid',name = conv_name_base + '2a')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
 
    # Second component of main path (≈3 lines)
    X = Conv2D(F2,(f,f),strides=(1,1),padding='same',name=conv_name_base+'2b')(X)
    X = BatchNormalization(axis=3,name=bn_name_base+'2b')(X)
    X = Activation('relu')(X)
 
    # Third component of main path (≈2 lines)
    X = Conv2D(F3,(1,1),strides=(1,1),padding='valid',name=conv_name_base+'2c')(X)
    X = BatchNormalization(axis=3,name=bn_name_base+'2c')(X)
 
    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(F3,(1,1),strides=(s,s),padding='valid',name=conv_name_base+'1')(X_shortcut)
    X_shortcut = BatchNormalization(axis=3,name =bn_name_base+'1')(X_shortcut)
 
    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    return X
    
# GRADED FUNCTION: ResNet50
 
def ResNet50_Model(input_shape = (64, 64, 3), classes = 30):
    # Define the input as a tensor with shape input_shape
    X_input = Input(shape=input_shape)
 
    # Zero-Padding
    # X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1')(X_input)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
 
    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
 
    ### START CODE HERE ###
 
    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3,filters= [128,128,512],stage=3,block='a',s=2)
    X = identity_block(X,3,[128,128,512],stage=3,block='b')
    X = identity_block(X,3,[128,128,512],stage=3,block='c')
    X = identity_block(X,3,[128,128,512],stage=3,block='d')
 
    # Stage 4 (≈6 lines)
    X = convolutional_block(X,f=3,filters=[256,256,1024],stage=4,block='a',s=2)
    X = identity_block(X,3,[256,256,1024],stage=4,block='b')
    X = identity_block(X,3,[256,256,1024],stage=4,block='c')
    X = identity_block(X,3,[256,256,1024],stage=4,block='d')
    X = identity_block(X,3,[256,256,1024],stage=4,block='e')
    X = identity_block(X,3,[256,256,1024],stage=4,block='f')
 
    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3,filters= [512,512,2048],stage=5,block='a',s=2)
    X = identity_block(X,3,[512,512,2048],stage=5,block='b')
    X = identity_block(X,3,[512,512,2048],stage=5,block='c')
 
    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    # X = AveragePooling2D((2,2),strides=(2,2))(X)
    X = GlobalAveragePooling2D()(X)
 
    # output layer
    # X = Flatten()(X)
    X = Dense(classes,activation='softmax')(X) #'sigmoid'
    # X = Softmax()(X)
    model = Model(inputs = X_input, outputs = X, name='resnet50')
    #load weights
    print('*******begin load')
    front_end = ResNet50(weights='imagenet', include_top=False)
    # front_end.summary()
    # model.summary()
    weights_front_end = []
    for layer in front_end.layers:
        if isinstance(layer,Conv2D):
            weights_front_end.append(layer.get_weights())
    counter_conv = 0
    layer_cnt = 0
    print('****loading')
    for layer in model.layers:
        if isinstance(layer,Conv2D):
            layer.set_weights(weights_front_end[counter_conv])
            counter_conv += 1
        layer_cnt +=1
    return model

def get_model(cls_num=21,imgsize=(112,112,3),load_weight='imagenet'):
    K.set_learning_phase(1)
    input_flow = Input(shape=imgsize)
    base_model = ResNet50(weights=load_weight,include_top=False,input_shape=imgsize)
    # K.set_learning_phase(1)
    x = base_model(input_flow)
    x = GlobalAveragePooling2D()(x)
    # x = Flatten()(x)
    # x = Dropout(0.5)(x)
    x = Dense(cls_num,activation='sigmoid')(x)
    model = Model(inputs=input_flow,outputs=x,name='resnet50')
    return model

if __name__=='__main__':
    # a=np.ones([1,112,112,3])
    tnet = ResNet50_Model((112,112,3),21)
    # tnet = get_model(load_weight=None)
    # tnet.summary()
    print(len(tnet.layers))
    # for i in range(len(tnet.layers)):
    #     print(tnet.layers[i].name)
    #     if i==1:
    #         print(len(tnet.layers[i].layers))
    #         for j in range(len(tnet.layers[i].layers)):
    #             if isinstance(tnet.layers[i].layers[j],Conv2D):
    #                 print(tnet.layers[i].layers[j].name)