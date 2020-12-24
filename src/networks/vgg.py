from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Conv2D, Input, MaxPooling2D, UpSampling2D,Softmax,Multiply
from keras import backend as K  
from keras.models import Model
from keras.initializers import RandomNormal
from keras.engine.topology import Layer

class MaxLayer(Layer):
    def __init__(self,**kwargs):
        super(MaxLayer,self).__init__(**kwargs)
    def bulid(self,input_shape):
        super(MaxLayer,self).buld(input_shape)
    def call(self,x):
        return K.max(x, axis=1, keepdims=True)
    def compute_output_shape(self,input_shape):
        return (input_shape[0],1,input_shape[2],input_shape[3])

class Squeeze(Layer):
    def __init__(self,**kwargs):
        super(Squeeze,self).__init__(**kwargs)
    def bulid(self,input_shape):
        super(Squeeze,self).buld(input_shape)
    def call(self,x):
        return K.squeeze(x, axis=1)
    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[2],input_shape[3])


def CSRNet(input_shape=(3,None,None),d_format='channels_first'):

    input_flow = Input(shape=input_shape)
    dilated_conv_kernel_initializer = RandomNormal(stddev=0.01)

    # front-end
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', data_format=d_format, activation='relu')(input_flow)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', data_format=d_format,activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2),data_format=d_format)(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', data_format=d_format,activation='relu')(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', data_format=d_format,activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2),data_format=d_format)(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', data_format=d_format, activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', data_format=d_format,activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', data_format=d_format,activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2),data_format=d_format)(x)

    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', data_format=d_format,activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', data_format=d_format,activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', data_format=d_format,activation='relu')(x)

    # back-end
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', data_format=d_format,dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', data_format=d_format,dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', data_format=d_format,dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', data_format=d_format,dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', data_format=d_format,dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', data_format=d_format,dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(x)
    
    seg_map = Conv2D(4, (3, 3), strides=(1, 1), padding='same',data_format=d_format, activation='relu',kernel_initializer=dilated_conv_kernel_initializer)(x)
    feature_map = Conv2D(1, (3, 3), strides=(1, 1), padding='same', data_format=d_format,activation='relu',kernel_initializer=dilated_conv_kernel_initializer)(x)
    outx = Conv2D(1, 1, strides=(1, 1), activation='relu',data_format=d_format, kernel_initializer=dilated_conv_kernel_initializer)(feature_map)
    output_flow = UpSampling2D(size=(8, 8), data_format=d_format, interpolation='nearest')(outx)
    # output_flow = Conv2D(1,(3,3), strides=(1, 1),padding='same', data_format=d_format,activation='relu',kernel_initializer=dilated_conv_kernel_initializer)(output_flow)
    output_flow = Squeeze()(output_flow)
    model = Model(inputs=input_flow, outputs=output_flow)

    # front_end = VGG16(weights='imagenet', include_top=False)
    # # front_end.summary()
    # # model.summary()
    # weights_front_end = []
    # for layer in front_end.layers:
    #     if 'conv' in layer.name:
    #         weights_front_end.append(layer.get_weights())
    # counter_conv = 0
    # for i in range(len(front_end.layers)):
    #     if counter_conv >= 10:
    #         break
    #     if 'conv' in model.layers[i].name:
    #         model.layers[i].set_weights(weights_front_end[counter_conv])
    #         counter_conv += 1
    return model
