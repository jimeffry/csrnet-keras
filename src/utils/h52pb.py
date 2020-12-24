from keras.models import load_model,model_from_json
import tensorflow as tf
import os 
import sys
from keras import backend as K
sys.path.append(os.path.join(os.path.dirname(__file__),'../networks'))
from vgg import CSRNet
from resnet import get_model,ResNet50_Model
from deeplabv3plus import Deeplabv3

def loadmodel(json_path,weight_path):
    # json_p = open(json_path,'r')
    # json_cnts = json_p.read()
    # model_net = model_from_json(json_cnts)
    # model_net = CSRNet()
    # model_net = get_model(cls_num=21,load_weight=None)
    K.set_learning_phase(0)
    # model_net = ResNet50_Model(input_shape=(112,112,3),classes=21)
    model_net = Deeplabv3(weights=None,input_shape=(1088,1920,3),classes=24,backbone='xception')
    model_net.load_weights(weight_path)
    # model_net.summury()
    return model_net

def h5_to_pb(h5_model,output_dir,model_name,out_prefix = "output_",log_tensorboard = False):
    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)
    out_nodes = []
    # for i in range(len(h5_model.outputs)):
    #     out_nodes.append(out_prefix + str(i + 1))
    #     tf.identity(h5_model.output[i],out_prefix + str(i + 1))
    # tf.identity(h5_model.input,'input')
    tf.identity(h5_model.output,'output')
    out_nodes.append('output')
    sess = K.get_session()
    from tensorflow.python.framework import graph_util,graph_io
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess,init_graph,out_nodes)
    graph_io.write_graph(main_graph,output_dir,name = model_name,as_text = False)
    if log_tensorboard:
        from tensorflow.python.tools import import_pb_to_tensorboard
        import_pb_to_tensorboard.import_to_tensorboard(os.path.join(output_dir,model_name),output_dir)

if __name__=='__main__':
    # net = loadmodel('/data/models/head/csr_ShangHai.json','/data/models/head/csr_nwpu_best.h5')
    # outpath = '/data/models/head/'
    # modelname = 'csr_keras2.pb'
    # h5_to_pb(net,outpath,modelname)
    # print('model saved')
    # net = loadmodel('/data/models/face_attribute/CelebA/faceattr_celeba.json','/data/models/face_attribute/CelebA/faceattr_celeba_best.h5')
    net = loadmodel(None,'/data/models/img_seg/deeplabplus_wuwei_best.h5')
    # outpath = '/data/models/face_attribute/CelebA/'
    # modelname = 'faceattr_resnet50.pb'
    outpath = '/data/models/img_seg/'
    modelname = 'deeplabplus_keras.pb'
    h5_to_pb(net,outpath,modelname)
    print('model saved')