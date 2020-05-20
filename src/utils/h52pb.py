from keras.models import load_model,model_from_json
import tensorflow as tf
import os 
import sys
from keras import backend as K
sys.path.append(os.path.join(os.path.dirname(__file__),'../networks'))
from vgg import CSRNet

def loadmodel(json_path,weight_path):
    # json_p = open(json_path,'r')
    # json_cnts = json_p.read()
    # model_net = model_from_json(json_cnts)
    model_net = CSRNet()
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
    tf.identity(h5_model.input,'input')
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
    net = loadmodel('/data/models/head/csr_ShangHai.json','/data/models/head/csr_ShangHai_best2.h5')
    outpath = '/data/models/head/'
    modelname = 'csr_keras2.pb'
    h5_to_pb(net,outpath,modelname)
    print('model saved')