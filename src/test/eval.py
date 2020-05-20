from keras.models import load_model,model_from_json
import numpy as np
import tqdm
import os 
import sys
from keras import backend as K
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__),'../networks'))
from vgg import CSRNet
sys.path.append(os.path.join(os.path.dirname(__file__),'../preparedata'))
from utils_dataloader import DataLoader

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
def params():
    parser = argparse.ArgumentParser(description='CSRNet test')
    parser.add_argument('--data_dir',default='ShangHai',help='test target')
    parser.add_argument('--batch_size',default=2, type=int,help='Batch size for training')
    parser.add_argument('--trained_model',default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--out_file',default='../logs/test2train.txt',type=str,help='output record')
    parser.add_argument('--test_part',default='part_A_final',type=str,help='test data')
    return parser.parse_args()

def eval_acc(args):
    weight_path = args.trained_model
    record_w = open(args.out_file,'w')
    record_w.write(' dataname diff_num \n')
    #*******load data
    val_dataseta = DataLoader(args.data_dir,1,image_sets=[('part_A_final', 'test_data')])
    val_datasetb = DataLoader(args.data_dir,1,image_sets=[('part_B_final', 'test_data')])
    # build net
    model_net = CSRNet()
    model_net.load_weights(weight_path)
    #print(">>",net)
    diffa = val(model_net,val_dataseta,1)
    diffb = val(model_net,val_datasetb,1)
    record_w.write('{}\t{:.3f}\n'.format('shanghaiA',diffa))
    record_w.write('{}\t{:.3f}\n'.format('shanghaiB',diffb))

def val(net,val_loader,batch_size):
    diff_sum = 0.0
    for batch_idx in tqdm.tqdm(range(int(val_loader.batch_num))):
        images ,targets = val_loader.next_batch()
        out = net.predict(images)
        # diff = cal_diffnum(out,targets)
        # outshape = K.shape(out)
        # out = K.reshape(out,(outshape[0],outshape[2],outshape[3]))
        diff = abs(np.sum(out)-np.sum(targets))
        diff_sum += diff
    total_num = batch_size *(batch_idx+1)
    print('test MAE:%.4f' % (diff_sum/total_num))
    return diff_sum/total_num

if __name__=='__main__':
    args = params()
    eval_acc(args)