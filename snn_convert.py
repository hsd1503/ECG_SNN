
import os
import sys
import torch
import numpy as np
import torch.nn as nn
from spikingjelly.clock_driven.ann2snn import parser, classify_simulator
sys.path.append('utils')
from utils.test_model import cal_F1, test_model
from utils.create_data import create_data
import utils.global_variables as gv
import matplotlib.pyplot as plt

data_dirc = './data/'
model_name = 'best_model'

AUGMENTED = True
RATIO = 19
PREPROCESS = 'zero'
RAW_LABELS = np.load(data_dirc+'raw_labels.npy')
PERMUTATION = np.load(data_dirc+'random_permutation.npy')
BATCH_SIZE = 64
MAX_SENTENCE_LENGTH = 18000  #这里自定义一个max_length
device = gv.device
LEARNING_RATE = 0.001
NUM_EPOCHS = 200  # number epoch to train
PADDING = 'two'

def convert(log_dir='saved_model'):

    ann = torch.load( log_dir+'./' + model_name + '.pth')

    data = np.load(data_dirc+'raw_data.npy',allow_pickle=True)

    train_data, train_label, val_data, val_label = create_data(data, RAW_LABELS, PERMUTATION, RATIO, PREPROCESS, MAX_SENTENCE_LENGTH, AUGMENTED, PADDING)

    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)
                                               
    val_dataset = torch.utils.data.TensorDataset(val_data, val_label)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            #  batch_size=BATCH_SIZE,
                                            batch_size=8,
                                            shuffle=False)
    
    print('validating best model...')
    ann_acc = test_model(data_loader=val_loader,model=ann)



    # 加载用于归一化模型的数据
    # Load the data to normalize the model
    percentage = 0.004 # load 0.004 of the data
    norm_data_list = []
    for idx, (imgs, targets) in enumerate(train_loader):
        norm_data_list.append(imgs)
        if idx == int(len(train_loader) * percentage) - 1:
            break
    norm_data = torch.cat(norm_data_list)
    print('use %d imgs to parse' % (norm_data.size(0)))

    # 调用parser，使用kernel为onnx
    # Call parser, use onnx kernel
    onnxparser = parser(name=model_name,
                        log_dir=log_dir + '/parser',
                        kernel='onnx')
    snn = onnxparser.parse(ann, norm_data.to(device))

    # 保存转换好的SNN模型
    # Save SNN model
    torch.save(snn, os.path.join(log_dir,'snn-'+model_name+'.pkl'))
    fig = plt.figure('simulator')

    # 定义用于分类的SNN仿真器
    # define simulator for classification task
    sim = classify_simulator(snn,
                             log_dir=log_dir + '/simulator',
                             device=device,
                             canvas=fig
                             )
    # 仿真SNN
    # Simulate SNN
    sim.simulate(val_loader,
                T=100,#仿真时长
                online_drawer=True,
                ann_acc=ann_acc,
                fig_name=model_name,
                step_max=True
                )

if __name__=="__main__":
    convert()