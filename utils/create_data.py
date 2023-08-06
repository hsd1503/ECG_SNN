import numpy as np
import torch
def create_data(raw_data = None, raw_labels = None, permutation = None, ratio = None, preprocess = None, max_length = None , augmented = None, padding = 'two'):
    ## Input:
    # raw_data: waveform data np.ndarray
    # raw_labels: label data np.ndarray
    # permutaion: fixed permutation when splitting train and valid data
    # ratio: splitting ratio when splitting train and valid data
    #        An Integer: 19 for 90% train and 10% valid
    # preprocess: 'zero' for zero padding; 'reflection' for reflection padding and simple normalization after it
    # max_length: max number of length you want to keep or pad to for an individual waveform datum
    # augmented: Boolean, augment train data to make it balance or not
    # padding: 'two' for two sides padding; 'one' for one side padding
    ## Output:
    # train_data(1*LENGTH torch.FloatTensor), train_label(LENGTH torch.LongTensor), val_data, val_label
    if preprocess == 'reflection':
        max_length = 10100
    data = np.zeros((len(raw_data), max_length))
    if preprocess == 'zero':
        for i in range(len(raw_data)):
            if len(raw_data[i]) >= max_length:
                data[i] = raw_data[i][:max_length]  # 如果数据长度长于限定的最长长度，则截掉
            else:
                remainder = max_length - len(raw_data[i])  # 如果数据长度短于限定的最长长度，则填上
                if padding == 'two':
                    data[i] = np.pad(raw_data[i], (int(remainder / 2), remainder - int(remainder / 2)), 'constant', constant_values=0)
                elif padding == 'one':
                    data[i] = np.pad(raw_data[i], (0, remainder), 'constant',
                                     constant_values=0)
    else:
        for i in range(len(raw_data)):
            if len(raw_data[i]) >= max_length:
                data[i] = raw_data[i][:max_length]
            else:
                b = raw_data[i][0:(max_length - len(raw_data[i]))]
                goal = np.hstack((raw_data[i], b))
                while len(goal) != max_length:
                    b = raw_data[i][0:(max_length - len(goal))]
                    goal = np.hstack((goal, b))  # 这里采用原ECG中的数据对样本进行加长
                data[i] = goal
        data = (data - data.mean())/(data.std())  # 标准化
    data = data[permutation]  # 对数据进行扰动
    labels = raw_labels[permutation]  # 对标签进行扰动
    if ratio == 19:
        mid = int(len(raw_data)*0.9)
    else:
        mid = int(len(raw_data)*0.7)
    train_data = data[:mid]
    val_data = data[mid:]
    train_label = labels[:mid]
    val_label = labels[mid:]
    if augmented == True:
        # replicate noisy class 5 times
        temp_data = np.tile(train_data[train_label == 3], (5,1))  # 复制数据以扩增数据
        temp_label = np.tile(train_label[train_label == 3], 5)
        train_data = np.concatenate((train_data, temp_data), axis = 0)
        train_label = np.concatenate((train_label, temp_label))

        # replicate AF class once
        temp_data = np.tile(train_data[train_label == 1], (1,1))
        temp_label = train_label[train_label == 1]
        train_data = np.concatenate((train_data, temp_data), axis = 0)
        train_label = np.concatenate((train_label, temp_label))

    train_data = torch.from_numpy(train_data).unsqueeze(-2).type(torch.FloatTensor)
    val_data = torch.from_numpy(val_data).unsqueeze(-2).type(torch.FloatTensor)
    train_label = torch.LongTensor(train_label)
    val_label = torch.LongTensor(val_label)

    # 现在的标签还是0,1,2,3这种数值，并未用0,1向量进行编码
    return train_data, train_label, val_data, val_label