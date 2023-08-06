# ECG_SNN


Yifei Feng, Shijia Geng, Jianjun Chu, Zhaoji Fu, Shenda Hong*. Building and Training a Deep Spiking Neural Network for ECG Classification. Biomedical Signal Processing and Control 77 (2022): 103749. https://www.sciencedirect.com/science/article/pii/S1746809422002713

# Background

The electrocardiogram (ECG) reflects the electrical activity of the heart, and is one the most widely used biophysical signals that evaluate heart-related conditions. With years of experiences, medical professionals are able to identify and classify various ECG patterns. However, manually classifying ECG signals is prone to errors and takes considerable amount of time and effort, and thus people start to explore computational models for ECG classification. In recent years, deep artificial neural networks (ANNs) have gained increasing popularity in many fields for their outstanding performances. Traditional ANNs consist of computational units which are inspired from biological neurons but ignore the neural signal transmission details. Spiking neural networks (SNNs), on the other hand, are based on impulse neurons that more closely mimic biological neurons, and thus have a great potential to achieve similar performance with much less power. Nevertheless, SNNs have not become prevalent, and one of the primary reasons is that training SNNs especially the ones with deep structures remains a challenge. 

## What we do

In this paper, we aim to propose an efficient way to build and train a deep SNN for ECG classification by constructing a counterpart structure of a deep ANN, transferring the trained parameters, and replacing the activation functions with leaky integrate-and-fire (LIF) neurons. The results show that the accuracy of the deep SNN even exceeds the original ANN. In addition, we compare and discuss the effects of different ANN activation functions on the SNN performance.


# Task Description

Please refer to the Challenge website https://physionet.org/challenge/2017/#introduction and Challenge description paper http://www.cinc.org/archives/2017/pdf/065-469.pdf. 

For SNN, we utilize SpikingJelly https://github.com/fangwei123456/spikingjelly, an open source framework which can accomplish the transformation from ANN to SNN.

## Dataset

**Data** Training data can be found at https://archive.physionet.org/challenge/2017/#challenge-data

**Label** Please use Revised labels (v3) at https://archive.physionet.org/challenge/2017/REFERENCE-v3.csv

**Preprocessed** Or you can download my preprocessed dataset challenge2017.pkl from https://drive.google.com/drive/folders/1AuPxvGoyUbKcVaFmeyt3xsqj6ucWZezf

```data/raw_data.npy``` is a large file, you can download it from above **Preprocessed** link and put it in the folder. 