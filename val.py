import sklearn
import xlrd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import json
import torch.nn as nn
import torch
from utils import loaddata, data_analysis, plot_curve
from NN_classfication import NNmodel
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score,precision_recall_curve, average_precision_score,precision_recall_fscore_support,accuracy_score

if __name__ == "__main__":
    device = torch.device("cpu")
    data_dict = loaddata("./DryBeanDataset/Dry_Bean_Dataset.xls")
    train_data, train_lb, test_data, test_lb, mean, std = data_analysis(data_dict)
    # print(mean)
    # print(std)

    train_data = (train_data - mean)/std
    test_data = (test_data - mean)/std
    test_lb_onehot = label_binarize(test_lb,classes = [0,1,2,3,4,5,6])


    model = torch.load("./best_pt.pt")
    model.eval()
    model.to(device)
    acc = 0
    test_num = test_data.shape[0]
    pred = np.zeros((test_num,7))
    pred_argmax = np.zeros((test_num,1))

    for i in range(test_num):
        test_in, _lb = torch.tensor(test_data[i,:]).to(device), torch.tensor(test_lb[i,:],dtype=torch.long).to(device)
        test_in = torch.unsqueeze(test_in,0)
        out = model(test_in)
        result = torch.argmax(out)
        acc += torch.eq(result,_lb).sum().item()

        pred[i,:] = out.detach().numpy()
        pred_argmax[i,:] = result.detach().numpy()
    pre  = average_precision_score(test_lb_onehot,pred)
    print(pre)
    print("{:.10f}".format(acc/test_num))

    pre_l= accuracy_score(test_lb.T[0,:], pred_argmax.T[0,:])
    print(pre_l)


