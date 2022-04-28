from random import sample
import sklearn
import xlrd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from utils import loaddata, data_analysis, plot_curve

if __name__ == "__main__":
    data_dict = loaddata("./DryBeanDataset/Dry_Bean_Dataset.xls")
    train_data, train_lb, test_data, test_lb, mean, std = data_analysis(data_dict)

    train_data = (train_data - mean)/std
    test_data = (test_data - mean)/std

    label_list = ["SEKER", "BARBUNYA", "BOMBAY", "CALI", "HOROZ", "SIRA", "DERMASON"]
    count = [0,0,0,0,0,0,0]
    count_train = [0,0,0,0,0,0,0]
    count_test = [0,0,0,0,0,0,0]

    for idx,item in enumerate(data_dict):
        for i,lb in enumerate(label_list):
            if item["label"] == i:
                count[i] = count[i]+1
    for idx in range(train_lb.shape[0]):
        for i,lb in enumerate(label_list):
            if train_lb[idx,:] == i:
                count_train[i] = count_train[i]+1
    for idx in range(test_lb.shape[0]):
        for i,lb in enumerate(label_list):
            if test_lb[idx,:] == i:
                count_test[i] = count_test[i]+1

    x_width = range(0,len(label_list))
    x1_width = [i-0.15 for i in x_width]
    x2_width = [i+0.15 for i in x_width]
    fig = plt.figure()

    plt.bar(x1_width,count,width=0.3,label="overall")
    plt.bar(x2_width,count_train,width=0.3,label="train_data")
    plt.bar(x2_width,count_test,width=0.3,bottom=count_train,label="test_data")
    plt.xticks(range(0,7),label_list)
    plt.legend()

    plt.savefig("./dataanalysis.jpg")
