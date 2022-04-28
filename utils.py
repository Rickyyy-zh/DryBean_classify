# from nbformat import write
import xlrd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import json
import random

def loaddata(path):
    workbook = xlrd.open_workbook(path)
    sheet = workbook.sheet_by_index(0)
    data_num = sheet.nrows -1 
    feature_num = sheet.ncols -1
    print("data instance number %d, feature number %d" %(data_num, feature_num))
    dataset = []
    feature_name = []
    label_dict = {"SEKER":0, "BARBUNYA":1, "BOMBAY":2, "CALI":3, "HOROZ":4, "SIRA":5, "DERMASON":6}
    for row in range(data_num+1):
        if row ==0:
            feature_name = sheet.row_values(0)
        else:
            dic_sample = {}
            dic_feature = {}
            for col in range(feature_num):
                dic_feature[feature_name[col]] = sheet.cell(row,col).value
                dic_sample["data"] = dic_feature
                dic_sample["label"] = label_dict[sheet.cell(row,feature_num).value]
            dataset.append(dic_sample)
    
    # with open(path.replace("xls","json"), "w+") as outputFile:
    #     sys.stdout = outputFile
    #     print(json.dumps(dataset))
    
    return dataset

def random_number(data_size):
    number_set = []
    for i in range(data_size):
        number_set.append(i)
    random.seed(9271)
    random.shuffle(number_set)
    with open("./randomnum.txt","w") as f:
        for num in number_set:
            s = str(number_set[num])+" "
            f.write(s)
    return number_set

def data_split(dataset,label,ratio):
    random_index = random_number(dataset.shape[0])
    traindata_sz = int( dataset.shape[0]*ratio)
    train_data = dataset[random_index[:traindata_sz],:]
    test_data = dataset[random_index[traindata_sz:],:]
    train_lb = label[random_index[:traindata_sz],:]
    test_lb = label[random_index[traindata_sz:],:]

    return train_data, train_lb, test_data, test_lb


def data_analysis(dic_data):
    data_num = len(dic_data)
    label = np.zeros((data_num,1))
    data = np.zeros((data_num, 16),dtype=np.float32)
    for i,item in enumerate(dic_data):
        label[i,0] = item["label"]
        data_list = []
        for j, feature in enumerate(item["data"]):
            data_list.append(item["data"][feature])
        data[i,:] = np.array((data_list))
    mean = np.mean(data,axis=0,dtype=np.float32)
    std  = np.std(data, axis=0,dtype=np.float32)

    train_data, train_label, test_data,test_label = data_split(data, label, ratio=0.9)
    

    return train_data, train_label, test_data,test_label, mean, std

def plot_curve(idx, data1,data2,data3,data4):
    fig = plt.figure(figsize=(20,5))
    
    ax_train_loss = fig.add_subplot(1,4,1)
    ax_train_loss.set_title("train loss")
    ax_train_loss.plot(idx,data1)
    
    ax_acc = fig.add_subplot(1,4,3)
    ax_acc.set_title("test accurancy")
    ax_acc.plot(idx,data3)
    
    ax_val_loss = fig.add_subplot(1,4,2)
    ax_val_loss.set_title("test loss")
    ax_val_loss.plot(idx,data2)
    
    ax_aps = fig.add_subplot(1,4,4)
    ax_aps.set_title("ap")
    ax_aps.plot(idx,data4)


    plt.subplots_adjust(wspace=0.5)
    
    plt.savefig("./train_process.jpg")
    plt.close(fig)