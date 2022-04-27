import sklearn
import xlrd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from utils import loaddata, data_analysis, plot_curve
from scipy.spatial.distance import euclidean


class KNN_classfier():
    def __init__(self, K, train_data, train_lb):
        self.K = K
        self.train_data = train_data
        self.train_lb = train_lb
    
    def predict(self, input):
        dist = np.zeros((self.train_data.shape[0]))
        for idx in range(self.train_data.shape[0]):
            dist[idx] = euclidean(input, self.train_data[idx,:])
        sorted_dist_idx = np.argsort(dist,axis=0)
        kmin_dist_idx = sorted_dist_idx[:self.K]
        kmin_cls = train_lb[kmin_dist_idx].reshape(-1).astype(np.int8)
        cls = np.argmax(np.bincount(kmin_cls))
        
        return cls
    
if __name__ == "__main__":
    data_dict = loaddata("./DryBeanDataset/Dry_Bean_Dataset.xls")
    train_data, train_lb, test_data, test_lb, mean, std = data_analysis(data_dict)

    train_data = (train_data - mean)/std
    test_data = (test_data - mean)/std

    K_list = [9,10]
    for k in K_list:
        KNN_model = KNN_classfier(K = k , train_data= train_data, train_lb= train_lb)
        acc = 0
        test_num = test_data.shape[0]
        pred = np.zeros((test_num,1))

        for i in range(test_num):
            result = KNN_model.predict(test_data[i,:])
            if np.equal(result,test_lb[i,:]):
                acc = acc+1

            pred[i,:] = result
        print("with K = {}, accurancy = {:.10f}".format(k,acc/test_num))
