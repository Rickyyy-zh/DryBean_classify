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
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score,precision_recall_curve, average_precision_score,precision_recall_fscore_support,accuracy_score,classification_report

class NNmodel(nn.Module):
    def __init__(self, depth, dims, dropoutP):
        super().__init__()
        self.layer = []
        self.depth = depth
        self.dropout = dropoutP
        self.input = nn.Sequential(nn.Linear(16,dims[0]),
                            nn.ReLU(),
                            nn.Dropout(self.dropout))
        for l in range(depth):
            if l<depth-1:
                self.layer.append( self.make_layer(dims[l],dims[l+1]))
        self.middle = nn.Sequential(*self.layer)
        self.output = nn.Sequential(nn.Linear(dims[-1],7),
                                nn.Softmax(dim=1))

    def make_layer(self,input_ch, output_ch):
        l = nn.Sequential(nn.Linear(input_ch,output_ch),
                        nn.ReLU(),
                        nn.Dropout(self.dropout))
        return l

    def forward(self,x):
        output = self.input(x)
        # for l in range(self.depth-1):
        #     output = self.layer[l](output)
        output = self.middle(output)
        output = self.output(output)
        return output
    
if __name__ == "__main__":
    data_dict = loaddata("./DryBeanDataset/Dry_Bean_Dataset.xls")
    train_data, train_lb, test_data, test_lb, mean, std = data_analysis(data_dict)
    # print(mean)
    # print(std)

    train_data = (train_data - mean)/std
    test_data = (test_data - mean)/std
    # label = np.eye(label.shape[0], 7, k=0, dtype=np.float32)
    # device = torch.device("cuda:2") #if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu") #if torch.cuda.is_available() else "cpu")
    
    loss_fn = nn.CrossEntropyLoss()

    model = NNmodel(7,[32,64,128,256,128,64,32],0.1)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9, weight_decay=0.0005)
    # warm_lr =  torch.optim.lr_scheduler.LinearLR(optimizer=optimizer,start_factor=0.5,end_factor = 0.1, total_iters=4)
    step_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,milestones=[5,20,40], gamma=0.8)
    # all_lr = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer,schedulers= [warm_lr,step_lr],milestones=[5])
    test_lb_onehot = label_binarize(test_lb,classes = [0,1,2,3,4,5,6])

    epochs = 80
    ax_epoch = []
    ax_trainloss = []
    ax_testloss = []
    ax_acc = []
    ax_ap = []
    pred = np.zeros((test_data.shape[0],7))
    for epoch in range(epochs):
        model.train()
        train_loss = 0 
        test_loss = 0
        train_num = train_data.shape[0]
        test_num = test_data.shape[0]
        for i in range(train_num):
            input, lb = torch.tensor(train_data[i,:]).to(device), torch.tensor(train_lb[i,:],dtype=torch.long).to(device)
            input = torch.unsqueeze(input,0)
            optimizer.zero_grad()
            out = model(input).to(device)
            loss = loss_fn(out, lb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        acc =0.0
        best_acc = 0.0
        with torch.no_grad():
            for i in range(test_num):
                test_in, _lb = torch.tensor(test_data[i,:]).to(device), torch.tensor(test_lb[i,:],dtype=torch.long).to(device)
                test_in = torch.unsqueeze(test_in,0)
                out = model(test_in)
                result = torch.argmax(out)
                loss = loss_fn(out, _lb)
                acc += torch.eq(result,_lb).sum().item()
                test_loss += loss.item()
                pred[i,:] = out.detach().numpy()

        print("train epoch [{}/{}]  train_loss:{:.3f} test_loss:{:.3f} accurancy:{:.3f}".format(
                                    epoch+1, epochs, train_loss/train_num,test_loss/test_num, acc/test_num))
        step_lr.step()
        if acc >= best_acc:
            best_acc = acc
            torch.save(model, "./best_pt.pt")
        torch.save(model,"./last_pt.pt")
        ap  = average_precision_score(test_lb_onehot,pred)

        ax_ap.append(ap)
        ax_trainloss.append( train_loss/train_num)
        ax_testloss.append(test_loss/test_num)
        ax_acc.append(acc/test_num)
        ax_epoch.append(epoch)
        plot_curve(ax_epoch,ax_trainloss,ax_testloss,ax_acc,ax_ap)

    print("finshed training")

