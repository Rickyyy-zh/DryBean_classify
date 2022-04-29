import sklearn
import xlrd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import json
import torch.nn as nn
import torch
from torch.utils.data import DataLoader,TensorDataset
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
                            # nn.Dropout(self.dropout)
                            )
        for l in range(depth):
            if l<depth-1:
                self.layer.append( self.make_layer(dims[l],dims[l+1]))
        self.middle = nn.Sequential(*self.layer)
        self.output = nn.Sequential(nn.Linear(dims[-1],7),
                                nn.Softmax(dim=1))
        self.init_weights()

    def make_layer(self,input_ch, output_ch):
        l = nn.Sequential(nn.Linear(input_ch,output_ch),
                        nn.ReLU(),
                        nn.Dropout(self.dropout))
        return l

    def init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                m.weight.data.normal_(0, 0.35)
                m.bias.data.zero_()

    def forward(self,x):
        output = self.input(x)
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

    device = torch.device("cpu") #if torch.cuda.is_available() else "cpu")
    
    # loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.09948339, 0.15253618, 0.38630809, 0.12371339, 0.10459171, 0.07649955,0.05686769]))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.09948339, 0.15253618, 9, 0.12371339, 0.10459171, 0.07649955,0.05686769]))


    model = NNmodel(7,[32,64,128,256,128,64,32],0.1)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9, weight_decay=0.005)
    # optimizer = torch.optim.Adam(model.parameters(),lr = 0.0002,weight_decay=0.00005)

    step_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,milestones=[40,80], gamma=1)
    # warm_lr =  torch.optim.lr_scheduler.LinearLR(optimizer=optimizer,start_factor=0.5,end_factor = 0.1, total_iters=4)
    # all_lr = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer,schedulers= [warm_lr,step_lr],milestones=[5])
    test_lb_onehot = label_binarize(test_lb,classes = [0,1,2,3,4,5,6])

    train_dataset = TensorDataset(torch.Tensor(train_data),torch.LongTensor(train_lb))
    test_dataset = TensorDataset(torch.Tensor(test_data),torch.LongTensor(test_lb))
    batch_sz = 5

    train_loader = DataLoader(train_dataset,batch_size= batch_sz, shuffle=False)
    test_loader = DataLoader(test_dataset,batch_size= batch_sz, shuffle=False)

    epochs = 100
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
        for i, data in enumerate(train_loader):
            # input, lb = torch.tensor(train_data[i,:]).to(device), torch.tensor(train_lb[i,:],dtype=torch.long).to(device)
            # input = torch.unsqueeze(input,0)
            input, lb = data
            optimizer.zero_grad()
            out = model(input).to(device)
            loss = loss_fn(out, torch.squeeze(lb))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        acc =0.0
        best_acc = 0.0
        with torch.no_grad():
            for i in range(test_data.shape[0]):
                test_in, _lb = torch.tensor(test_data[i,:]).to(device), torch.tensor(test_lb[i,:],dtype=torch.long).to(device)
                test_in = torch.unsqueeze(test_in,0)
                # test_in, _lb = data
                out = model(test_in)
                result = torch.argmax(out)
                loss = loss_fn(out,  _lb)
                acc += torch.eq(result,_lb).sum().item()
                test_loss += loss.item()
                pred[i,:] = out.detach().numpy()
        ap  = average_precision_score(test_lb_onehot,pred)
        print("train epoch [{}/{}]  train_loss:{:.3f} test_loss:{:.3f} accurancy:{:.3f}, ap:{:.3f}".format(
                                    epoch+1, epochs, batch_sz*train_loss/train_num,test_loss/test_num, acc/test_num,ap))
        step_lr.step()
        if acc >= best_acc:
            best_acc = acc
            torch.save(model, "./best_pt.pt")
        torch.save(model,"./last_pt.pt")
        

        ax_ap.append(ap)
        ax_trainloss.append( batch_sz*train_loss/train_num)
        ax_testloss.append(test_loss/test_num)
        ax_acc.append(acc/test_num)
        ax_epoch.append(epoch)
        plot_curve(ax_epoch,ax_trainloss,ax_testloss,ax_acc,ax_ap)

    print("finshed training")

