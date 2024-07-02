import torch
import numpy as np
import pandas as pd
import time
from dataset import *
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  
import seaborn as sns  

def train(model,epochs,train_loader,test_loader,device):
    criterion=torch.nn.CrossEntropyLoss().to(device)
    #criterion=torch.nn.NLLLoss().to(device)

    optimizer=torch.optim.Adam(model.parameters())
    train_loss_list=[]
    test_loss_list=[]
    train_acc_list=[]
    test_acc_list=[]
    y_pre=[]
    y_lab=[]
    for epoch in range(epochs):
        train_loss=0
        number=0
        correct=0
        model.train()
        epoch_time = time.time()
        for i,(images,lables) in enumerate(train_loader):    
            images,lables=images.float().to(device),lables.long().to(device)
            optimizer.zero_grad()
            output=model(images).to(device)

            loss=criterion(output,lables).to(device)

            train_loss+=loss.item()*images.shape[0]

            loss.backward()
            optimizer.step()

            preditction=torch.max(output,dim=1)[1]
            correct+=(lables==preditction).sum().item()       

            number+=images.shape[0]
        train_loss = train_loss/number
        print("Train Epoch: {} cost time: {:.3f} train_loss {:.3f} accuracy {:.2f}%".format(epoch + 1, time.time() - epoch_time,train_loss,correct/number*100))
        train_accuracy=correct/number
        train_loss_list.append(train_loss)
        train_acc_list.append(train_accuracy)
        model.eval()
        y_pre_epoch=[]
        y_lab_epoch=[]
        with torch.no_grad():
            epoch_time = time.time()
            test_loss=0
            number=0
            correct=0
            for i,(images,lables) in enumerate(test_loader,0):
                images,lables=images.float().to(device),lables.long().to(device)
                output=model(images)
                loss=criterion(output,lables)
                test_loss+=loss.item()*images.shape[0]
                number+=images.shape[0]

                preditction=torch.max(output,dim=1)[1]
               # print(preditction)
                correct+=(lables==preditction).sum().item()
                y_pre_epoch+=preditction
                y_lab_epoch+=lables
            test_loss = test_loss/number
            test_loss_list.append(test_loss)
            test_acc_list.append(correct/number)
            print("Epoch: {} cost time: {:.4f} test_loss {:.4f} test accuracy {:.2f}%" .format(epoch + 1, time.time() - epoch_time,test_loss,correct/number*100))
            print(' ')
        y_pre.append(y_pre_epoch)
        y_lab.append(y_lab_epoch)
    return train_loss_list,test_loss_list,train_acc_list,test_acc_list,np.array(torch.Tensor(y_pre)),np.array(torch.Tensor(y_lab))

def plot_epoch(train_loss_list,test_loss_list,train_acc_list,test_acc_list,name):
    plt.plot(train_loss_list)  
    plt.plot(test_loss_list)  
    plt.xlabel('epoch')  
    plt.ylabel('loss')  
    plt.grid(True)  
    plt.savefig("pic/epoch_loss_"+name+'.png')
    plt.show()

    plt.plot(train_acc_list)  
    plt.plot(test_acc_list)  
    plt.xlabel('epoch')  
    plt.ylabel('acc')  
    plt.grid(True)  
    plt.savefig("pic/epoch_acc_"+name+'.png')
    plt.show()

def confusion_matrix_plot_Report(y_lab, y_pre,name):
    conf_mat = confusion_matrix(y_lab, y_pre)  
    np.save('result/conf_mat_'+name+'.npy',conf_mat)

    print("Confusion Matrix:")  
    print(conf_mat)  
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap='Blues')  
    plt.xlabel('Predicted')  
    plt.ylabel('True')  
    plt.title('Confusion Matrix')  
    plt.savefig("pic/Confusion_Matrix_"+name+'.png')

    plt.show()
    report = classification_report(y_lab, y_pre)  
    print("Classification Report:")  
    np.save('result/Classification_Report_'+name+'.npy',report)
    print(report)

def dataset_read(batch_size,flag,feature=0):
    dataset_dic={0:myDatasetBinaryPad,1:myDatasetClassesPad}
    dataset=dataset_dic[flag]
    data_set=dataset(0,feature)
    train_loader=DataLoader(data_set,batch_size=batch_size,shuffle=True)
    test_set=dataset(1,feature)
    test_loader=DataLoader(test_set,batch_size=batch_size,shuffle=True)
    return train_loader,test_loader

def save_result(name,result):
    item=['train_loss_list','test_loss_list','train_acc_list','test_acc_list','y_pre','y_lab']
    for i in range(6):
        np.save('result/result_'+name+'_'+item[i]+'.npy',result[i])