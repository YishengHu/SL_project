import numpy as np
import torch, torchvision
import librosa
from torch.utils.data import DataLoader

featureClass={0:"mfccPadData.npy",1:"mfccmeanData.npy"}
labelClass={0:"labelBinary.npy",1:"labelMulti.npy",2:"labelWho.npy"}

class myDatasetBinaryPad(torch.utils.data.Dataset):
    def __init__(self,flag,feature):
        #flag [0 train,1 test]
        self.df=np.load('data/'+featureClass[feature],allow_pickle=True)
        self.labels=np.load('data/'+labelClass[0],allow_pickle=True)
        LabelTypes=2
        data=[[]for i in range(LabelTypes)]
        for i in range(len(self.df)):
            data[self.labels[i][0]].append(self.df[i])
        self.borders=[[0,int(len(data[i])*0.7),int(len(data[i]))] for i in range(LabelTypes)]
        dataBinary=np.array(data)
        
        data_df=[[]for i in range(LabelTypes)]
        lables=[[]for i in range(LabelTypes)]
        for i in range(0,LabelTypes):
            for j in range(self.borders[i][flag],self.borders[i][flag+1]):
                data_df[i].append(dataBinary[i][j])
                lables[i].append(i)

        if flag==0:
            maxnumber=max([len(i) for i in data_df])
            data_df_enh=[]
            lables_enh=[]
            for i in range(LabelTypes):
                data_df_enh.append(np.repeat(data_df[i],maxnumber//len(data_df[i]),axis=0))
                lables_enh.append(np.repeat(lables[i],maxnumber//len(data_df[i]),axis=0))

            self.data_df=np.array(np.concatenate(data_df_enh,axis=0))
            self.lables=np.array(np.concatenate(lables_enh,axis=0))
        else :
            self.data_df=np.array(np.concatenate(data_df,axis=0))
            self.lables=np.array(np.concatenate(lables,axis=0) )            
        
    def __getitem__(self, index):
        return  self.data_df[index],self.lables[index]
    def __len__(self):
        return len(self.data_df)
    
class myDatasetClassesPad(torch.utils.data.Dataset):
    def __init__(self,flag,feature):
        #flag [0 train,1 test]
        self.df=np.load('data/'+featureClass[feature],allow_pickle=True)
        self.labels=np.load('data/'+labelClass[1],allow_pickle=True)
        LabelTypes=6
        data=[[]for i in range(LabelTypes)]
        for i in range(len(self.df)):
            if(self.labels[i][0]>1):
                data[self.labels[i][0]-2].append(self.df[i])
        self.borders=[[0,int(len(data[i])*0.7),int(len(data[i]))] for i in range(LabelTypes)]
        dataBinary=np.array(data)
        
        data_df=[[]for i in range(LabelTypes)]
        lables=[[]for i in range(LabelTypes)]
        for i in range(0,LabelTypes):
            for j in range(self.borders[i][flag],self.borders[i][flag+1]):
                data_df[i].append(dataBinary[i][j])
                lables[i].append(i)

        if flag==0:
            maxnumber=max([len(i) for i in data_df])
            data_df_enh=[]
            lables_enh=[]
            for i in range(LabelTypes):
                data_df_enh.append(np.repeat(data_df[i],maxnumber//len(data_df[i]),axis=0))
                lables_enh.append(np.repeat(lables[i],maxnumber//len(data_df[i]),axis=0))

            self.data_df=np.array(np.concatenate(data_df_enh,axis=0))
            self.lables=np.array(np.concatenate(lables_enh,axis=0))
        else :
            self.data_df=np.array(np.concatenate(data_df,axis=0))
            self.lables=np.array(np.concatenate(lables,axis=0) )            
        
    def __getitem__(self, index):
        return  self.data_df[index],self.lables[index]
    def __len__(self):
        return len(self.data_df)
