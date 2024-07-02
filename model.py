import torch
import numpy as np

class mlp(torch.nn.Module):
    def __init__(self,c_out):
        super(mlp, self).__init__()
        self.flatten=torch.nn.Flatten(1)
        self.fc=torch.nn.Linear(40*706,2048)
        self.fc2=torch.nn.Linear(2048,c_out) 
    def forward(self,x):
        out=self.flatten(x)
        out=torch.relu(self.fc(out))
        out=self.fc2(out)
        return out
    
class LSTM(torch.nn.Module):
    def __init__(self,c_in,d_model,lstm_num_layer,dropout,c_out):
        super(LSTM, self).__init__()
        self.lstm1=torch.nn.LSTM(input_size=c_in,
                            hidden_size=d_model,
                            num_layers=lstm_num_layer,
                            dropout=dropout,
                            bias=True,batch_first=True)
        self.flatten=torch.nn.Flatten(1)
        self.fc=torch.nn.Linear(d_model*706,c_out)
    def forward(self, x):
        output,(h_n,c_n)=self.lstm1(x)
        return self.fc(self.flatten(output))
    
    
class CNN(torch.nn.Module):
    def __init__(self,c_out):
        super(CNN,self).__init__()                                  
        self.conv1 = torch.nn.Conv2d(1, 4, (10,2), padding=1) 
        self.conv2 = torch.nn.Conv2d(4, 8, (10,2), padding=1)  
        self.bm1 = torch.nn.MaxPool2d(2)                              
        self.conv3 = torch.nn.Conv2d(8, 8, (10,2), padding=1) 
        self.bm2 = torch.nn.MaxPool2d(2)                             
        self.fc1 = torch.nn.Linear(8*169*11, 64)
        self.fc2 = torch.nn.Linear(64, c_out)
    def forward(self, x):
        x=torch.reshape(x,[-1,1,706,40])
        out = torch.relu(self.conv1(x))
        out = torch.relu(self.conv2(out))
        out = self.bm1(out)
        out = torch.relu(self.conv3(out))
        out = self.bm2(out)
        out = torch.flatten(out, 1)
        return  self.fc2(torch.relu(self.fc1(out)))