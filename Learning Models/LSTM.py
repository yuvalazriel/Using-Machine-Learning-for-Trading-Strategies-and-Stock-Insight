import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch #pytorch
import torch.nn as nn
from torch.autograd import Variable
import argparse
from keras import optimizers


'''
lstm model 
https://cnvrg.io/pytorch-lstm/?gclid=EAIaIQobChMIucH8stGg8AIVdwIGAB0rdQkpEAAYASAAEgJCFPD_BwE 
'''
#parser = argparse.ArgumentParser(description='PyTorch LSTM Transformer Language Model')
#parser.add_argument('--clip', type=float, default=0.25,help='gradient clipping')
#args = parser.parse_args()
##############Variables#########################
num_epochs = 1000 #1000 epochs
learning_rate = 0.01 #0.001 lr
input_size = 33 #number of features
hidden_size = 5 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers
num_classes = 1 #number of output classes
columns =['proba','class', 'realClass']
##############utiliy functions #########################
mm = MinMaxScaler()
ss = StandardScaler()

def isnan(value):
    try:
        import math
        return math.isnan(float(value))
    except:
        return False
'''
Accuracy and precision function (the ratio of the correctly identified positive cases to all the predicted positive cases)
where: TP = True positive; FP = False positive; TN = True negative; FN = False negative
Accuracy= (TP + TN)/(TP + TN + FP + FN)
precision = TP / (TP + FP)
https://www.python-course.eu/metrics.php
'''
def Accuracy(dataFrame):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in  range (dataFrame.shape[0]):
        if dataFrame['class'][i]== dataFrame['realClass'][i]:#dataFrame.loc(i,'class')== dataFrame.loc(i,'realClass'):
            if dataFrame['class'][i]==1:
                TP+=1
            else:
                TN+=1
        else:
            if dataFrame['class'][i]==1:
                FP+=1
            else:
                FN+=1

    accuracy = (TP + TN)/(TP + TN + FP + FN)
    precision = TP / (TP + FP)
    return accuracy,precision


##############LSTM class #########################
class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc_2 =  nn.Linear(128, 11) #fully connected 2
        self.fc = nn.Linear(11, num_classes) #fully connected last layer

        self.relu = nn.ReLU()

    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc_2(out) #second Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out


##############open test and train files#########################
fullTable=pd.read_csv(rf'../FINAL DATA/vectorForYear-1981 to Year-2020.csv')
fullTable=fullTable.drop(fullTable.columns[[0]],1)#drop the first column
#print("before fillna- shape",fullTable.shape[0])
#fullTable=fullTable.fillna(0)

#fullTable.to_csv(rf'FINAL DATA\vectorForYear-1981 to Year-2020.csv', index=True)
print("done- shape",fullTable.shape[0])
'''
cnt=0
for i in range(fullTable.shape[0]):
    for j in range(fullTable.shape[1]):
        if isnan(fullTable.iloc[i][j]):
            cnt+=1
            if cnt>0:
                break
            fullTable.iloc[i][j]=0
    if cnt>0:
        break

print("cnt",cnt)
'''
trainIndex=int(fullTable.shape[0]*0.8)
X_train= fullTable[:trainIndex]
X_train=X_train.drop(X_train.columns[[0]],1)
print(X_train.head())
fullClass=pd.read_csv(rf'../FINAL DATA/classForYear-1981 to Year-2020.csv')
y_train= fullClass[:trainIndex]
y_train=y_train.drop(y_train.columns[[0]],1)#drop the first column
y_train=y_train.drop(y_train.columns[[0]],1)#drop the first column
X_test=fullTable[trainIndex+1:]
X_test=X_test.drop(X_test.columns[[0]],1)
print(X_test.head())
y_test=fullClass[trainIndex+1:]
print(y_test.head())

##############create data frame for predictions#########################
res = pd.DataFrame( columns = columns, index=y_test.iloc[:,1])
y_test.index=res.index
# Assign the columns.
res[['realClass']] = y_test[['class']]
print(res.head())
y_test=y_test.drop(y_test.columns[[0]],1)#drop the first column
y_test=y_test.drop(y_test.columns[[0]],1)#drop the first column
print(y_test.head())

X_train_ss = ss.fit_transform(X_train)
y_train_mm = mm.fit_transform(y_train)
print("X_train_ss",X_train_ss)
print("y_train_mm",y_train_mm)
X_test_ss = ss.fit_transform(X_test)
y_test_mm = mm.fit_transform(y_test)

print("Training Shape", X_train_ss.shape, y_train_mm.shape)
print("Testing Shape", X_test_ss.shape, y_test_mm.shape)

##############transformtion to tensors#########################
X_train_tensors = Variable(torch.Tensor(X_train_ss))
X_test_tensors = Variable(torch.Tensor(X_test_ss))

y_train_tensors = Variable(torch.Tensor(y_train_mm))
y_test_tensors = Variable(torch.Tensor(y_test_mm))

###############reshaping to rows, timestamps, features##############

X_train_tensors_final = torch.reshape(X_train_tensors,   (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
#print(X_train_tensors_final)
X_test_tensors_final = torch.reshape(X_test_tensors,  (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

print("Training Shape", X_train_tensors_final.shape, y_train_tensors.shape)
print("Testing Shape", X_test_tensors_final.shape, y_test_tensors.shape)

###############our lstm class##############
lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1])
###############define optimizer##############
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()   # mean-squared error for regression
###############training ##############
for epoch in range(num_epochs):
    outputs = lstm1.forward(X_train_tensors_final) #forward pass
    #print("outputs",outputs)
    #print("y_train_tensors",y_train_tensors.view(-1,1))
    # obtain the loss function
    loss = criterion(outputs, y_train_tensors.view(-1,1))
    #print("loss",loss)
    #print("grad",lstm1.fc_1.weight.grad)
    optimizer.zero_grad() #caluclate the gradient, manually setting to 0
    loss.backward() #calculates the loss of the loss function
    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    #clipping_value = 1.0# arbitrary value of your choosing
    nn.utils.clip_grad_norm_(lstm1.parameters(), max_norm=2.0, norm_type=2)#torch.nn.utils.clip_grad_value_(lstm1.parameters(),clip_value=1.0)
    optimizer.step() #improve from loss, i.e backprop
    #for p in lstm1.parameters():
    #   p.register_hook(lambda grad: torch.clamp(grad, -clipping_value, clipping_value))
    #for p in lstm1.parameters():
    #   p.data.add_(-learning_rate, p.grad.data)#p.grad.data.clamp_(max=clipping_value)

    if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))



#reshaping the dataset
df_X_ss = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

train_predict = lstm1(df_X_ss)#forward pass
print(train_predict)
data_predict = train_predict.data.numpy() #numpy conversion
data_predict = mm.inverse_transform(data_predict)#reverse transformation
print('reverse transformation:',data_predict)
print('len data_predict',len(data_predict))
print('len res',res.shape[0])
###############update probality column in data frame#############
res['proba']=data_predict
###############update class column in data frame#############
classes=[]
for i in  range (res.shape[0]):
    if res.iloc[i][0]>=0.5:
        classes.append(1)
    else:
        classes.append(0)

res['class']=classes
print(res.head())
accuracy,precision=Accuracy(res)
print('accuracy',accuracy)
print('precision',precision)
res.to_csv(rf'FINAL DATA\PredictionsForMonth-{str(10)},Year-{str(2020)}.csv',index = True)
