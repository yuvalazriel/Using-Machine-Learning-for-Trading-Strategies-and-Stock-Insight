import torch

import copy
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split

from torch import nn, optim

import torch.nn.functional as F
from arff2pandas import a2p
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_dataset(df):
    sequences = df.astype(np.float32).to_numpy().tolist()
    dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
    n_seq, seq_len, n_features = torch.stack(dataset).shape
    return dataset, seq_len, n_features

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )
    def forward(self, x):
        #print("in encoder")
        x = x.reshape((1, self.seq_len, self.n_features))
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        return hidden_n.reshape((self.n_features, self.embedding_dim))


class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)
    def forward(self, x):
        #print("in decoder")
        x = x.repeat(self.seq_len, self.n_features)
        x = x.reshape((self.n_features, self.seq_len, self.input_dim))
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((self.seq_len, self.hidden_dim))
        return self.output_layer(x)

class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)
    def forward(self, x):
        #print("encoder")
        x = self.encoder(x)
        #print("decoder")
        x = self.decoder(x)

        return x

def train_model(model, train_dataset, val_dataset, n_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss(reduction='sum').to(device)
    history = dict(train=[], val=[])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    for epoch in range(1, n_epochs + 1):
        print("######################################################################epoch-",epoch)
        model = model.train()
        print("model.train()-")
        train_losses = []
        for seq_true in train_dataset:
            #print("######################################################################for-train_dataset")
            optimizer.zero_grad()
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        val_losses = []
        model = model.eval()
        print("val_dataset()-")
        with torch.no_grad():
            for seq_true in val_dataset:
                seq_true = seq_true.to(device)
                seq_pred = model(seq_true)
                loss = criterion(seq_pred, seq_true)
                val_losses.append(loss.item())
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
    model.load_state_dict(best_model_wts)
    return model.eval(), history

def predict(model, dataset):
    predictions, losses = [], []
    criterion = nn.L1Loss(reduction='sum').to(device)
    with torch.no_grad():
        model = model.eval()
        for seq_true in dataset:
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            predictions.append(seq_pred.cpu().numpy().flatten())
            losses.append(loss.item())
    return predictions, losses


'''
fullClass=pd.read_csv(rf'FINAL DATA\classForYear-1981 to Year-2020.csv')
fullTable=pd.read_csv(rf'FINAL DATA\vectorForYear-1981 to Year-2020.csv')
fullTable=fullTable.drop(fullTable.columns[[0]],1)#drop the first column
fullTable=fullTable.fillna(0)

fullClass=pd.read_csv(rf'FINAL DATA\classForYear-1981 to Year-2020.csv')
fullTable=pd.read_csv(rf'FINAL DATA\vectorForYear-1981 to Year-2020.csv')
fullTable["class"]=fullClass.iloc[:,2]
print("done- shape",fullTable.shape[0])
print("table+class",fullTable.head)
fullTable.to_csv(rf'FINAL DATA\TableAndClassForYear-1981 to Year-2020.csv', index=True)
#print("done")

fullClass=pd.read_csv(rf'FINAL DATA\classForYear-1981 to Year-2020.csv')
fullTable=pd.read_csv(rf'FINAL DATA\vectorForYear-1981 to Year-2020.csv')
fullTable=fullTable.drop(fullTable.columns[[0]],1)#drop the first column
fullTable["class"]=fullClass.iloc[:,2]
print("done- shape",fullTable.shape[0])
print("table+class",fullTable.head)
fullTable.to_csv(rf'FINAL DATA\TableAndClassForYear-1981 to Year-2020.csv', index=True)
#print("done")
'''
fullTable=pd.read_csv(rf'../FINAL DATA/TableAndClassForYear-1981 to Year-2020.csv')
print("shape",fullTable.shape[1])
fullTable=fullTable.drop(fullTable.columns[[0]],1)#drop the first column
fullTable=fullTable.drop(fullTable.columns[[0]],1)#drop the second column
print("shape",fullTable.shape[1])
df = fullTable.drop(fullTable.columns[[-1]], axis=1)
print(df.head())
print("shape",df.shape[1])
train_df, val_df = train_test_split(df,test_size=0.15,random_state=RANDOM_SEED)
#from val_df
val_df, test_df = train_test_split(val_df,test_size=0.33,random_state=RANDOM_SEED)

train_dataset, seq_len, n_features = create_dataset(train_df)
val_dataset, _, _ = create_dataset(val_df)
test_dataset, _, _ = create_dataset(test_df)

print("train_dataset,seq_len, n_features", seq_len, n_features,len(train_dataset))

model = RecurrentAutoencoder(seq_len, n_features, 128)
print("before-device")
model = model.to(device)
print("after-device")
model, history = train_model(model, train_dataset,val_dataset,n_epochs=150)
print("after-train")
MODEL_PATH = 'model.pth'
torch.save(model, MODEL_PATH)
#Uncomment the next lines, if you want to download and load the pre-trained model:
# !gdown --id 1jEYx5wGsb7Ix8cZAw3l5p5pOwHs3_I9A
# model = torch.load('model.pth')
# model = model.to(device)

_, losses = predict(model, train_dataset)
#sns.distplot(losses, bins=50, kde=True);

THRESHOLD = 26
predictions, pred_losses = predict(model, test_df)

correct = sum(l <= THRESHOLD for l in pred_losses)
print(f'Correct normal predictions: {correct}/{len(test_df)}')
