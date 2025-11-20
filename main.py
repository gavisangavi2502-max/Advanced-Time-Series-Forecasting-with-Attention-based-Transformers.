
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Minimal illustrative Transformer forecasting pipeline
class SeqDataset(Dataset):
    def __init__(self, data, in_win=24, out_win=12):
        self.in_win=in_win; self.out_win=out_win
        self.X=[]; self.y=[]
        for i in range(len(data)-in_win-out_win):
            self.X.append(data[i:i+in_win])
            self.y.append(data[i+in_win:i+in_win+out_win])
        self.X=np.array(self.X); self.y=np.array(self.y)

    def __len__(self): return len(self.X)
    def __getitem__(self,idx): 
        return torch.tensor(self.X[idx],dtype=torch.float32), torch.tensor(self.y[idx],dtype=torch.float32)

class TransformerModel(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model,nhead), num_layers)
        self.decoder = nn.Linear(d_model,1)

    def forward(self,x):
        x = x.permute(1,0,2)
        out = self.encoder(x)
        out = self.decoder(out[-1])
        return out

def main():
    data = np.sin(np.arange(1000)/20)
    ds = SeqDataset(data)
    dl = DataLoader(ds,batch_size=32,shuffle=True)

    model = TransformerModel()
    opt = torch.optim.Adam(model.parameters(),lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(2):
        for X,y in dl:
            pred = model(X)
            loss = loss_fn(pred.squeeze(), y[:,0])
            opt.zero_grad(); loss.backward(); opt.step()
    print("Training complete.")

if __name__=="__main__":
    main()
