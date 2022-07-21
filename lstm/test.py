import torch
import pytorch_lightning as pl
from torch.optim import optimizer
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset
from lstm import LSTMModel
import numpy as np

Z =  torch.from_numpy(np.load("Z.npy"))
R = torch.from_numpy(np.load("R.npy"))
tmp = torch.from_numpy(np.load("x.npy"))
xh = tmp[:,:,:3]
x = tmp[:,:,:3]
y = tmp[:,:,3:]

dataset = TensorDataset(Z, R, xh,x,y)

n_samples = len(dataset)

n_train = int(n_samples*0.8)
n_val = int(n_samples*0.1)
n_test = n_samples - n_train - n_val

train_set, val_set, test_est = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(3543))

batch_size = 256

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_est, batch_size=batch_size, shuffle=False, pin_memory=True)

class LitHipnnLSTM(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = LSTMModel(n_in=3, n_out=7)
    
    def forward(self, Z, R, x):
        output, nonblank = self.model(Z, R, x)
        return output, nonblank
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1.0e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        Z, R, xh, x, y = batch
        y_hat, nonblank = self.model(Z, R, xh, x)
        loss = torch.nn.functional.mse_loss(y_hat, y[nonblank], reduction='mean')
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_indx):
        Z, R, x, xh, y = batch
        y_hat, nonblank = self.model(Z, R, xh, x)
        loss = torch.nn.functional.mse_loss(y_hat, y[nonblank], reduction='mean')
        self.log("val_loss", loss)
        return loss

model = LitHipnnLSTM()
trainer = pl.Trainer(precision=64)
trainer.fit(model, train_loader, val_loader)