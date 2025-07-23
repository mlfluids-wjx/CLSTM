import numpy as np
import torch
import torch.nn as nn

class encoder(nn.Module):
    def __init__(self, numberx=192, numbery=96, t=4, activ='tanh'):
        super(encoder, self).__init__()
        fs = 3
        lc = np.logspace(0,5,6,base=1/2)*32
        lx = np.logspace(0,5,6,base=1/2)*numberx
        ly = np.logspace(0,5,6,base=1/2)*numbery
        shape = []
        for i in range(6):
            shape.append(int(lx[i]*ly[i]*lc[i]))

        self.lc = [int(_) for _ in lc]
        self.lx = [int(_) for _ in lx]
        self.ly = [int(_) for _ in ly]
        self.shape = [int(_) for _ in shape]
        
        self.conv1 = nn.Conv2d(1, self.lc[0], fs, padding='same')
        self.conv2 = nn.Conv2d(self.lc[0], self.lc[1], fs, padding='same')
        self.conv3 = nn.Conv2d(self.lc[1], self.lc[2], fs, padding='same')
        self.conv4 = nn.Conv2d(self.lc[2], self.lc[3], fs, padding='same')
        self.conv5 = nn.Conv2d(self.lc[3], self.lc[4], fs, padding='same')
        self.conv6 = nn.Conv2d(self.lc[4], self.lc[5], fs, padding='same')
        
        self.ln1 = nn.LayerNorm([self.lc[0], self.lx[0], self.ly[0]])
        self.ln2 = nn.LayerNorm([self.lc[1], self.lx[1], self.ly[1]])
        self.ln3 = nn.LayerNorm([self.lc[2], self.lx[2], self.ly[2]])
        self.ln4 = nn.LayerNorm([self.lc[3], self.lx[3], self.ly[3]])
        self.ln5 = nn.LayerNorm([self.lc[4], self.lx[4], self.ly[4]])
        self.ln6 = nn.LayerNorm([self.lc[5], self.lx[5], self.ly[5]])
        
        self.fn = nn.LayerNorm([t])
        
        self.rs = nn.Flatten()
        self.id = nn.Identity()
        
        self.fc1 = nn.Linear(self.shape[0], t)
        self.fc2 = nn.Linear(self.shape[1], t)
        self.fc3 = nn.Linear(self.shape[2], t)
        self.fc4 = nn.Linear(self.shape[3], t)
        self.fc5 = nn.Linear(self.shape[4], t)
        self.fc6 = nn.Linear(self.shape[5], t)
        
        self.pool = nn.MaxPool2d(2)
        if activ == 'tanh':
            self.t = nn.Tanh()
        elif activ == 'relu':
            self.t = nn.ReLU(True)
              
    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = self.ln1(conv1)
        conv1 = self.t(conv1)
        fc1 = self.rs(conv1)
        fc1 = self.fc1(fc1)
        fc1 = self.fn(fc1)
        fc1 = self.t(fc1)
        conv1 = self.pool(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.ln2(conv2)
        conv2 = self.t(conv2)
        fc2 = self.rs(conv2)
        fc2 = self.fc2(fc2)
        fc2 = self.fn(fc2)
        fc2 = self.t(fc2)
        conv2 = self.pool(conv2)
        conv3 = self.conv3(conv2)
        conv3 = self.ln3(conv3)
        conv3 = self.t(conv3)
        fc3 = self.rs(conv3)
        fc3 = self.fc3(fc3)
        fc3 = self.fn(fc3)
        fc3 = self.t(fc3) 
        conv3 = self.pool(conv3)
        conv4 = self.conv4(conv3)
        conv4 = self.ln4(conv4)
        conv4 = self.t(conv4)
        fc4 = self.rs(conv4)
        fc4 = self.fc4(fc4)
        fc4 = self.fn(fc4)
        fc4 = self.t(fc4)
        conv4 = self.pool(conv4)
        conv5 = self.conv5(conv4)
        conv5 = self.ln5(conv5)
        conv5 = self.t(conv5)
        fc5 = self.rs(conv5)
        fc5 = self.fc5(fc5)
        fc5 = self.fn(fc5)
        fc5 = self.t(fc5)
        conv5 = self.pool(conv5)
        conv6 = self.conv6(conv5)
        conv6 = self.ln6(conv6)
        conv6 = self.t(conv6)
        fc6 = self.rs(conv6)
        fc6 = self.fc6(fc6)
        fc6 = self.fn(fc6)
        fc6 = self.t(fc6)
        
        encoded = fc1+fc2+fc3+fc4+fc5+fc6
        encoded /= 6
        del conv1, conv2, conv3, conv4, conv5, conv6
        return encoded
                
        
class decoder(nn.Module):
    def __init__(self, numberx=192, numbery=96, t=4, activ='tanh'):
        super(decoder, self).__init__()
        fs = 3
        lc = np.logspace(0,5,6,base=1/2)*32
        lx = np.logspace(0,5,6,base=1/2)*numberx
        ly = np.logspace(0,5,6,base=1/2)*numbery
        shape = []
        for i in range(6):
            shape.append(int(lx[i]*ly[i]*lc[i]))

        self.lc = [int(_) for _ in lc]
        self.lx = [int(_) for _ in lx]
        self.ly = [int(_) for _ in ly]
        self.shape = [int(_) for _ in shape]
        self.conv6t = nn.Conv2d(self.lc[5], self.lc[4], fs, padding='same')
        self.conv5t = nn.Conv2d(self.lc[4]+self.lc[4], self.lc[3], fs, padding='same')
        self.conv4t = nn.Conv2d(self.lc[3]+self.lc[3], self.lc[2], fs, padding='same')
        self.conv3t = nn.Conv2d(self.lc[2]+self.lc[2], self.lc[1], fs, padding='same')
        self.conv2t = nn.Conv2d(self.lc[1]+self.lc[1], self.lc[0], fs, padding='same')
        self.conv1t = nn.Conv2d(self.lc[0]+self.lc[0], 1, fs, padding='same')
        
        self.rs = nn.Flatten()
        self.id = nn.Identity()
        
        self.ln1 = nn.LayerNorm([self.lc[0]*2, self.lx[0], self.ly[0]])
        self.ln2 = nn.LayerNorm([self.lc[0], self.lx[1], self.ly[1]])
        self.ln3 = nn.LayerNorm([self.lc[1], self.lx[2], self.ly[2]])
        self.ln4 = nn.LayerNorm([self.lc[2], self.lx[3], self.ly[3]])
        self.ln5 = nn.LayerNorm([self.lc[3], self.lx[4], self.ly[4]])
        self.ln6 = nn.LayerNorm([self.lc[4], self.lx[5], self.ly[5]])
        
        self.fn6 = nn.LayerNorm([self.shape[5]])
        self.fn5 = nn.LayerNorm([self.shape[4]])
        self.fn4 = nn.LayerNorm([self.shape[3]])
        self.fn3 = nn.LayerNorm([self.shape[2]])
        self.fn2 = nn.LayerNorm([self.shape[1]])
        self.fn1 = nn.LayerNorm([self.shape[0]])
        
        self.fc6 = nn.Linear(t, self.shape[5])
        self.fc5 = nn.Linear(t, self.shape[4])
        self.fc4 = nn.Linear(t, self.shape[3])
        self.fc3 = nn.Linear(t, self.shape[2])
        self.fc2 = nn.Linear(t, self.shape[1])
        self.fc1 = nn.Linear(t, self.shape[0])
        
        self.upsamp = nn.Upsample(scale_factor=(2,2))
        if activ == 'tanh':
            self.t = nn.Tanh()
        elif activ == 'relu':
            self.t = nn.ReLU(True)

    def forward(self, x):
        fc6 = self.fc6(x)
        fc6 = self.fn6(fc6)
        fc6 = self.t(fc6)
        fc6 = fc6.view(-1,self.lc[5],self.lx[5],self.ly[5])
        fc5 = self.fc5(x)
        fc5 = self.fn5(fc5)
        fc5 = self.t(fc5)
        fc5 = fc5.view(-1,self.lc[4],self.lx[4],self.ly[4])
        fc4 = self.fc4(x)
        fc4 = self.fn4(fc4)
        fc4 = self.t(fc4)
        fc4 = fc4.view(-1,self.lc[3],self.lx[3],self.ly[3])
        fc3 = self.fc3(x)
        fc3 = self.fn3(fc3)
        fc3 = self.t(fc3)
        fc3 = fc3.view(-1,self.lc[2],self.lx[2],self.ly[2])
        fc2 = self.fc2(x)
        fc2 = self.fn2(fc2)
        fc2 = self.t(fc2)
        fc2 = fc2.view(-1,self.lc[1],self.lx[1],self.ly[1])
        fc1 = self.fc1(x)
        fc1 = self.fn1(fc1)
        fc1 = self.t(fc1)
        fc1 = fc1.view(-1,self.lc[0],self.lx[0],self.ly[0])
        xx = self.conv6t(fc6)
        xx = self.ln6(xx)
        xx = self.t(xx)
        xx = self.upsamp(xx)
        xx = torch.cat((xx, fc5), 1)
        xx = self.conv5t(xx)
        xx = self.ln5(xx)
        xx = self.t(xx)
        xx = self.upsamp(xx)
        xx = torch.cat((xx, fc4), 1)
        xx = self.conv4t(xx)
        xx = self.ln4(xx)
        xx = self.t(xx)
        xx = self.upsamp(xx)
        xx = torch.cat((xx, fc3), 1)
        xx = self.conv3t(xx)
        xx = self.ln3(xx)
        xx = self.t(xx)
        xx = self.upsamp(xx)
        xx = torch.cat((xx, fc2), 1)
        xx = self.conv2t(xx)
        xx = self.ln2(xx)
        xx = self.t(xx)
        xx = self.upsamp(xx)
        xx = torch.cat((xx, fc1), 1)
        xx = self.conv1t(xx)
        return xx


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1 
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        pred = self.linear(output)
        return pred

class autoencoder(nn.Module):
    def __init__(self, encoder, decoder, t=4, offset=1, lstm=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lstm = lstm
        self.offset = offset
          
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x  
    
    def weights_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

class EarlyStopper():
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False