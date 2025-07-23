import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd 
import datetime
import sys
sys.path.append('../utils')
from ML_utils import wrap_data

torch.backends.cudnn.benchmark = True

def ML_train(inp, numberx, numbery, path, offset = 1, activ = 'relu',
             valid = None, opt = 'adam', epoch = 1000, batch_size=32,
             patience=200, t = 4, skc = True): 
    monitor = 'loss' #(loss / val_loss)
    loss_monitor = 'loss' #(loss / val_loss)
    save_steps = 1
    log_step_freq = 1
    c_lin = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    inp = np.transpose(inp)
    inp = np.reshape(inp, (-1, 1, numberx, numbery), order = 'F')
    
    if skc:
        import ML_SkipAutoencoder as m
    else:
        import ML_NoneSkipAutoencoder as m

    lstm = m.LSTM(input_size=t, hidden_size=32, num_layers=5, output_size=t).to(device)
    enc = m.encoder(numberx, numbery, t, activ).to(device)
    dec = m.decoder(numberx, numbery, t, activ).to(device)
    model = m.autoencoder(enc, dec, t, offset, lstm).to(device)
    lstm = model.lstm.to(device)
        
    
    X_train, X_test = train_test_split(inp, test_size=0.205, shuffle=False)
    
    num_t = offset + 1
    
    X_train = wrap_data(X_train, num_t)
    X_test = wrap_data(X_test, num_t)
    
    X_train = torch.tensor(X_train)
    X_test = torch.tensor(X_test)
              
    data = TensorDataset(X_train)
    val_data = TensorDataset(X_test)
    
    X_train = DataLoader(
        data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    if valid:
        X_test = DataLoader(
            val_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    else:
        X_test = None
    
    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    scheduler = None
    epochs = epoch
    
    # Training
    best_steps = 0
    best_loss = float(np.Inf)
    if not (patience==None or patience==0):
        early_stopper = m.EarlyStopper(patience=patience, min_delta=0)

    dfhistory = pd.DataFrame(columns = ["epoch","loss","loss_rec","loss_lin",
                                        "loss_pred","val_loss","val_pred","lr"]) 

    print("Start Training...")
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("=========="*8 + "%s"%nowtime)
         
    for epoch in tqdm(range(1,epochs+1)):
        model.train()
        loss_total = .0
        for step, (data) in enumerate(X_train, 1): 
            x0 = data[0][:,:-1].to(device)
            x1 = data[0][:,1:].to(device)
            
            y0 = x0[:,0]
            y1 = model(y0)
            
            x1_list = []  
            Gx_list = []
            KGx_list = []
            for idx in range(offset):
                Gx = enc(x0[:,idx])
                Gx_list.append(Gx)
                KGx_list.append(enc(x1[:,idx]))
            
            KGx_list = torch.stack(KGx_list, dim=1)
            Gx_list = torch.stack(Gx_list, dim=1)
            
            Gx1_list = lstm(Gx_list)
            
            for idx in range(offset):
                x1_list.append(dec(Gx1_list[:,idx]))
            x1_list = torch.stack(x1_list, dim=1)

            l_rec = loss_fn(y0, y1)
            l_lin = loss_fn(KGx_list, Gx1_list)
            l_pred = loss_fn(x1_list, x1) 
                   
            loss = l_rec + l_pred +  c_lin*l_lin
            
            loss.backward()

            loss_total += loss.detach().item()
            optimizer.step()
            optimizer.zero_grad()
            
            lr = optimizer.param_groups[0]["lr"]

            print('\n')
            if step % log_step_freq == 0:   
                print(("[step = %d] loss: %.3f,") % (step, loss_total/step))
    
        if valid:
            model.eval()
            val_loss_total = .0
            for val_step, (val_x0, val_x1) in enumerate(X_test, 1):
                with torch.no_grad():
                    val_x0 = val_x0.to(device)
                    val_x1 = val_x1.to(device)
                    
                    val_y1 = model(val_x0)
                    
                    val_l_pred = loss_fn(val_x1, val_y1)
                    
                    if val_step % log_step_freq == 0: 
                        print('------validation------')
                        print(("[val_step = %d] val_loss: %.3f,") % (val_step, val_loss_total))
                    if 'mse' in path:
                        val_loss = val_l_pred
                    else:       
                        val_loss = val_l_pred
                val_loss_total += val_loss.detach().item()
                    
        else:
            val_loss_total = 0
            val_pred = 0    
        if valid:
            val_pred = val_l_pred.detach().item()
        loss_rec = l_rec.item()
        loss_lin = l_lin.item()
        loss_pred = l_pred.item()

        info = (epoch, loss_total/step, loss_rec, loss_lin,
                loss_pred, val_loss_total/step,
                val_pred, lr)

        dfhistory.loc[epoch-1] = info
        
        if epoch % log_step_freq == 0:   

            print(("\nEPOCH = %d, loss = %.5f," " loss_rec = %.5f, " " loss_lin = %.5f, "
                    " loss_pred = %.5f, "  " val_loss = %.5f, " " val_pred = %.5f, "
                    " lr = %.5f, ") %info)

        if(not scheduler is None):
            scheduler.step()
            if(epoch%10 == 0):
                for param_group in optimizer.param_groups:
                    print('Epoch {:d}; Learning-rate: {:0.05f}'.format(epoch, param_group['lr']))
        
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if valid and loss_monitor == 'val_loss':
            monitor = val_loss_total
        else:
            monitor = loss_total
            
        if monitor < best_loss:
            best_steps += 1
            if best_steps % save_steps == 0:
                best_loss = monitor
                print('------Save model:best loss is %.3f------' %best_loss)
                torch.save(model.state_dict(), path+'ckpt.pth')
            
        # early stopping
        if not (patience==None or patience==0):
            if early_stopper.early_stop(monitor):       
                print('------Early stopping------')
                break

        print("\n"+"=========="*6 + "%s"%nowtime)
    print('Finished Training...')
    model.load_state_dict(torch.load(path+'ckpt.pth'))
    torch.save(model, path+'autoencoder.pth')
    torch.save(lstm, path+'lstm.pth')
    
    dfhistory.to_csv(path+'training_log.csv', sep=',')
    torch.cuda.empty_cache()
    return model, enc, dec, lstm
 

def encoder_pred(ori, numberx, numbery, model):
    ori = np.transpose(ori)
    ori = np.reshape(ori, (-1, 1, numberx, numbery), order = 'F')
    try:
        pred = model(torch.tensor(ori)).detach().numpy()
    except:
        pred = model(torch.tensor(ori).to('cuda')).detach()
        pred = pred.cpu().numpy()
    pred = np.transpose(pred)
    return pred

def decoder_pred(ori, numberx, numbery, model):
    ori = np.transpose(ori)
    try:
        pred = model(torch.tensor(ori)).detach().numpy()
    except:
        pred = model(torch.tensor(ori).to('cuda')).detach()
        pred = pred.cpu().numpy()
    pred = np.reshape(pred, (-1, numberx*numbery), order = 'F')
    pred = np.transpose(pred)
    return pred



