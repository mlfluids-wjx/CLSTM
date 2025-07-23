import numpy as np
import os
import ML_utils as ut
import torch
from sklearn import preprocessing

case = 'cylinder'
method = 'C-LSTM'  
backend = 'Torch' 

# Hyperparamss
rank = 3
offset = 5
activ = 'tanh'
patience = 500 
valid = False
skc =  False 
seed = 1
epoch = 2000
batch_size = 32

# Pre&post-process
normalize = True

os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)

# Load data
numberx, numbery = 192, 96
S_full = np.load('Cylinder_interpolate192x96y_order_C.npy')
S_full = np.float32(S_full)
S_col = 120
S_col_full = 151
ob = 'vorticity'
snapshots = list(S_full.T)
projName = '{}_{}_{}_{}'.format(method, backend, case, ob)

# Run script
if 'LSTM' in method:
    if skc == False:       
        projName = 'Noskc_' + projName
       
if normalize:
    scaler = preprocessing.MinMaxScaler(copy=True, clip=True)
    S_full = scaler.fit_transform(S_full)

S_train = S_full[:, :S_col]
S_test = S_full[:, S_col:]
          
path = './ML_' + projName +'_'+ activ + '/'  

if not os.path.exists(path):
    os.makedirs(path) 
writepath = path + '/output_data'
if not os.path.exists(writepath):
    os.makedirs(writepath) 
     
if 'LSTM' in method:
    import ML_CLSTM as ML_C 
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
    autoencoder, encoder, decoder, lstm = \
    ML_C.ML_train(S_full, numberx, numbery, path, offset, activ, 
                    valid=valid, opt='adam', epoch=epoch, batch_size=batch_size, 
                    patience=patience, t=rank, skc=skc)
    
    encoded_true = ML_C.encoder_pred(S_full, numberx, numbery, encoder) 
    encoded_ori = ML_C.encoder_pred(S_train, numberx, numbery, encoder)

    encoded = encoded_ori
    encoded_pred = np.zeros((encoded_ori.shape[0],S_col_full+offset), dtype=np.float32)
    
    encoded = ut.wrap_data(encoded_ori.transpose(), offset)
    encoded = torch.from_numpy(encoded).to('cuda')
    encoded_pred = encoded_pred.transpose()
    offset = 1
    for mm, i in enumerate(range(S_col, S_col_full, offset), 0): 
        op_num = mm + 1
        print('lstm cycle:{}, {}-{} to {}-{}'.format(op_num,mm,i,op_num*offset,i+offset))
        y1 = lstm(encoded)
        encoded_pred[i:i+offset,:] = y1[-offset,-offset,:].detach().cpu().numpy()  
        y1 = torch.cat((encoded[:,offset:], y1[:,-offset:]), 1) 
        encoded = y1
    encoded_pred = encoded_pred.transpose()   

    encoded_pred = encoded_pred[:,:S_col_full]
    encoded_pred[:,:S_col] = encoded_ori

    S_pred = ML_C.decoder_pred(encoded_pred, numberx, numbery, decoder)
    S_recon = ML_C.decoder_pred(encoded_true, numberx, numbery, decoder)

    snapshots2 = list(np.transpose(S_pred))

print('Task completes')
print('data path: {}'.format(path))