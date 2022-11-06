# snntorch
import snntorch as snn
import snntorch.functional as SF 
from snntorch import spikegen
import torch.nn.functional as F

# torch
import torch.nn as nn
import torch
from torch.utils.data import DataLoader,TensorDataset

# optuna
import optuna
import joblib

# misc
import numpy as np
import json
import os
import os.path
import glob
import random
import itertools
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
import logging
import time

# local imports
# utils imports
# from utils.load_signals_CHBMIT import PrepData
from utils.prep_data import train_val_loo_split, train_val_test_split
from utils.prep_data_TUH import load_evalute, train_val_split_file_list,train_val_split_diff_patient_file_list
from utils.log import log
from myio.save_load import write_out
from utils.early_stoping import EarlyStopping
from utils.seiz_data_loader import SeizDataset

#utils optuna imports
from utils.optuna.conf import *
from utils.optuna.printer import *
from utils.optuna.multiple_pruner import *
from utils.w_loss import mse_membrane_loss


#model imports
# from models.SNN_new import Net
from models.SNN_convlstm_3_test import Net


def normal(x):
    normalized = (x-np.min(x))/(np.max(x)-np.min(x))+1e-6

    return normalized

def makedirs(dir):
    try:
        os.makedirs(dir)
    except:
        pass


def print_epoch_AUC(data_loader, SNN_net,data_size, device, data_type,batch_size):

    y_label = np.empty((data_size), dtype=int)
    y_predict = np.empty((data_size), dtype=float)

    for i_batch, sample_batched in enumerate(data_loader):

        labels = sample_batched['landmarks'].to(device,dtype=torch.int)
        images = sample_batched['image'].to(device,dtype=torch.float)
        
        outputs, test_mem_rec = SNN_net(images.permute(1,0,2,3))
        _, predicted = outputs.sum(dim=0).max(1)

        y_label[i_batch*batch_size:(i_batch+1)*batch_size] = labels.cpu().numpy()
        y_predict[i_batch*batch_size:(i_batch+1)*batch_size] = predicted.cpu().numpy()
    fpr, tpr, thresholds = metrics.roc_curve(y_label, y_predict) 

    result = metrics.auc(fpr, tpr)
    print(data_type+' AUC: ', result)
    return result

def avg_loss(train_losses, test_losses, avg_train_losses, avg_test_losses):
    train_loss = np.average(train_losses)
    test_loss = np.average(test_losses)

    avg_train_losses.append(train_loss)
    avg_test_losses.append(test_loss)

    # scheduler.step()
    train_losses = []
    test_losses = []

    return avg_train_losses, avg_test_losses, train_loss, test_loss

##########function for printout the train/test accuracy#######################
def main(dataset, build_type, fusion, sph):
    print ('Main')
    with open('SETTINGS_%s.json' %dataset) as f:
        settings = json.load(f)
    makedirs(str(settings['cachedir']))
    # makedirs(str(settings['ckptdir']))
    makedirs(str(settings['resultdir']))

    if settings['dataset'] == 'CHBMIT':
        # skip Patient 12, not able to read
        targets = [
            # '1',
            # '2',
            # '3',
            '4',
            # '5',
            # '6',
            # '7',
            # '8',
            # '9',
            # '10',
            # '11',
            # '12',
            # '13',
            # '14',
            # '15',
            # '16',
            # '17',
            # '18',
            # '19',
            # '20',
            # '21',
            # '22',
            # '23'
        ]
    elif settings['dataset'] == 'FB':
        targets = [
            '1',
            '3',
            '4',
            '5',
            '6',
            '14',
            '15',
            '16',
            '17',
            '18',
            '19',
            '20',
            '21',
        ]
    elif settings['dataset'] == 'Kaggle2014Det':
        targets = [
            'Dog_1',
            'Dog_2',
            'Dog_3',
            'Dog_4',
            'Patient_1',
            'Patient_2',
            'Patient_3',
            'Patient_4',
            'Patient_5',
            'Patient_6',
            'Patient_7',
            'Patient_8',

        ]

    elif settings['dataset']=='EpilepsiaSurf':
        targets = [
            #'1',
             #'2',
             #'3',
            # '4',
            # '5',
            # '6',
            # '7',
            # '8',
            # '9',
            # '10',
             '11',
            # '12',
            # '13',
            # '14',
            # '15',
            # '16',
            # '17',
            # '18',
            # '19',
            # '20',
            # '21',
            # '22',
            # '23',
            # '24',
            # '25',
            # '26',
            # '27',
            # '28',
            # '29',
            # '30'
        ]
    elif settings['dataset']=='TUH':
        targets = [
            'all',
        ]


    summary = {}
    epochs = 20
    batch_size = 128
    for target in targets:

        if settings['dataset'] == 'TUH':
            # load the training data ans split to 80-20%
            # train_path = settings["datadir_train"]
            train_path = "/mnt/data9_NAS/TUH_ICA_EEG1216s"
            train_path2 = "/mnt/data9_NAS/TUH_ICA_EEG1216s_v2"

            train_dir, val_dir,class_weight = train_val_split_diff_patient_file_list(train_path,train_path2, val_ratio=0.10)
            train_dir = train_dir[:1000]
            val_dir = val_dir[:1000]
            # load the TUH dataloader
            Tuh_train_data = SeizDataset(root_dir=train_dir)
            Tuh_val_data = SeizDataset(root_dir=val_dir)

            # load the test data if needed
            # test_path = settings["datadir_dev"]
            test_path = "/mnt/data9_NAS/tuh_stft_ica_dev12s"
            test_dir = glob.glob(test_path + "/*.npy")[:1000]
            Tuh_test_data = SeizDataset(root_dir= test_dir)

            #Print out the information
            print('train size:',len( train_dir),' val size: ',len(val_dir))
            train_size = len(train_dir) - len(train_dir)%batch_size
            val_size = len(val_dir)  - len(val_dir)%batch_size
            test_size = len(test_dir) - len(test_dir)%batch_size
            print('train size:',train_size,' val size: ',val_size, ' test size:', test_size)
       
        if build_type=='dect':

            if settings['dataset'] == 'TUH':
                # get the dataloader
                train_loader = DataLoader(Tuh_train_data, batch_size=batch_size,shuffle=True,drop_last=True)
                val_loader = DataLoader(Tuh_val_data, batch_size=batch_size,shuffle=False,drop_last=True)
                test_loader = DataLoader(Tuh_test_data, batch_size=batch_size, shuffle=False,drop_last=True)
                print(train_loader,val_loader,test_loader)
            # transform to torch tensor
            #print(y_test[0])
            ###### training parameter to use ######################
            device = torch.device("cuda")
            beta = 0.9181805491303656
            snn.slope = 13.42287274232855
            # thr = 0.17301711062043745
            thr = 0.05
            # time_step = 1e-3
            # tau_mem = 6.5e-4
            # tau_syn = 5.5e-4
            # alpha = float(np.exp(-time_step / tau_syn))
            # beta = float(np.exp(-time_step / tau_mem))

            SNN_net = Net().to(device)
            dtype = torch.float
            #######define the loss function########################
            optimizer = torch.optim.Adam(SNN_net.parameters(), lr=0.0003950327143842849, betas=(0.9, 0.999))

            on_target = thr + thr * 0.2
            off_target = thr * 0.2
            loss_fn = mse_membrane_loss(on_target=on_target, off_target=off_target, weight=class_weight)
            #loss_fn = nn.MSELoss()
            #######define the loss function########################
            ###### training parameter to use ######################
            
            
            early_stopping = EarlyStopping(patience=50, verbose=True)
            early_stopping.early_stop = False
            early_stopping.best_score = None
            avg_train_losses = []
            avg_test_losses = []
            #########for training#################################
            loss_hist = []
            val_loss_hist = []
            counter = 0
            # Outer training loop
            for epoch in range(epochs):
                minibatch_counter = 0
                train_batch = iter(train_loader)

                for i_batch, sample_batched in enumerate(train_loader):
                    
                    targets_it = sample_batched['landmarks'].to(device,dtype=dtype)
                    data_it = sample_batched['image'].to(device,dtype=dtype)
                    output, mem_rec = SNN_net(data_it.permute(1,0, 2, 3))  # permute to num_step x batch x num_elec x freq

                    loss_val = loss_fn(mem_rec, targets_it)

                    # Gradient calculation
                    optimizer.zero_grad()
                    # loss_val.backward(retain_graph=True)
                    loss_val.backward()

                    # Weight Update
                    nn.utils.clip_grad_norm_(SNN_net.parameters(), 1)
                    optimizer.step()

                    # Store loss history for future plotting
                    loss_hist.append(loss_val.item())

                    # val set
                    if i_batch !=0 and i_batch%5 ==0:
                        print('Batch: '+ str(i_batch))
                    # val set
                        for val_batch, val_sample_batched in enumerate(val_loader):
                            valdata_it = val_sample_batched['image'].to(device,dtype=dtype)
                            valtargets_it = val_sample_batched['landmarks'].to(device,dtype=dtype)

                            # val set forward pass
                            val_output, val_mem_rec = SNN_net(valdata_it.permute(1,0,2,3))

                            # val set loss
                            loss_val_test = loss_fn(val_mem_rec, valtargets_it)

                            val_loss_hist.append(loss_val_test.item())

                            minibatch_counter += 1
                            counter += 1
                    
                        avg_train_losses, avg_test_losses, train_loss, test_loss = avg_loss(loss_hist, val_loss_hist,
                                                                                            avg_train_losses,
                                                                                            avg_test_losses)
                        epoch_len = len(str(epochs))

                        print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                                        f'train_loss: {train_loss:.5f} ' +
                                        f'test_loss: {test_loss:.5f}')


                        print(print_msg)
                        train_AUC= print_epoch_AUC(train_loader,SNN_net, train_size, device, 'train ',batch_size)
                        val_AUC = print_epoch_AUC(val_loader, SNN_net, val_size, device, 'val',batch_size)
                        test_AUC = print_epoch_AUC(test_loader, SNN_net,test_size, device, 'test', batch_size)
                        early_stopping(test_AUC, SNN_net)

                        if early_stopping.early_stop:
                            print("Early stopping")
                            early_stopping.early_stop = False
                            early_stopping.best_score = None
                            break

                        loss_hist_true_grad = loss_hist
                        test_loss_hist_true_grad = val_loss_hist

                        total = 0
                        correct = 0

            with torch.no_grad():
                SNN_net = Net().to(device)
                SNN_net.eval()
                test_AUC = print_epoch_AUC(test_loader, test_size, device, 'test', beta)

            
            
            print('AUC: ',test_AUC )
            

            

    print (summary)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="TUH",
                        help="FB, CHBMIT or Kaggle2014Det")
    parser.add_argument("--mode", default='dect',
                        help="cv or test. cv is for leave-one-out cross-validation")
    parser.add_argument("--sph", type=int, default=5,
                        help="seizure prediction horizon in seconds")
    parser.add_argument('--fusion', dest='fusion', action='store_true')
    parser.add_argument('--no-fusion', dest='fusion', action='store_false')
    parser.set_defaults(fusion=False)

    args = parser.parse_args()
    assert args.dataset in ["FB", "CHBMIT", "Kaggle2014Det", "EpilepsiaSurf","TUH"]
    assert args.mode in ['pred','dect']

    log('********************************************************************')
    log('--- START --dataset %s --mode %s --sph %s ---'
        %(args.dataset,args.mode,args.sph))
    if args.fusion == True:
        print ('Using data fusion!')
    else:
        print ('Using EEG signals only!')
    main(
        dataset=args.dataset,
        build_type=args.mode,
        fusion=args.fusion,
        sph=args.sph)
