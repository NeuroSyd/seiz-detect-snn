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
import gc
# local imports
# utils imports
# from utils.load_signals_CHBMIT import PrepData
from utils.prep_data import train_val_loo_split, train_val_FB_loo_split,train_val_test_split
from utils.prep_data_TUH import load_evalute, train_val_split_file_list,train_val_split_diff_patient_file_list
from utils.load_data_FB_kaggle.load_signals import PrepData
from utils.log import log
from myio.save_load import write_out
from utils.early_stoping import EarlyStopping
from utils.seiz_data_loader import SeizDataset
#load iEEG data
from utils.load_data_ieeg.Load_signal import LoadSignalsACS

#utils optuna imports
from utils.optuna.conf import *
from utils.optuna.printer import *
from utils.optuna.multiple_pruner import *
from utils.w_loss import mse_membrane_loss


#model imports
# from models.SNN_new import Net
# from models.iEEG.pat1.SNN_convlstm_3 import Net
from models.iEEG.FB.SNN_convlstm_3 import Net

def find_largest_divide_num(num):
    for i in range(1, 350):
        if num % i == 0:
            #4
            largest_divisor = i
    print("Largest divisor of {} is {}".format(num,largest_divisor))
    return largest_divisor


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

    test_batch = iter(data_loader)
    i_batch = 0
    # Minibatch training loop
    for data_it, targets_it in test_batch:

        labels = targets_it.to(device,dtype=torch.int)
        images = data_it.to(device,dtype=torch.float)
        
        outputs, test_mem_rec = SNN_net(images.permute(1,0,2,3))
        _, predicted = outputs.sum(dim=0).max(1)

        y_label[i_batch*batch_size:(i_batch+1)*batch_size] = labels.cpu().numpy()
        y_predict[i_batch*batch_size:(i_batch+1)*batch_size] = predicted.cpu().numpy()
        i_batch +=1
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
def main(dataset, build_type):
    print ('Main')
    if args.dataset == 'EpilepsiaSurf_iEEG':
        with open('utils/load_data_ieeg/SETTINGS_%s.json' %args.dataset) as f:
            settings = json.load(f)
    elif args.dataset == 'FB' or args.dataset == 'CHBMIT':
        with open('utils/load_data_FB_kaggle/SETTINGS_%s.json' %args.dataset) as f:
            settings = json.load(f)

    if settings['dataset'] == 'FB':
        # skip Patient 12, not able to read
        targets = [
            '1',
            # '2',
            '3',
            '4',
            '5',
            '6',
            # '7',
            # '8',
            # '9',
            # '10',
            # '11',
            # '12',
            # '13',
            '14',
            '15',
            '16',
            '17',
            '18',
            # '19',
            '20',
            '21',
            # '22',
            # '23'
        ]
    elif settings['dataset'] == 'CHBMIT':
        # skip Patient 12, not able to read
        targets = [
            '1',
            '2',
            '3',
            '4',
            '5',
            '6',
            '7',
            '8',
            '9',
            '10',
            '11',

            '13',
            '14',
            '15',
            # '16',
            '17',
            '18',
            '19',
            '20',
            '21',
            '22',
            '23'
        ]
    elif settings['dataset']=='EpilepsiaSurf_iEEG':
        targets = [
            # '1',
            '2',
            # '3',
            # '4',
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
            # '23',
            # '24',
            # '25',
            # '26',
            # '27',
            # '28',
            # '29',
            # '30'
        ]
    


    summary = {}
    epochs = 40
    batch_size = 32

    for target in targets:
        ckpt_target = os.path.join(settings["resultdir"],target)
        makedirs(ckpt_target)

        if settings['dataset'] == 'EpilepsiaSurf_iEEG':
            seiz_X, seiz_y = LoadSignalsACS(target, type='seiz',segement=12, settings=settings).apply()
            bckg_X, bckg_y = LoadSignalsACS(target, type='bckg',segement=12, settings=settings).apply()
            loo_folds = train_val_loo_split(seiz_X, seiz_y, bckg_X, bckg_y, 0.25)
        elif settings['dataset'] == 'FB' or settings['dataset'] == 'CHBMIT':
            seiz_X, seiz_y = PrepData(target, type='ictal', settings=settings).apply()
            bckg_X, bckg_y = PrepData(target, type='interictal', settings=settings).apply()
            loo_folds = train_val_FB_loo_split(seiz_X, seiz_y, bckg_X, bckg_y, 0.25)
            # print(seiz_y)
            # print(bckg_y)
         # print one example
        # emp = seiz_X[0]
        # print(emp.shape)
        # import matplotlib
        # matplotlib.use('Agg')
        # import matplotlib.pyplot as plt
        # plt.imshow(emp[0][0])
        # plt.savefig('myfig.pdf')
        # stop
        ind = 1
        for X_train, y_train, X_val, y_val, X_test, y_test,class_weight in loo_folds:
            gc.collect()
            ckpt = os.path.join(ckpt_target,str(ind))
            ind +=1

            # ind_interest_li =[5,7,8]
            # if ind in ind_interest_li:
            if ind ==2:
                print(ind, ckpt)
                makedirs(ckpt)
                print (X_train.shape, y_train.shape,
                    X_val.shape, y_val.shape,
                    X_test.shape, y_test.shape)
                
                if settings['dataset'] == 'EpilepsiaSurf_iEEG':
                    X_train = np.moveaxis(X_train, 2, 1)
                    X_val = np.moveaxis(X_val, 2, 1)
                    X_test = np.moveaxis(X_test, 2, 1)
                
                print (X_train.shape, y_train.shape,
                    X_val.shape, y_val.shape,
                    X_test.shape, y_test.shape)

                train_size = y_train.shape[0]-y_train.shape[0]%batch_size
                val_size = y_val.shape[0]-y_val.shape[0]%batch_size
                test_size = y_test.shape[0]

                test_batch = find_largest_divide_num(test_size)
                #.type(torch.cuda.FloatTensor)
                # transform to torch tensor
                tensor_X_train = torch.Tensor(X_train)
                tensor_y_train = torch.Tensor(y_train).type(torch.cuda.ShortTensor)
                tensor_X_val = torch.Tensor(X_val)
                tensor_y_val = torch.Tensor(y_val).type(torch.cuda.ShortTensor)
                tensor_X_test = torch.Tensor(X_test)
                tensor_y_test = torch.Tensor(y_test).type(torch.cuda.ShortTensor)

                train_loader = DataLoader(TensorDataset(tensor_X_train,tensor_y_train), batch_size=batch_size,drop_last=True, shuffle=True)
                val_loader = DataLoader(TensorDataset(tensor_X_val,tensor_y_val), batch_size=batch_size,drop_last=True, shuffle=True)
                test_loader = DataLoader(TensorDataset(tensor_X_test,tensor_y_test), batch_size=test_batch, drop_last=False,shuffle=False)
                print(train_loader,val_loader,test_loader)

            
                ###### training parameter to use ######################
                beta = 0.20912401102746153
                snn.slope = 41.425174265894256
                # thr = 0.17301711062043745
                thr =  0.27676454635641057
                #multiple GPU
                # device = torch.device("cuda")
                # SNN_net = Net()
                # SNN_net= nn.DataParallel(SNN_net)
                # SNN_net.to(device)
                #single GPU
                device = torch.device("cuda")
                SNN_net = Net((X_train.shape[1],X_train.shape[2],X_train.shape[3])).to(device)
                dtype = torch.float
                #######define the loss function########################
                optimizer = torch.optim.AdamW(SNN_net.parameters(), lr=7.205682423400944e-05, betas=(0.9, 0.999))

                on_target = thr + thr * 0.2
                off_target = thr * 0.2
                loss_fn = mse_membrane_loss(on_target=on_target, off_target=off_target, weight=class_weight)
                #loss_fn = nn.MSELoss()
                #######define the loss function########################
                ###### training parameter to use ######################
                early_stopping = EarlyStopping(patience=20, verbose=True,delta=0.004,path=os.path.join(ckpt,'checkpoint.pt'))
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

                    for data_it, targets_it in train_batch:
                        targets_it = targets_it.to(device)
                        data_it =data_it.to(device)
                    
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
                    val_batch = iter(val_loader)
                    for valdata_it, valtargets_it in val_batch:
                        # print(valdata_it)
                        valdata_it = valdata_it.to(device)
                        valtargets_it = valtargets_it.to(device)

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
                    with torch.no_grad():
                        SNN_net.eval()
                        train_AUC= print_epoch_AUC(train_loader,SNN_net, train_size, device, 'train ',batch_size)
                        val_AUC = print_epoch_AUC(val_loader, SNN_net, val_size, device, 'val',batch_size)
                        # test_AUC = print_epoch_AUC(test_loader, SNN_net,test_size, device, 'test', test_batch)
                    # test_AUC = print_epoch_AUC(test_loader, SNN_net,test_size, device, 'test', 1)
                    early_stopping((val_AUC + train_AUC)/2 , SNN_net)

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
                    SNN_net = Net((X_train.shape[1],X_train.shape[2],X_train.shape[3])).to(device)
                    print('load model from', os.path.join(ckpt,'checkpoint.pt'))
                    SNN_net.eval()
                    SNN_net.load_state_dict(torch.load(os.path.join(ckpt,'checkpoint.pt')))
                    # train_AUC= print_epoch_AUC(train_loader,SNN_net, train_size, device, 'train ',batch_size)
                    # val_AUC = print_epoch_AUC(val_loader, SNN_net, val_size, device, 'val',batch_size)
                    test_AUC = print_epoch_AUC(test_loader, SNN_net,test_size, device, 'test', test_batch)
                    print('AUC: ',test_AUC )
                    with open(os.path.join(ckpt,'result.txt'),'w') as f:
                        f.write(str(test_AUC))
            

            

    print (summary)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='EpilepsiaSurf_iEEG',
                        help="FB, CHBMIT or Kaggle2014Det,EpilepsiaSurf_iEEG")
    parser.add_argument("--mode", default='dect',
                        help="cv or test. cv is for leave-one-out cross-validation")

    args = parser.parse_args()
    assert args.dataset in ["FB", "CHBMIT", "Kaggle2014Det", "EpilepsiaSurf_iEEG","TUH"]
    assert args.mode in ['pred','dect']

    log('********************************************************************')
    log('--- START --dataset %s --mode %s  ---'
        %(args.dataset,args.mode))

    main(
        dataset=args.dataset,
        build_type=args.mode
        )
