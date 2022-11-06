import os
import glob
import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import resample
import stft

from myio.save_load import save_pickle_file, load_pickle_file, \
    save_hickle_file, load_hickle_file
# from utils.group_seizure_Kaggle2014Pred import group_seizure

def load_signals_EpilepsiaSurf(data_dir='', target='1', data_type='preictal', sph=5):
    #########################################################################
    def load_raw_data(filename):
        fn = filename + '.data'
        hd = filename + '.head'

        h = pd.read_csv(hd, header=None, index_col=None, sep='=')
        start_ts = h[h[0] == 'start_ts'][1].values[0]
        num_samples = int(h[h[0] == 'num_samples'][1])
        sample_freq = int(h[h[0] == 'sample_freq'][1])
        conversion_factor = float(h[h[0] == 'conversion_factor'][1])
        num_channels = int(h[h[0] == 'num_channels'][1])
        elec_names = h[h[0] == 'elec_names'][1].values[0]
        elec_names = elec_names[1:-1]
        elec_names = elec_names.split(',')
        duration_in_sec = int(h[h[0] == 'duration_in_sec'][1])

        # print ('start_ts', start_ts)
        # print ('num_samples', num_samples)
        # print ('sample_freq', sample_freq)
        # print ('conversion_factor', conversion_factor)
        # print ('num_channels', num_channels)
        # print ('elec_names', elec_names)
        # print ('duration_in_sec', duration_in_sec)

        m = np.fromfile(fn, '<i2')
        m = m * conversion_factor
        m = m.reshape(-1, num_channels)
        assert m.shape[0] == num_samples

        ch_fn = './utils/chs.txt'
        with open(ch_fn, 'r') as f:
            chs = f.read()
            chs = chs.split(',')
        ch_ind = np.array([elec_names.index(ch) for ch in chs])
        m_s = m[:, ch_ind]
        assert m_s.shape[1] == len(chs)
        #print (m.shape)
        return m_s, chs, int(sample_freq)
    #########################################################################

    #########################################################################
    # Load all filenames per patient
    all_fn = './utils/epilepsia_recording_blocks.csv'
    all_pd = pd.read_csv(all_fn, header=0, index_col=None)
    pat_pd = all_pd[all_pd['pat']==int(target)]
    #print (pat_pd)
    pat_fd = pat_pd['folder'].values
    pat_fd = list(set(list(pat_fd)))
    assert len(pat_fd)==1
    #print (pat_fd[0])
    pat_adm = os.path.join(data_dir,pat_fd[0])
    pat_adm = glob.glob(pat_adm + '/adm_*')
    assert len(pat_adm)==1
    #print (pat_adm[0])
    pat_fns = list(pat_pd['filename'].values)
    #########################################################################

    #########################################################################
    # Load seizure info
    all_sz_fn = './utils/epilepsia_seizure_master.csv'
    all_sz_pd = pd.read_csv(all_sz_fn, header=0, index_col=None)
    pat_sz_pd = all_sz_pd[all_sz_pd['pat']==int(target)]
    pat_sz_pd = pat_sz_pd[pat_sz_pd['leading_sz']==1]
    print (pat_sz_pd)


    #########################################################################
    ii=0
    fmt = "%d/%m/%Y %H:%M:%S"

    # exi
    count_interictal = 0
    for i_fn in range(len(pat_fns)):
        pat_fn = pat_fns[i_fn]
        #print (pat_fn)
        rec_fd = pat_fn.split('_')[0]
        rec_fd = 'rec_' + rec_fd
        #print (rec_fd)
        fn = pat_fn.split('.')[0]
        fn = os.path.join(pat_adm[0],rec_fd,fn)
        # print (fn)
        this_fn_pd = pat_pd[pat_pd['filename']==pat_fn]
        gap = list(this_fn_pd['gap'].values)
        assert len(gap)==1
        gap = gap[0]
        # print (gap)

        begin_rec = list(this_fn_pd['begin'].values)
        assert len(begin_rec)==1
        begin_rec = datetime.datetime.strptime(begin_rec[0], fmt)
        # print (begin_rec)
       
        if data_type=='interictal':
            # check if current recording is at least 4 hour away from sz
            flag_sz = False
            dist_to_sz = 0
            ind = 0
            while (dist_to_sz < 4*3600*256) and (ind <= i_fn):
                full_fn = pat_fns[i_fn-ind]
                fn_ = full_fn.split('.')[0]
                fn_pd_ = pat_pd[pat_pd['filename']==full_fn]
                if ind > 0:
                    dist_to_sz = dist_to_sz + int(fn_pd_['samples']) + int(fn_pd_['gap'])*256
                ind += 1
                pd_ = pat_sz_pd[pat_sz_pd['filename']==fn_]
                #print ('!DEBUG:', i_c, pat_fns[i_c], pd_)
                if pd_.shape[0]>0:
                    flag_sz = True
                    break
                print ('DEBUG: 1', dist_to_sz,ind)
            dist_to_sz = 0
            ind = 1
            while (dist_to_sz < 4*3600*256) and (ind <= (len(pat_fns)-i_fn-1)):
                full_fn = pat_fns[i_fn+ind]
                fn_ = full_fn.split('.')[0]
                fn_pd_ = pat_pd[pat_pd['filename']==full_fn]
                dist_to_sz = dist_to_sz + int(fn_pd_['samples']) + int(fn_pd_['gap'])*256
                ind += 1
                pd_ = pat_sz_pd[pat_sz_pd['filename']==fn_]
                #print ('!DEBUG:', i_c, pat_fns[i_c], pd_)
                if pd_.shape[0]>0:
                    flag_sz = True
                    break
                print ('DEBUG: 2', dist_to_sz,ind)

            if not flag_sz:
                m, elec_names, sample_freq = load_raw_data(filename=fn)
                count_interictal += 1
                print (data_type, count_interictal, m.shape, sample_freq, elec_names)
                yield m, begin_rec
        
        elif data_type=='ictal': # actually preictal

            pat_onset_pd = pat_sz_pd[pat_sz_pd['filename']==os.path.basename(fn)]
            onset = list(pat_onset_pd['onset'].values)
            # print (os.path.basename(fn))
            # print ('!!!ONSET', len(onset), onset)
            if len(onset)>0:
                m, elec_names, sample_freq = load_raw_data(filename=fn)
                print (data_type, m.shape, sample_freq, elec_names)

                SOP = 30 * 60 * sample_freq  # SOP = 30 min
                SPH = sph * 60 * sample_freq

                dt = datetime.datetime.strptime(onset[0], fmt)
                print (begin_rec, dt)
                time_to_sz = dt - begin_rec
                time_to_sz = int(np.floor(time_to_sz.total_seconds())) * sample_freq
                print (time_to_sz)
                if time_to_sz >= SOP + SPH:
                    #yield data here
                    st = time_to_sz - SOP - SPH
                    sp = time_to_sz - SPH
                    data = m[st:sp]
                    print ('!DATA shape', data.shape)

                else: # concatenate preictal signals from previous recording
                    if time_to_sz > SPH:
                        n_spls_fr_pre = SOP + SPH - time_to_sz

                        pat_fn_pre = pat_fns[i_fn-1]
                        rec_fd_pre = pat_fn_pre.split('_')[0]
                        rec_fd_pre = 'rec_' + rec_fd_pre

                        fn_pre = pat_fn_pre.split('.')[0]
                        fn_pre = os.path.join(pat_adm[0], rec_fd_pre, fn_pre)
                        print (fn_pre)

                        m_pre, _, _ = load_raw_data(filename=fn_pre)

                        data = np.concatenate((m_pre[-n_spls_fr_pre:], m[0:time_to_sz - SPH]), axis=0)
                        print ('!DATA shape with pre', data.shape)
                    else: # all preictal data extracted from previous recording
                        pat_fn_pre = pat_fns[i_fn-1]
                        rec_fd_pre = pat_fn_pre.split('_')[0]
                        rec_fd_pre = 'rec_' + rec_fd_pre

                        fn_pre = pat_fn_pre.split('.')[0]
                        fn_pre = os.path.join(pat_adm[0], rec_fd_pre, fn_pre)
                        print (fn_pre)

                        m_pre, _, _ = load_raw_data(filename=fn_pre)

                        data = m[m_pre.shape[0]+time_to_sz - SPH - SOP:m_pre.shape[0]+time_to_sz - SPH]
                        print ('!DATA shape with pre', data.shape)

                yield data, begin_rec



    #return None



def load_signals_FB(data_dir, target, data_type, sph):
    print ('load_signals_FB for Patient', target)

    def strcv(i):
        if i < 10:
            return '000' + str(i)
        elif i < 100:
            return '00' + str(i)
        elif i < 1000:
            return '0' + str(i)
        elif i < 10000:
            return str(i) 

    if int(target) < 10:
        strtrg = '00' + str(target)
    elif int(target) < 100:
        strtrg = '0' + str(target)

    if data_type == 'ictal':

        SOP = 30*60*256
        target_ = 'pat%sIktal' % strtrg
        dir = os.path.join(data_dir, target_)
        df_sz = pd.read_csv(
            os.path.join(data_dir,'seizure.csv'),index_col=None,header=0)
        df_sz = df_sz[df_sz.patient==int(target)]
        df_sz.reset_index(inplace=True,drop=True)

        print (df_sz)
        print ('Patient %s has %d seizures' % (target,df_sz.shape[0]))
        for i in range(df_sz.shape[0]):
            data = []
            filename = df_sz.iloc[i]['filename']
            st = df_sz.iloc[i]['start'] - sph*60*256
            print ('Seizure %s starts at %d' % (filename, st))
            for ch in range(1,7):
                filename2 = '%s/%s_%d.asc' % (dir, filename, ch)
                if os.path.exists(filename2):
                    tmp = np.loadtxt(filename2)
                    seq = int(filename[-4:])
                    prevfile = '%s/%s%s_%d.asc' % (dir, filename[:-4], strcv(seq - 1), ch)

                    if st - SOP >= 0:
                        tmp = tmp[st - SOP:st]
                    else:
                        prevtmp = np.loadtxt(prevfile)
                        if os.path.exists(prevfile):
                            if st > 0:
                                tmp = np.concatenate((prevtmp[st - SOP:], tmp[:st]))
                            else:
                                tmp = prevtmp[st - SOP:st]
                        else:
                            if st > 0:
                                tmp = tmp[:st]
                            else:
                                raise Exception("file %s does not contain useful info" % filename)

                    tmp = tmp.reshape(1, tmp.shape[0])
                    data.append(tmp)

                else:
                    raise Exception("file %s not found" % filename)
            if len(data) > 0:
                concat = np.concatenate(data)
                print (concat.shape)
                yield (concat)

    elif data_type == 'interictal':
        target_ = 'pat%sInteriktal' % strtrg
        dir = os.path.join(data_dir, target_)
        text_files = [f for f in os.listdir(dir) if f.endswith('.asc')]
        prefixes = [text[:8] for text in text_files]
        prefixes = set(prefixes)
        prefixes = sorted(prefixes)

        totalfiles = len(text_files)
        print (prefixes, totalfiles)

        done = False
        count = 0

        for prefix in prefixes:
            i = 0
            while not done:

                i += 1

                stri = strcv(i)
                data = []
                for ch in range(1, 7):
                    filename = '%s/%s_%s_%d.asc' % (dir, prefix, stri, ch)

                    if os.path.exists(filename):
                        try:                           
                            tmp = np.loadtxt(filename)
                            tmp = tmp.reshape(1, tmp.shape[0])
                            data.append(tmp)
                            count += 1
                        except:
                            print ('OOOPS, this file can not be loaded', filename)
                    elif count >= totalfiles:
                        done = True
                    elif count < totalfiles:
                        break
                    else:
                        raise Exception("file %s not found" % filename)

                if i > 99999:
                    break

                if len(data) > 0:
                    yield (np.concatenate(data))	
	
def load_signals_CHBMIT(data_dir, target, data_type, sph):
    print ('load_signals_CHBMIT for Patient', target, 'using SPH =', sph)
    from mne.io import RawArray, read_raw_edf
    from mne.channels import read_montage
    from mne import create_info, concatenate_raws, pick_types
    from mne.filter import notch_filter

    onset = pd.read_csv(os.path.join(data_dir, 'seizure_summary.csv'),header=0)
    #print (onset)
    osfilenames,szstart,szstop = onset['File_name'],onset['Seizure_start'],onset['Seizure_stop']
    osfilenames = list(osfilenames)
    #print ('Seizure files:', osfilenames)

    segment = pd.read_csv(os.path.join(data_dir, 'segmentation.csv'),header=None)
    nsfilenames = list(segment[segment[1]==0][0])

    nsdict = {
            '0':[]
    }
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
        '12',
        '13',
        '14',
        '15',
        '16',
        '17',
        '18',
        '19',
        '20',
        '21',
        '22',
        '23'
    ]
    for t in targets:
        nslist = [elem for elem in nsfilenames if
                  elem.find('chb%s_' %t)!= -1 or
                  elem.find('chb0%s_' %t)!= -1 or
                  elem.find('chb%sa_' %t)!= -1 or
                  elem.find('chb%sb_' %t)!= -1 or
                  elem.find('chb%sc_' %t)!= -1]
        nsdict[t] = nslist
    #nsfilenames = shuffle(nsfilenames, random_state=0)

    special_interictal = pd.read_csv(os.path.join(data_dir, 'special_interictal.csv'),header=None)
    sifilenames,sistart,sistop = special_interictal[0],special_interictal[1],special_interictal[2]
    sifilenames = list(sifilenames)

    def strcv(i):
        if i < 10:
            return '0' + str(i)
        elif i < 100:
            return str(i)

    strtrg = 'chb' + strcv(int(target))    
    dir = os.path.join(data_dir, strtrg)
    text_files = [f for f in os.listdir(dir) if f.endswith('.edf')]
    #print (target,strtrg)
    #print (text_files)

    if data_type == 'ictal':
        filenames = [filename for filename in text_files if filename in osfilenames]
        #print ('ictal files', filenames)
    elif data_type == 'interictal':
        filenames = [filename for filename in text_files if filename in nsdict[target]]
        #print ('interictal files', filenames)

    totalfiles = len(filenames)
    print ('Total %s files %d' % (data_type,totalfiles))
    for filename in filenames:
        exclude_chs = []
        if target in ['4','9']:
            exclude_chs = [u'T8-P8']

        if target in ['13','16']:
            chs = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3', u'F3-C3', u'C3-P3', u'P3-O1', u'FP2-F4', u'F4-C4', u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'T8-P8', u'FZ-CZ', u'CZ-PZ']
        elif target in ['4']:
            chs = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3', u'F3-C3', u'C3-P3', u'P3-O1', u'FP2-F4', u'F4-C4', u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'P8-O2', u'FZ-CZ', u'CZ-PZ', u'P7-T7', u'T7-FT9', u'FT10-T8']
        elif target in ['9']:
            chs = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3', u'F3-C3', u'C3-P3', u'P3-O1', u'FP2-F4', u'F4-C4', u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'P8-O2', u'FZ-CZ', u'CZ-PZ', u'P7-T7', u'T7-FT9', u'FT9-FT10', u'FT10-T8']
        else:
            chs = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3', u'F3-C3', u'C3-P3', u'P3-O1', u'FP2-F4', u'F4-C4', u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'T8-P8', u'P8-O2', u'FZ-CZ', u'CZ-PZ', u'P7-T7', u'T7-FT9', u'FT9-FT10', u'FT10-T8']


        rawEEG = read_raw_edf('%s/%s' % (dir, filename),
                              #exclude=exclude_chs,  #only work in mne 0.16
                              verbose=0,preload=True)

        #print (rawEEG.ch_names)
        rawEEG.pick_channels(chs)
        #print (rawEEG.ch_names)
        #rawEEG.notch_filter(freqs=np.arange(60,121,60))
        tmp = rawEEG.to_data_frame()
        tmp = tmp.as_matrix()

        if data_type == 'ictal':
            SOP = 30 * 60 * 256
            # get seizure onset information
            indices = [ind for ind,x in enumerate(osfilenames) if x==filename]
            if len(indices) > 0:
                print ('%d seizures in the file %s' % (len(indices),filename))
                prev_sp = -1e6
                for i in range(len(indices)):
                    st = szstart[indices[i]]*256 - sph * 60 * 256 #e.g., SPH=5min
                    sp = szstop[indices[i]]*256
                    #print ('Seizure %s %d starts at %d stops at %d last sz stop is %d' % (filename, i, (st+5*60*256),sp,prev_sp))

                    # take care of some special filenames
                    if filename[6]=='_':
                        seq = int(filename[7:9])
                    else:
                        seq = int(filename[6:8])
                    if filename == 'chb02_16+.edf':
                        prevfile = 'chb02_16.edf'
                    else:
                        if filename[6]=='_':
                            prevfile = '%s_%s.edf' %(filename[:6],strcv(seq-1))
                        else:
                            prevfile = '%s_%s.edf' %(filename[:5],strcv(seq-1))

                    if st - SOP > prev_sp:
                        prev_sp = sp
                        if st - SOP >= 0:
                            data = tmp[st - SOP : st]
                        else:
                            if os.path.exists('%s/%s' % (dir, prevfile)):
                                rawEEG = read_raw_edf('%s/%s' % (dir, prevfile), preload=True,verbose=0)
                                rawEEG.pick_channels(chs)
                                prevtmp = rawEEG.to_data_frame()
                                prevtmp = prevtmp.as_matrix()
                                if st > 0:
                                    data = np.concatenate((prevtmp[st - SOP:], tmp[:st]))
                                else:
                                    data = prevtmp[st - SOP:st]

                            else:
                                if st > 0:
                                    data = tmp[:st]
                                else:
                                    #raise Exception("file %s does not contain useful info" % filename)
                                    print ("WARNING: file %s does not contain useful info" % filename)
                                    continue
                    else:
                        prev_sp = sp
                        continue

                    #print ('data shape', data.shape)
                    if data.shape[0] == SOP:
                        yield(data)
                    else:
                        continue

        elif data_type == 'interictal':
            if filename in sifilenames:
                st = sistart[sifilenames.index(filename)]
                sp = sistop[sifilenames.index(filename)]
                if sp < 0:
                    data = tmp[st*256:]
                else:
                    data = tmp[st*256:sp*256]
            else:
                data = tmp
            #print ('data shape', data.shape)
            yield(data)

class LoadSignals():
    def __init__(self, target, type, settings, sph):
        self.target = target
        self.settings = settings
        self.type = type
        self.sph = sph
        self.global_proj = np.array([0.0]*114)
        self.significant_channels = None

    def read_raw_signal(self):
        if self.settings['dataset'] == 'CHBMIT':
            self.samp_freq = 256
            self.freq = 256
            self.global_proj = np.array([0.0]*114)

            from utils.CHBMIT_channels import channels
            try:
                self.significant_channels = channels[self.target]
            except:
                pass
            print (self.target,self.significant_channels)    
            return load_signals_CHBMIT(self.settings['datadir'], self.target, self.type, self.sph)
        elif self.settings['dataset'] == 'FB':
            self.samp_freq = 256
            self.freq = 256
            self.global_proj = np.array([0.0]*114)
            return load_signals_FB(self.settings['datadir'], self.target, self.type, self.sph)
        elif self.settings['dataset'] == 'Kaggle2014Pred':
            if self.type == 'ictal':
                data_type = 'preictal'
            else:
                data_type = self.type
            from utils.Kaggle2014Pred_channels import channels
            try:
                self.significant_channels = channels[self.target]
            except:
                pass
            print (self.target,self.significant_channels)
            return load_signals_Kaggle2014Pred(self.settings['datadir'], self.target, data_type)
        elif self.settings['dataset'] == 'EpilepsiaSurf':
            self.samp_freq = 256
            self.freq = 256
            self.global_proj = np.array([0.0] * 128)
            return  load_signals_EpilepsiaSurf(self.settings['datadir'], self.target, self.type, self.sph)

        return 'array, freq, misc'

    
    def preprocess(self, data_):
        ictal = self.type == 'ictal'
        interictal = self.type == 'interictal'
        targetFrequency = self.freq  # re-sample to target frequency
        numts = 30
        
        df_sampling = pd.read_csv(
            'sampling_%s.csv' % self.settings['dataset'],
            header=0,index_col=None)
        trg = int(self.target)
        print (df_sampling)
        print (df_sampling[df_sampling.Subject==trg].ictal_ovl.values)
        ictal_ovl_pt = \
            df_sampling[df_sampling.Subject==trg].ictal_ovl.values[0]
        ictal_ovl_len = int(targetFrequency*ictal_ovl_pt*numts)

        def process_raw_data(mat_data):            
            print ('Loading data')
            X = []
            X_other = []
            y = []
            #scale_ = scale_coef[target]
            for data, begin_rec in mat_data:
                print ('Recording begin time', begin_rec, begin_rec.hour, begin_rec.weekday())
                if self.settings['dataset'] == 'FB':
                    data = data.transpose()
                if self.significant_channels is not None:
                    print ('Reducing number of channels')
                    data = data[:,self.significant_channels]
                if ictal:
                    y_value=1
                else:
                    y_value=0

                X_temp = []
                X_other_temp = []
                y_temp = []
    
                totalSample = int(data.shape[0]/targetFrequency/numts) + 1
                window_len = int(targetFrequency*numts)
                for i in range(totalSample):
                    if (i+1)*window_len <= data.shape[0]:
                        s_time = begin_rec + timedelta(seconds=int(i*numts))
                        s = data[i*window_len:(i+1)*window_len,:]

                        stft_data = stft.spectrogram(s,framelength=targetFrequency,centered=False)
                        stft_data = np.transpose(stft_data,(2,1,0))
                        stft_data = np.abs(stft_data)+1e-6

                        if self.settings['dataset'] == 'FB':
                            stft_data = np.concatenate((stft_data[:,:,1:47],
                                                        stft_data[:,:,54:97],
                                                        stft_data[:,:,104:]),
                                                       axis=-1)
                        elif self.settings['dataset'] == 'CHBMIT':
                            stft_data = np.concatenate((stft_data[:,:,1:57],
                                                        stft_data[:,:,64:117],
                                                        stft_data[:,:,124:]),
                                                       axis=-1)
                        elif self.settings['dataset'] == 'EpilepsiaSurf':
                            stft_data = stft_data[:,:,1:]
                        stft_data = np.log10(stft_data)
                        indices = np.where(stft_data <= 0)
                        stft_data[indices] = 0

                        proj = np.sum(stft_data,axis=(0,1),keepdims=False)
                        # self.global_proj += proj/1000.0


                        # from matplotlib import cm
                        # from matplotlib import pyplot as plt
                        # plt.matshow(stft_data[0]/np.max(stft_data[0]))
                        # plt.colorbar()
                        # plt.show()

                        #stft_data = np.multiply(stft_data,1.0/scale_)

                        '''
                        plt.matshow(stft_data[0]/np.max(stft_data[0]))
                        plt.colorbar()
                        plt.show()
                        '''

                        if self.settings['dataset'] in ['FB', 'CHBMIT']:
                            stft_data = stft_data[:,:56,:112]
                        elif self.settings['dataset'] == 'EpilepsiaSurf':
                            stft_data = stft_data[:,:56,:]
                        stft_data = stft_data.reshape(-1, stft_data.shape[0],
                                                      stft_data.shape[1],
                                                      stft_data.shape[2])


                        X_temp.append(stft_data)
                        X_other_temp.append(np.array([s_time.hour, s_time.weekday()]))
                        y_temp.append(y_value)

                #overdsampling
                if ictal:
                    i = 1
                    print ('ictal_ovl_len =', ictal_ovl_len)
                    while (window_len + (i + 1)*ictal_ovl_len <= data.shape[0]):
                        s_time = begin_rec + timedelta(seconds=int(i*ictal_ovl_len/targetFrequency))
                        s = data[i*ictal_ovl_len:i*ictal_ovl_len + window_len, :]

                        stft_data = stft.spectrogram(s, framelength=targetFrequency,centered=False)
                        stft_data = np.transpose(stft_data, (2, 1, 0))
                        stft_data = np.abs(stft_data)+1e-6

                        if self.settings['dataset'] == 'FB':
                            stft_data = np.concatenate((stft_data[:,:,1:47],
                                                        stft_data[:,:,54:97],
                                                        stft_data[:,:,104:]),
                                                       axis=-1)
                        elif self.settings['dataset'] == 'CHBMIT':
                            stft_data = np.concatenate((stft_data[:,:,1:57],
                                                        stft_data[:,:,64:117],
                                                        stft_data[:,:,124:]),
                                                       axis=-1)
                        elif self.settings['dataset'] == 'EpilepsiaSurf':
                            stft_data = stft_data[:, :, 1:]
                        stft_data = np.log10(stft_data)
                        indices = np.where(stft_data <= 0)
                        stft_data[indices] = 0

                        proj = np.sum(stft_data,axis=(0,1),keepdims=False)
                        self.global_proj += proj/1000.0

                        #stft_data = np.multiply(stft_data,1.0/scale_)

                        if self.settings['dataset'] in ['FB', 'CHBMIT']:
                            stft_data = stft_data[:,:56,:112]
                        elif self.settings['dataset'] == 'EpilepsiaSurf':
                            stft_data = stft_data[:,:56,:]
                        stft_data = stft_data.reshape(-1, stft_data.shape[0],
                                                      stft_data.shape[1],
                                                      stft_data.shape[2])

                        X_temp.append(stft_data)
                        X_other_temp.append(np.array([s_time.hour, s_time.weekday()]))
                        # differentiate between non-overlapped and overlapped
                        # samples. Testing only uses non-overlapped ones.
                        y_temp.append(2)
                        i += 1

                if len(X_temp)>0:
                    X_temp = np.concatenate(X_temp, axis=0)
                    X_other_temp = np.array(X_other_temp)
                    y_temp = np.array(y_temp)
                    X.append(X_temp)
                    X_other.append(X_other_temp)
                    y.append(y_temp)

            if ictal or interictal:
                #y = np.array(y)
                try:
                    print ('X', len(X), X[0].shape, 'y', len(y), y[0].shape)
                except:
                    print ('!!!!!!!!!!!!DEBUG!!!!!!!!!!!!!:', X)
                return X, X_other, y
            else:
                print ('X', X.shape)
                return X, X_other

        data = process_raw_data(data_)

        return  data

    def apply(self, save_STFT=False, dir=''):
        def save_STFT_to_files(X):
            if isinstance(X, list):
                index=0
                for x in X:
                    for i in range(x.shape[0]):
                        fn = '%s_%s_%d_%d.npy' % (self.target,self.type,index,i)
                        if self.settings['dataset'] in ['FB','CHBMIT']:
                            x_ = x[i,:,:56,:112]
                        elif self.settings['dataset'] == 'Kaggle2014Pred':
                            if 'Dog' in self.target:
                                x_ = x[i,:,:56,:96]
                            elif 'Patient' in self.target:
                                x_ = x[i,:,:112,:96]
                        elif self.settings['dataset'] == 'EpilepsiaSurf':
                            x_ = x[i,:,:,:]

                        np.save(os.path.join(dir,fn),x_)
                         # 2352, 1, 16, 59, 100 - Kaggle Dog
                        # (567, 1, 15, 119, 100) - Kaggle Patient
                        # 3020, 1, 6, 59, 114 - Freiburg
                    index += 1
            else:
                for i in range(X.shape[0]):
                    fn = '%s_%s_0_%d.npy' % (self.target,self.type,i)
                    if self.settings['dataset'] in ['FB','CHBMIT']:
                        x_ = X[i,:,:56,:112]

                    elif self.settings['dataset'] == 'Kaggle2014Pred':
                        if 'Dog' in self.target:
                            x_ = X[i,:,:56,:96]
                        elif 'Patient' in self.target:
                            x_ = X[i,:,:112,:96]

                    np.save(os.path.join(dir,fn),x_)
                    # print (fn, X.shape)
                    # sss
            print ('Finished saving STFT to %s' % dir)
            return None

        filename = '%s_%s' % (self.type, self.target)
        cache = load_hickle_file(
            os.path.join(self.settings['cachedir'], filename))
        if cache is not None:
            if save_STFT:
                X, _, _ = cache
                return save_STFT_to_files(X)
            else:
                return cache

        data = self.read_raw_signal()
        if self.settings['dataset']=='Kaggle2014Pred':
            X, y = self.preprocess_Kaggle(data)
            X_other = 'not_implemented'
        elif self.settings['dataset']=='EpilepsiaSurf':
            X, X_other, y = self.preprocess(data)
        else:
            X, y = self.preprocess(data)
            X_other = 'not_implemented'
        save_hickle_file(
            os.path.join(self.settings['cachedir'], filename),
            [X, X_other, y])

        if save_STFT:
            return save_STFT_to_files(X)
        else:
            return X, X_other, y

