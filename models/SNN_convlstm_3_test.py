import snntorch as snn
import torch
import torch.nn as nn
import torch.nn.functional as F
from snntorch import surrogate

# Temporal Dynamics
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # self.thr = 0.17301711062043745
        self.thr = 0.05
        slope = 13.42287274232855
        beta = 0.9181805491303656
        p1 = 0.5083664100388336
        p2 = 0.26260898840708335
        spike_grad = surrogate.straight_through_estimator()
        spike_grad2 = surrogate.fast_sigmoid(slope=slope)

    # initialize layers
        self.lstm1 = snn.SConvLSTM(in_channels=1, out_channels=16, kernel_size=3, max_pool=2, threshold=self.thr, spike_grad=spike_grad)
        self.lstm2 = snn.SConvLSTM(in_channels=16, out_channels=32, kernel_size=3, max_pool=2, threshold=self.thr, spike_grad=spike_grad)
        self.lstm3 = snn.SConvLSTM(in_channels=32, out_channels=64, kernel_size=3, max_pool=2, threshold=self.thr, spike_grad=spike_grad)
        # self.fc1 = nn.Linear(64*2*15, 512)
        self.fc1 = nn.Linear(64*2*16, 512)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad2, threshold=self.thr,)
        self.dropout1 = nn.Dropout(p1)
        self.fc2 = nn.Linear(512, 2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad2, threshold=self.thr,)
        self.dropout2 = nn.Dropout(p2)

    def forward(self, x):
        # Initialize LIF state variables and spike output tensors
        mem4 = self.lif1.init_leaky()
        mem5 = self.lif2.init_leaky()
        syn1, mem1 = self.lstm1.init_sconvlstm()
        syn2, mem2 = self.lstm2.init_sconvlstm()
        syn3, mem3 = self.lstm3.init_sconvlstm()
        # spk1_rec = []
        # mem1_rec = []
        # spk2_rec = []
        # mem2_rec = []
        # spk3_rec = []
        # mem3_rec = []
        spk5_rec = []
        mem5_rec = []

        # 
        for step in range(x.size(0)):
            # print(x.unsqueeze(2).size()) 
            spk1, syn1, mem1 = self.lstm1(x[step].unsqueeze(1), syn1, mem1)
            # print(f"Layer 1 success: {spk1.size()}")
            spk2, syn2, mem2 = self.lstm2(spk1, syn2, mem2)
            # print(f"Layer 2 success: {spk2.size()}")
            spk3, syn3, mem3 = self.lstm3(spk2, syn3, mem3)
            # print(f"Layer 3 success: {spk3.size()}")
            cur4 = self.dropout1(self.fc1(spk3.flatten(1)))
            spk4, mem4 = self.lif1(cur4, mem4)
            # print(f"Layer 3 success: {spk4.size()}")
            cur5 = self.dropout2(self.fc2(spk4))
            spk5, mem5 = self.lif2(cur5, mem5)
            # print(f"Layer 4 success: {spk4.size()}")
            
            spk5_rec.append(spk5)
            mem5_rec.append(mem5)

        #print(torch.stack(mem1_rec, dim=0)[:2],torch.stack(mem2_rec, dim=0)[:2],torch.stack(mem3_rec, dim=0)[:2],torch.stack(mem4_rec, dim=0)[:2])
        return torch.stack(spk5_rec), torch.stack(mem5_rec)