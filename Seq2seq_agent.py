import torch
import torch.nn as nn
import numpy as np
import random
from torch import FloatTensor as FT
import matplotlib.pyplot as plt
import argparse
from parsing import make_parser
from torch.autograd import Variable
import torch.nn.init as init

class ScaledDotProductAttention(nn.Module):
  def __init__(self):
    super(ScaledDotProductAttention, self).__init__()
  def forward(self, Q, K, V, attn_mask=False):
    scores = torch.matmul(Q, K.transpose(-1, -2)) / 1
    if attn_mask is not None:
      scores.masked_fill_(scores == 0, -1e9)
    attn = nn.Softmax(dim=-1)(scores)
    context = torch.matmul(attn, V)
    return context, attn

class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_size, dec_size, device):
        super(PositionWiseFeedForward, self).__init__()
        self.fclayer = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size/2), dec_size),
        ).to(device)
    def forward(self, x):
        return self.fclayer(x)


class DGPS(nn.Module):
    def __init__(self, args = make_parser(),
                 input_dim = 6, dof_seq = 21, dof = 2, Ksignum = 25, sig = 2,
                 buffer_size = 5000, batch_size = 32, gamma = 0.001,
                 enc_size = 6, dec_size = 6, hidden_size = 32, num_layers = 1):
        super(DGPS, self).__init__()
        self.args = args
        self.device = self.args.device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.enc_embed = nn.Sequential(
            nn.Linear(enc_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),).to(self.device)

        self.enc_embedk = nn.Sequential(
            nn.Linear(enc_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), ).to(self.device)

        self.enc_embedv = nn.Sequential(
            nn.Linear(enc_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), ).to(self.device)

        self.enc_embed2 = nn.Sequential(
            nn.Linear(enc_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),).to(self.device)

        self.enc_embed2k = nn.Sequential(
            nn.Linear(enc_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), ).to(self.device)

        self.enc_embed2v = nn.Sequential(
            nn.Linear(enc_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), ).to(self.device)

        #self.enc_embed = nn.Linear(Ksignum, hidden_size).to(self.device)
        #self.enc_embed2 = nn.Linear(Ksignum, hidden_size).to(self.device)
        self.eye = torch.eye(input_dim, device=self.device)
        self.eye2 = torch.eye(args.output_dim, device=self.device)
        self.eyedof = torch.eye(dof_seq - 1, device=self.device)
        self.Al = torch.zeros((dof_seq-1, dof_seq-1), device = self.device)
        for i in range(dof_seq-2):
            self.Al[i,i+1] = 1
        self.h = torch.zeros((dof_seq-1, 1), device = self.device)
        self.h[-1,0] = 1
        self.W = self.eye2


        self.Ktheta = {}
        self.Ktheta_buffer = torch.zeros((input_dim * Ksignum, args.output_dim), device = self.device)


        self.Alu = self.Al
        self.hu = self.h
        self.re_u = 1
        self.W = self.eye2
        self.DOF_NN = (input_dim) * 2 ###!
        self.ctheta = Variable((torch.rand((int(self.DOF_NN), Ksignum)) - 0.5).type(torch.FloatTensor).to(self.device),requires_grad=True)

        self.slamb = sig*torch.ones((Ksignum), device = self.device)
        self.fsignalu = torch.zeros((dof_seq-1, input_dim), device = self.device)
        self.fsignalx = torch.zeros((dof_seq-1, args.output_dim), device = self.device)
        self.fsignalq = torch.zeros((dof_seq-1, input_dim), device = self.device)
        self.Ksignal = torch.empty((Ksignum, 1), device = self.device)
        self.Ksignal_admittance = torch.empty((Ksignum, 1), device = self.device)
        self.buffer = torch.empty((buffer_size, Ksignum + 6*input_dim + 9 + 6 + 0), device = self.device)
        self.signalbuffer = torch.empty((buffer_size, dof_seq-1, input_dim + input_dim + args.output_dim), device = self.device) # u, q, x
        self.input_dim = input_dim
        self.output_dim = args.output_dim
        self.Ksignum = Ksignum
        self.gamma = gamma
        self.sig = sig
        self.violation_count = 0
        self.violation_buffer = 0
        self.learning_count = 0
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.costerror = 1
        self.costerror_prev = 0
        self.error_rev = 0
        self.w_reuse = args.w_reuse

        self.u_ma = 0.01*torch.ones((10,input_dim), device = self.device)
        self.costerror_ma = 0.01*torch.ones(1000, device = self.device)
        self.costerror_ma_kin = 0.01*torch.ones(1000, device = self.device)
        self.crit_cul = torch.zeros((self.Ksignum, 1), device = self.device)
        self.q_des_prev = torch.zeros((self.input_dim, 1), device = self.device)
        self.q_des = torch.zeros((self.input_dim, 1), device = self.device)
        self.q_des_init = torch.zeros((self.input_dim, 1), device = self.device)
        self.make_weights(args.w_reuse)
        self.cloudpoint = torch.ones((100,1), device = self.device)
        self.sf_score_qq = 1 / dof_seq
        self.null_ratio = 0.15
        self.count_cul = 0
        self.enc_self_attn = ScaledDotProductAttention()
        self.pos_ffn = PositionWiseFeedForward(hidden_size, dec_size, args.device).to(self.device)
        self.enc_self_attn2 = ScaledDotProductAttention()
        self.pos_ffn2= PositionWiseFeedForward(hidden_size, dec_size, args.device).to(self.device)
        self.fig = plt.figure() if args.visual == 1 else None
        self.ax = self.fig.add_subplot(121) if args.visual == 1 else None
        self.ax2 = self.fig.add_subplot(122) if args.visual == 1 else None
        self.pos_encoding_table = self.get_sinusoid_encoding_table(dof_seq-1, hidden_size)
        self.zero1 = torch.zeros((self.input_dim, self.input_dim), device = self.device)
        self.zero2 = torch.zeros((self.output_dim, self.input_dim), device = self.device)
        netlist = list(self.Kthetam.parameters()) + list(self.Kthetac.parameters()) + list(self.Kthetaj.parameters()) + \
                  list(self.Kthetaf.parameters()) + list(self.Kthetag.parameters()) + list(self.Kthetaj0.parameters()) + [self.ctheta]
        self.optimizer = torch.optim.Adam(netlist, lr = gamma)
        a_list = list(self.enc_self_attn.parameters()) + list(self.pos_ffn.parameters()) + list(
            self.enc_embed.parameters()) + list(self.enc_embedk.parameters()) + list(self.enc_embedv.parameters())
        self.optimizer2 = torch.optim.Adam(a_list, lr=5e-4)

        b_list = list(self.enc_self_attn2.parameters()) + list(self.pos_ffn2.parameters()) + list(
            self.enc_embed2.parameters()) + list(
            self.enc_embed2k.parameters()) + list(
            self.enc_embed2v.parameters())
        self.optimizer3 = torch.optim.Adam(b_list, lr=5e-4)


    def forward_attn(self, enc_inputs):
        enc_embed = self.enc_embed(enc_inputs).unsqueeze(0)
        enc_embedk = self.enc_embedk(enc_inputs).unsqueeze(0)
        enc_embedv = self.enc_embedv(enc_inputs).unsqueeze(0)
        enc_embed_q = enc_embed + 1 * self.pos_encoding_table.unsqueeze(0)
        enc_embed_k = enc_embedk + 1 * self.pos_encoding_table.unsqueeze(0)
        enc_outputs, attn = self.enc_self_attn(enc_embed_q, enc_embed_q, enc_embed)
        enc_outputs = self.pos_ffn(enc_outputs.squeeze(0))
        return enc_outputs, attn

    def forward_attn2(self, enc_inputs):
        enc_embed = self.enc_embed2(enc_inputs).unsqueeze(0)
        enc_embedk = self.enc_embed2k(enc_inputs).unsqueeze(0)
        enc_embedv = self.enc_embed2v(enc_inputs).unsqueeze(0)
        enc_embed_q = enc_embed + 1 * self.pos_encoding_table.unsqueeze(0)
        enc_embed_k = enc_embedk + 1 * self.pos_encoding_table.unsqueeze(0)
        enc_outputs, attn = self.enc_self_attn2(enc_embed_q, enc_embed_q, enc_embed)
        enc_outputs = self.pos_ffn2(enc_outputs.squeeze(0))
        return enc_outputs, attn


    def get_sinusoid_encoding_table(self,n_position, d_model):
        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(d_model)]
        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return FT(sinusoid_table).to(self.device)

    def make_initial_qdes(self, q):
        self.q_des_init = q

    def make_weights(self, w_reuse):
        if not w_reuse:
            self.Kthetam = nn.ModuleList([nn.Linear(self.Ksignum, self.input_dim) for _ in range(self.input_dim + 1)]).to(self.device)
            self.Kthetac = nn.ModuleList([nn.Linear(self.Ksignum, self.input_dim) for _ in range(self.input_dim)]).to(self.device)
            self.Kthetaj = nn.ModuleList([nn.Linear(self.Ksignum, self.input_dim) for _ in range(self.output_dim)]).to(self.device)
            self.Kthetaf = nn.ModuleList([nn.Linear(self.Ksignum, self.input_dim) for _ in range(self.input_dim)]).to(self.device)
            self.Kthetag = nn.Linear(self.Ksignum, self.input_dim).to(self.device)
            self.Kthetaj0 = nn.Linear(self.Ksignum, self.output_dim).to(self.device)
            init.uniform_(self.Kthetaj0.weight, -0.1, 0.1)
            init.uniform_(self.Kthetaj0.bias, -0.1, 0.1)
        else:
            raise

    def save_buffer(self, Ksignal, qdot, q2dot, u, x, q, f, R, xi):
        self.buffer[self.count_cul % self.buffer_size, :] = torch.cat([Ksignal.view(-1), qdot, q2dot, u, x, q, f, R, xi])
        self.signalbuffer[self.count_cul % self.buffer_size, :, :] = torch.cat([self.fsignalu, self.fsignalq, self.fsignalx], dim = -1)
        self.count_cul += 1


    def generate_action(self, u, q, x, qv, xdes, v, f, sf, control_mode = 'force'):
        device = self.device
        with torch.no_grad():
            self.fsignalu = self.Al @ self.fsignalu + self.h @ u.view(1,-1).tanh()
            self.fsignalq = self.Al @ self.fsignalq + self.h @ q.view(1,-1)
            self.fsignalx = self.Al @ self.fsignalx + self.h @ x.view(1,-1)
            fsignal = torch.cat([self.fsignalq[-1,:],self.fsignalq[-2,:]],dim=0)


        AA, BB = self.forward_attn(self.fsignalu)
        lossA = torch.nn.functional.mse_loss(AA, self.fsignalq)


        AA2, BB2 = self.forward_attn2(self.fsignalq)
        lossB = torch.nn.functional.mse_loss(AA2, self.fsignalx)


        sf_score_q_prev = self.sf_score_qq

        BBB = BB.cpu().detach().numpy().squeeze(0)
        BBB2 = BB2.cpu().detach().numpy().squeeze(0)

        self.sf_score_qq = np.trace(BBB2)
        """
        if self.count_cul % 100 == 50 and self.args.visual == 1:
            self.ax.pcolor(BBB)
            if self.violation_buffer <= 0:
                self.ax.set_title("Normal: {:.3f}".format(np.trace(BBB)))
            else:
                self.ax.set_title("--External Perturbation--: {:.3f}".format(np.trace(BBB)))

            if self.violation_buffer <= 0:
                self.ax2.set_title("Normal: {:.3f}".format(np.trace(BBB2)))
            else:
                self.ax2.set_title("--External Perturbation--: {:.3f}".format(np.trace(BBB2)))
            self.ax2.pcolor(BBB2)

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.fig.show()
        """
        loss_ratio =  (np.trace(BBB2)-8.5) / (sf_score_q_prev-8.5)
        self.re_u = 10 / np.trace(BBB) if self.count_cul >= 500 and self.violation_buffer <= 0 else 0
        if self.violation_buffer > 0:
            sf = sf  * 1
        else:
            sf = np.clip(sf * (1 + (loss_ratio - 1) * (-3.0)), sf * 0.9, sf * 1.1)
        if self.re_u < 1 and self.violation_buffer < 0 and self.count_cul % 10 == 0:
            print(self.count_cul, '{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(sf, np.trace(BBB), np.trace(BBB2), lossA.cpu().detach().numpy(), lossB.cpu().detach().numpy()))



        sf = np.clip(sf, 0.005, 2)
        func_m = torch.zeros_like(self.zero1)
        func_c = torch.zeros_like(self.zero1)
        func_j = torch.zeros_like(self.zero2)
        func_f = torch.zeros_like(self.zero1)
        #feature = torch.exp()
        self.Ksignal = torch.exp(-((fsignal.unsqueeze(1).repeat([1, self.Ksignum]) - self.ctheta).pow(2).sum(0)) * (1/(self.slamb**2))).unsqueeze(0)
        func_g = self.Kthetag(self.Ksignal).T
        func_j0 = self.Kthetaj0(self.Ksignal).T
        for i in range(self.input_dim):
            func_m[:,i] = self.Kthetam[i](self.Ksignal)
            func_c[:,i] = self.Kthetac[i](self.Ksignal)
            func_j[:,i] = self.Kthetaj[i](self.Ksignal)
            func_f[:,i] = self.Kthetaf[i](self.Ksignal)
        func_diag = torch.diag_embed(self.Kthetam[-1](self.Ksignal).abs()).squeeze(0)
        L = torch.tril(func_m, diagonal = -1) + func_diag
        func_m = L @ L.transpose(-2,-1)

        #self.null_ratio = 0.0 * np.clip(sf - 0.0, 0, 0.25)
        q_des_2prev = self.q_des_prev
        self.q_des_prev = self.q_des
        func_jinv = func_j.T @ torch.inverse(func_j @ func_j.T + sf * self.eye)
        WW = self.eye
        self.q_des = func_j.T @ WW @ torch.inverse(sf * self.eye + func_j @ WW @ func_j.T) @ \
                     (xdes - func_j0 - self.args.gain * (x - xdes))
        qv_des = torch.tanh((self.q_des - self.q_des_prev) * 1000) - 0.5
        qa_des = torch.tanh((self.q_des - 2 * self.q_des_prev + q_des_2prev) * 1000) - 0.5


        if control_mode == 'impedance':
            Kd = 5.0 * self.eye
            Kp = 25.0 * self.eye
            u_next = func_c @ qv + func_g \
                     - (func_j.T @ Kd @ func_j + sf * Kd * 1 * self.eye) @ (qv - qv_des) \
                     - (func_j.T @ Kp @ func_j + sf * Kp * 1 * self.eye) @ (q - self.q_des)

        elif control_mode == 'admittance':
            # Note that xdes is qdes
            Kd = 5.0 * self.eye
            Kp = 100.0 * self.eye
            u_next = func_g - Kd @ qv - Kp @ (q - xdes) + func_f @ f

        elif control_mode == 'force':
            Kd = 5.0 * self.eye
            Kp = 25.0 * self.eye
            Kl = 15.0 * self.eye
            u_next = func_m @ (qa_des - Kl @ (qv - qv_des)) + func_c @ (qv_des - Kl @ (q - self.q_des)) + \
                     func_g - Kd @ (qv - qv_des) - Kp @ Kl @ (q - self.q_des)

            

        return u_next, sf, torch.trace(BB.squeeze(0)), torch.trace(BB2.squeeze(0)), lossA, func_j

    def train(self, learn, control_mode):

        if self.count_cul <= self.buffer_size - 1:
            numlist = np.linspace(0, self.count_cul, self.count_cul+1)
        else:
            numlist = np.linspace(0, self.buffer_size-1, self.buffer_size)
        numlist = numlist.tolist()
        try:
            replay_sample = random.sample(numlist, self.batch_size)
        except:
            replay_sample = numlist
        replay_sample[-1] = (self.count_cul - 1) % self.buffer_size

        with torch.no_grad():
            Ksignal_ = self.buffer[replay_sample, 0:self.Ksignum]
            f_ = self.buffer[replay_sample, self.Ksignum + 6 + 6 + 6 + 6 + 6:self.Ksignum + 6 + 6 + 6 + 6 + 6 + 6]
            u_ = self.buffer[replay_sample, self.Ksignum + 6 + 6:self.Ksignum + 6 + 6 + 6]
            x_ = self.buffer[replay_sample, self.Ksignum + 6 + 6 + 6:self.Ksignum + 6 + 6 + 6 + 6]
            q_ = self.buffer[replay_sample, self.Ksignum + 6 + 6 + 6 + 6:self.Ksignum + 6 + 6 + 6 + 6 + 6]
            qdot_ = self.buffer[replay_sample, self.Ksignum:self.Ksignum + 6]
            q2dot_ = self.buffer[replay_sample, self.Ksignum + 6:self.Ksignum + 6 + 6]
            fsignal_u = self.signalbuffer[replay_sample, :, 0:self.input_dim]
            fsignal_q = self.signalbuffer[replay_sample, :, self.input_dim:self.input_dim*2]
            fsignal_x = self.signalbuffer[replay_sample, :, self.input_dim*2:self.input_dim*2+self.output_dim]


        func_g_ = self.Kthetag(Ksignal_).unsqueeze(2)
        func_j0_ = self.Kthetaj0(Ksignal_).unsqueeze(2)
        func_m_ = torch.zeros_like(self.zero1).repeat([Ksignal_.size(0), 1,1])
        func_c_ = torch.zeros_like(self.zero1).repeat([Ksignal_.size(0), 1,1])
        func_j_ = torch.zeros_like(self.zero2).repeat([Ksignal_.size(0), 1,1])
        func_f_ = torch.zeros_like(self.zero1).repeat([Ksignal_.size(0), 1,1])

        for x in range(self.input_dim):
            func_m_[:, :, x] = self.Kthetam[x](Ksignal_)
            func_c_[:, :, x] = self.Kthetac[x](Ksignal_)
            try:
                func_j_[:, :, x] = self.Kthetaj[x](Ksignal_)
            except:
                continue
            func_f_[:, :, x] = self.Kthetaf[x](Ksignal_)
        func_diag = torch.diag_embed(self.Kthetam[-1](Ksignal_).abs())
        L = torch.tril(func_m_, diagonal=-1) + func_diag
        func_m_ = L @ L.transpose(-2, -1)

        self.optimizer.zero_grad()
        costerror_ = torch.nn.functional.mse_loss(func_g_ + func_m_ @ q2dot_.unsqueeze(2) + func_c_ @ qdot_.unsqueeze(2), u_.unsqueeze(2))
        costerror_kin = torch.nn.functional.mse_loss(func_j0_ + func_j_ @ q_.unsqueeze(2), x_[:, 0:self.output_dim].unsqueeze(2))
        if self.count_cul % 10 == 0:
            print('Loss:', '{:.3f}, {:.3f}'.format(costerror_.item(), costerror_kin.item()))
        if control_mode == 'admittance':
            self.re_u = 0
        if learn and self.re_u < 1 and self.violation_buffer <= 0:
            (costerror_ + costerror_kin).backward()
            self.optimizer.step()


        AA, BB = self.forward_attn(fsignal_u)
        lossA = torch.nn.functional.mse_loss(AA, fsignal_q) + 10 * torch.nn.functional.mse_loss(BB.squeeze(0),self.eyedof)
        AA2, BB2 = self.forward_attn2(fsignal_q)
        #print(AA2.shape)
        #print(fsignal_x.shape)
        lossB = torch.nn.functional.mse_loss(AA2, fsignal_x) + 10 * torch.nn.functional.mse_loss(BB2.squeeze(0),self.eyedof)


        self.optimizer2.zero_grad()
        lossA.backward()
        self.optimizer2.step()

        self.optimizer3.zero_grad()
        lossB.backward()
        self.optimizer3.step()

        if self.re_u >= 1:
            self.violation_count += 1
            self.violation_buffer = 25
        if self.re_u < 1:
            self.costerror_ma[self.count_cul % 1000] = torch.norm(costerror_)
            self.costerror_ma_kin[self.count_cul % 1000] = torch.norm(costerror_kin)
        if self.count_cul <= 0:
            self.error_rev = 0
        else:
            self.error_rev = (torch.norm(costerror_)) / (torch.median(self.costerror_ma)) - 1
        self.violation_buffer -= 1
        if self.violation_buffer >= 0:
            print('--- Robust Masking --- ', self.violation_buffer, 'counts left', self.re_u)
        self.costerror_prev = torch.norm(costerror_kin)
        return (func_j0_ + func_j_ @ q_.unsqueeze(2)).squeeze(2), lossA, lossB, costerror_, costerror_kin
