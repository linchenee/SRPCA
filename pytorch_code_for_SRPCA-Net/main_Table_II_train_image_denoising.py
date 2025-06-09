import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
from scipy import ndimage
from scipy.stats import ortho_group
import os
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
import platform
from argparse import ArgumentParser
from time import time
from torch.nn import init
import torch_dct as dct

parser = ArgumentParser(description='SRPCA-Net')
parser.add_argument('--train_data_name', type=str, default='YaleB_trainCR20', help='training dataset name')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=60, help='epoch number of end training')
parser.add_argument('--stage_num', type=int, default=6, help='phase number of SRPCA-Net')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--beta', type=float, default=1e-3, help='regularization parameter in the training loss function')
parser.add_argument('--slim', type=int, default=6, help='degree of slimness of a transform')
#parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--model_dir', type=str, default='models', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
args = parser.parse_args()
start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
stage_num = args.stage_num
slim = args.slim
beta = args.beta
train_data_name = args.train_data_name
#gpu_list = args.gpu_list
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

num_train = 25 # number of training samples
n1 = 192 # size of a 3D tensor
n2 = 168 # size of a 3D tensor
n3 = 64 # size of a 3D tensor
lambda_SRPCA = 1 / pow(max(n1, n2) * n3, 0.5) # regularization parameter in SRPCA
gamma_init = 0.1 # penalty parameter in ADMM
transform_init = torch.t(dct.dct(torch.eye(slim * n3), norm='ortho')).to(device) # initialization of the transformation matrix
bias = torch.zeros(n3).to(device)

train_data = sio.loadmat('../%s/%s%s' % (args.data_dir, train_data_name, '.mat'))
train_label = train_data[train_data_name]


def psnr(img1, img2):
    result = 0
    for i in range(n3):
        mse = np.mean((img1[:, :, i] - img2[:, :, i]) ** 2)
        result = result + 10 * math.log10(pow(255, 2) / mse)
    return result / n3


class SRPCANet(torch.nn.Module):
    def __init__(self):
        super(SRPCANet, self).__init__()
        self.gamma = nn.Parameter(gamma_init * torch.ones(stage_num, 1))
        self.FC_forwards = nn.Linear(in_features=n3, out_features=slim * n3, bias=False)
        self.FC_forwards.weight = nn.Parameter((transform_init[:, 0:n3]).float())

    def forward(self, M, lambda_SRPCA, L_init):
        Z1 = torch.zeros(n1, n2, slim * n3).to(device)
        Z2 = torch.zeros(n1, n2, n3).to(device)
        L = L_init
        FC_forward = self.FC_forwards
        TL = FC_forward(L)
        for i in range(stage_num):
            [TL, L, Z1, Z2] = self.func(TL, L, Z1, Z2, M, lambda_SRPCA, FC_forward, i)
        return [L, self.loss_orth()]

    def func(self, TL, L, Z1, Z2, M, lambda_SRPCA, FC_forward, i):
        N = self.shrL(TL - Z1 / self.gamma[i], 1 / (self.gamma[i] * pow(slim * n3, 0.5)))
        E = self.shrS(M - L - Z2 / self.gamma[i], lambda_SRPCA / self.gamma[i])
        temp1 = F.linear(N + Z1 / self.gamma[i], torch.t(FC_forward.weight), bias)
        temp2 = M - E - Z2 / self.gamma[i]
        L = 0.5 * (temp1 + temp2)
        TL = FC_forward(L)
        Z1 = Z1 + self.gamma[i] * (N - TL)
        Z2 = Z2 + self.gamma[i] * (L + E - M)
        return [TL, L, Z1, Z2]

    def loss_orth(self):
        result = torch.linalg.cond(self.FC_forwards.weight) - 1
        return result

    def shrL(self, X, thr):
        U1, S1, V1 = torch.svd(X.permute(2, 0, 1))
        result = torch.matmul(torch.matmul(U1, F.relu(torch.diag_embed(S1 - thr))), V1.permute(0, 2, 1)).permute(1, 2, 0)
        return result

    def shrS(self, S, thr):
        result = torch.sign(S) * F.relu(torch.abs(S) - thr)
        return result


model = SRPCANet()
if device.type != 'cpu':
    model = nn.DataParallel(model)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)


class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len


if (platform.system() == "Windows"):
    rand_loader = DataLoader(dataset=RandomDataset(train_label, num_train), batch_size=1, num_workers=0, shuffle=True)
else:
    rand_loader = DataLoader(dataset=RandomDataset(train_label, num_train), batch_size=1, num_workers=4, shuffle=True)

model_dir = "./%s/%s stage_%d slim_%d beta_%.3f" % (args.model_dir, train_data_name, stage_num, slim, beta)
log_file_name = "./%s/%s stage_%d slim_%d beta_%.3f.tLx" % (args.model_dir, train_data_name, stage_num, slim, beta)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if start_epoch > 0:
    checkpoint = torch.load('./%s/net_params_%d.pkl' % (model_dir, start_epoch))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

PSNR_EPOCH = np.zeros([end_epoch + 1, num_train], dtype=np.float32)
epoch_i = start_epoch + 1
while epoch_i <= end_epoch:
    img_count = 0
    for data in rand_loader:
        start = time()
        datas = data.float()
        L_groundtruth = (datas[0, 0, :, :, :] * (1 / 255)).to(device) # ground-truth low-rank tensor
        M = (datas[0, 1, :, :, :]).view(n1, n2, n3) # M: observation tensor = low-rank tensor + sparse tensor
        M_numpy = M.data.numpy()
        L_init_numpy = np.zeros_like(M_numpy)
        for chan in range(M_numpy.shape[2]):
            L_init_numpy[:, :, chan] = ndimage.median_filter(M_numpy[:, :, chan], size=(4, 4)) # initialization of the low-rank tensor for SRPCA-Net
        L_init = torch.from_numpy(L_init_numpy)

        [L_output, loss_constraint] = model(M.to(device), lambda_SRPCA, L_init.to(device))

        loss_discrepancy = torch.mean(torch.pow(L_output - L_groundtruth, 2))
        loss_all = loss_discrepancy + beta * loss_constraint

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        PSNR_EPOCH[epoch_i, img_count] = psnr(L_output.cpu().data.numpy() * 255, L_groundtruth.cpu().data.numpy() * 255)
        end = time()

        output_flag = "[%02d/%02d] Run time is %.4f, Total Loss: %.4f, Discrepancy Loss: %.4f, Constraint Loss: %.4f, PSNR: %.4f" % (
        epoch_i, end_epoch, (end - start), loss_all.item(), loss_discrepancy.item(), beta * loss_constraint,
        PSNR_EPOCH[epoch_i, img_count])
        print(output_flag)
        img_count += 1
    print(epoch_i)
    if np.mean(PSNR_EPOCH[epoch_i - 1, :]) - np.mean(PSNR_EPOCH[epoch_i, :]) > 2:
        checkpoint = torch.load('./%s/net_params_%d.pkl' % (model_dir, epoch_i - 2))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('change now\n')
        epoch_i -= 1
    else:
        output_file = open(log_file_name, 'a')
        output_file.write(output_flag)
        output_file.close()
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state, "./%s/net_params_%d.pkl" % (model_dir, epoch_i))
        epoch_i += 1
