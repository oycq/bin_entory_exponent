import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import sys
import cifar_dataset as my_dataset 
import cv2

torch.manual_seed(0)

IF_WANDB = 0
IF_SAVE = 0
SIX = 6
BATCH_SIZE = 100
WORKERS = 15
CLASS = 10
TESTING_LEN = 10000
LUT_RANDN_K = 4
CONNECT_RANDN_K = 4

if IF_WANDB:
    import wandb
    wandb.init(project = 'lut_cifar')#, name = '.')



dataset = my_dataset.MyDataset(train = True, margin = 2, noise_rate = 0.01)
dataset_test = my_dataset.MyDataset(train = False)
data_feeder = my_dataset.DataFeeder(dataset, BATCH_SIZE, num_workers = WORKERS)
images_t,labels_t = dataset_test.get_all()



class Quantized(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        r = torch.cuda.FloatTensor(input.shape).uniform_()
        return (input >= r).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class LutLayer(nn.Module):
    def __init__(self, depth):
        super(LutLayer, self).__init__()
        p_q_2_lut_table = torch.zeros(SIX*2, 2**SIX)
        for i in range(2**SIX):
            bins = ('{0:0%db}'%SIX).format(i)
            for j in range(SIX):
                if bins[j] == '0':
                    p_q_2_lut_table[j+SIX][i] = 1
                else:
                    p_q_2_lut_table[j][i] = 1
        self.p_q_2_lut_table = p_q_2_lut_table.cuda()
        lut = torch.randn(depth, 2 ** SIX) * LUT_RANDN_K
        self.lut = torch.nn.Parameter(lut)

    def forward(self, inputs, infer=False):
        lut_infer = torch.zeros_like(self.lut) 
        lut_infer[self.lut > 0] = 1
        eps = 1e-7
        p_q = inputs.unsqueeze(2).repeat(1,1,2,1)
        p_q[:,:,0,:] = 1 - p_q[:,:,0,:]
        p_q = torch.nn.functional.relu(p_q) + eps
        p_q = p_q.view(p_q.shape[0], p_q.shape[1], -1)
        p_q_log = p_q.log()
        lut_p = (p_q_log.matmul(self.p_q_2_lut_table)).exp()
        if infer:
            output = (lut_p * lut_infer).sum(-1)
        else:
            output = (lut_p * torch.sigmoid(self.lut)).sum(-1)
        return output


class MyCNN(nn.Module):
    def __init__(self, inputs_d, kernal_d, kernal_a, stride):
        super(MyCNN, self).__init__()
        self.stride = stride
        self.kernal_d = kernal_d
        connect_kernal = torch.randn(kernal_d*SIX, inputs_d, kernal_a, kernal_a) *CONNECT_RANDN_K
        self.connect_kernal = torch.nn.Parameter(connect_kernal)
        self.lut_layer = LutLayer(kernal_d)
        self.quantized = Quantized.apply

    def get_infer_kernal(self):
        connect_kernal_shape = self.connect_kernal.shape
        connect_kernal= self.connect_kernal.view(self.kernal_d*SIX, -1)
        max_idx = connect_kernal.argmax(-1)
        connect_kernal_infer = torch.zeros_like(connect_kernal).\
                scatter(1, max_idx.unsqueeze(1), 1.0)
        connect_kernal_infer = connect_kernal_infer.view(connect_kernal_shape)
        return connect_kernal_infer

    def forward(self, inputs, infer=False, quant=False):
        connect_kernal_shape = self.connect_kernal.shape
        connect_kernal= self.connect_kernal.exp()
        connect_kernal = connect_kernal.view(self.kernal_d*SIX, -1)
        connect_kernal = connect_kernal / connect_kernal.sum(-1).unsqueeze(-1)
        connect_kernal = connect_kernal.view(connect_kernal_shape)
        connect_kernal_infer = self.get_infer_kernal()
        if infer:
            x = F.conv2d(inputs, connect_kernal_infer, stride=self.stride)
        else:
            x = F.conv2d(inputs, connect_kernal, stride=self.stride)
        x = x.permute(0,2,3,1)
        output_shape = (x.shape[0],x.shape[1],x.shape[2],self.kernal_d)
        x = x.reshape(-1, self.kernal_d, SIX)
        x = self.lut_layer(x, infer)
        if quant:
            x = self.quantized(x)
        x = x.view(output_shape).permute(0,3,1,2)
        return x


class Net(nn.Module):
    def __init__(self, input_size=784):
        super(Net, self).__init__()
        self.cnn1 = MyCNN(inputs_d=24, kernal_d=8,  kernal_a=1, stride = 1) #(32,32,8)8192
        self.cnn2 = MyCNN(inputs_d=8, kernal_d=16,  kernal_a=6, stride = 2) #(14,14,16)3136
        self.cnn3 = MyCNN(inputs_d=16, kernal_d=32,  kernal_a=6, stride = 1) #(9,9,32)2592
        self.cnn4 = MyCNN(inputs_d=32, kernal_d=64, kernal_a=4, stride = 1) #(6,6,64)2304
        self.cnn5 = MyCNN(inputs_d=64, kernal_d=1280, kernal_a=6, stride = 1) #(1,1,1280)1280
        score_K = torch.zeros(1) + 3
        self.score_K = torch.nn.Parameter(score_K)

    def forward(self, inputs, infer=False):
        x = inputs
        x = self.cnn1(x,infer,quant=True)
        x = self.cnn2(x,infer,quant=True)
        x = self.cnn3(x,infer,quant=True)
        x = self.cnn4(x,infer,quant=True)
        x = self.cnn5(x,infer,quant=True)
        x = x.view(x.shape[0], -1)
        x = (x - 0.5) * self.score_K
        return x


def get_loss_acc(x, labels):
    x = x.view(x.shape[0], CLASS, -1)
    x = x.mean(-1)
    accurate = (x.argmax(-1) == labels.argmax(-1)).float().mean() * 100
    x = x.exp()
    x = x / x.sum(-1).unsqueeze(-1)
    x = -x.log()
    loss = (x * labels).sum(-1).mean()
    return loss, accurate

def get_fpga_acc(train=True):
    acc = 0
    with torch.no_grad():
        for i in range(TESTING_LEN//BATCH_SIZE):
            if train:
                images, labels = data_feeder.feed()
            else:
                a = i * BATCH_SIZE
                b = i * BATCH_SIZE + BATCH_SIZE
                images, labels = images_t[a:b], labels_t[a:b]
            x = net(images,infer=True)
            loss, accurate = get_loss_acc(x, labels)
            acc += accurate.item() * 1.0 * BATCH_SIZE / TESTING_LEN
    if train:
        print('train_acc:%8.3f%%   train_loss:%8.3f'%(acc,loss))
        if IF_WANDB:
            wandb.log({'train_acc':acc})
    else:
        print(' test_acc:%8.3f%%    test_loss:%8.3f'%(acc,loss))
        if IF_WANDB:
            wandb.log({'test_acc':acc})



net = Net().cuda()
optimizer = optim.Adam(net.parameters())

for i in range(100000000):
    images, labels = data_feeder.feed()
    optimizer.zero_grad()
    x = net(images)
    loss,acc = get_loss_acc(x,labels)
    loss.backward()
    if i % 50 == 0:
        print('%5d  %7.3f  %7.4f'%(i,acc,loss))
        if IF_WANDB:
            wandb.log({'acc':acc})
    if i % 500 == 0:
        get_fpga_acc(train = True)
        get_fpga_acc(train = False)
    if i % 4999 == 0 and IF_SAVE:
        torch.save(net.state_dict(), 'lut_cnn_cifar.model')
    optimizer.step()


