# RNN分类

import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#parameters

EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28  #按顺序一共有多少步
INPUT_SIZE = 28 #每一步输入多少数据
LR = 0.01
DOWNLOAD_MNIST = False

train_data = dsets.MNIST(root='./mnist',train=True,transform=transforms.ToTensor(),download=DOWNLOAD_MNIST)
#批训练
train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)
test_data = dsets.MNIST(root='./mnist',train=False,transform=transforms.ToTensor())
test_x = Variable(test_data.test_data,volatile=True).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels.numpy().squeeze()[:2000]

class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()

        #使用LSTM模型
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,#隐藏层的节点个数
            num_layers=1,#有几个RNN堆叠在一起
            batch_first=True,#batch_size是不是在输入数据的第一位置 输入x的格式：(batch,time_step,input_size)
        )

        self.out = nn.Linear(64,10) #定义一个全连接层，输入是64，输出是10

    def forward(self, x):
        r_out,(h_n,h_c) = self.rnn(x,None) # (h_n,h_c) 表示每一步产生的中间记忆hidden state，会被输入到下一步中；LSTM有两个中间记忆，h_n分线的，h_c是主线的
        # None表示，第一步时有没有hidden state; None表示没有，否则要输入一个hidden state
        out = self.out(r_out[:,-1,:]) # r_out的格式是(batch, time_step, input_size)；选取最后一步的输出
        return out

rnn = RNN()

#print(rnn)

#优化器
optimizer = torch.optim.Adam(rnn.parameters(),lr=LR)
#误差函数
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):
        b_x = Variable(x.view(-1,28,28)) #reshape x ； -1表示由机器推断
        b_y = Variable(y)
        output = rnn(b_x)
        loss = loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() #优化器调整参数

        if step%50==0:
            test_output = rnn(test_x.view(-1,28,28))
            pred_y = torch.max(test_output,1)[1].data.numpy().squeeze()
            accuracy = sum(pred_y==test_y) / test_y.size
            print('Epoch',epoch,' train loss: %0.4f'%loss.item(),' | test accuracy: %.2f'%accuracy)

#print 10 examples
test_output = rnn(test_x[:10].view(-1,28,28))
pred_y = torch.max(test_output,1)[1].data.numpy().squeeze()
print(pred_y,'prediction result')
print(test_y[:10],'real number')



