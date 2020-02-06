# RNN回归

#用sin 预测 cos
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

#为CPU设置种子，用于生成随机数
torch.manual_seed(1)

TIME_STEP =10
INPUT_SIZE=1
LR = 0.02

#show data
#steps = np.linspace(0,np.pi*2,100,dtype=np.float32) #(start,stop,num) 在指定的范围内给出间隔均匀的数据
#x_np = np.sin(steps)
#y_np = np.cos(steps)
#plt.plot(steps,y_np,'r-',label="target:cos")
#plt.plot(steps,x_np,'b-',label="input:sin")
#plt.legend(loc='best')
#plt.show()

class RNN(nn.Module):

    def __init__(self):
        super(RNN,self).__init__()
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(32,1)

    def forward(self, x, h_state):
        #RNN forward以一个time_step为单位
        #h_state 每10个输入才会产生一个，而r_out每一个输入都会产生一个
        r_out,h_state = self.rnn(x,h_state)
        outs=[]
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:,time_step,:]))
        return torch.stack(outs,dim=1),h_state

rnn = RNN()

optimizer = torch.optim.Adam(rnn.parameters(),lr=LR)
lossfunc = nn.MSELoss()

h_state = None
for step in range(80):
    start,end = step*np.pi, (step+1)*np.pi
    steps = np.linspace(start,end, TIME_STEP, dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)

    x = Variable(torch.from_numpy(x_np[np.newaxis,:, np.newaxis])) #reshape x; np.newaxis，在这一维度增加一维
    y = Variable(torch.from_numpy(y_np[np.newaxis,:, np.newaxis]))

    prediction, h_state = rnn(x,h_state)
    h_state = h_state.data #important!!

    loss = lossfunc(prediction,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    plt.plot(steps,y_np.flatten(),'r-')
    plt.plot(steps,prediction.data.numpy().flatten(),'b-')
    plt.draw()
    plt.pause(0.05)

plt.ioff()
plt.show()








