import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

EPOCH = 1   #数据集训练几遍
BATCH_SIZE = 50  #一批数据的个数
LR = 0.001
DOWNLOAD_MNIST = False

#加载数据集
train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),  #(0,1)
    download=DOWNLOAD_MNIST
)

#生成训练器
train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
#加载测试集
test_data = torchvision.datasets.MNIST(root='./mnist/',train=False)

#!!!cuda change here!!!
test_x=Variable(torch.unsqueeze(test_data.test_data,dim=1),volatile=True).type(torch.FloatTensor)[:2000].cuda()/255 #每个像素点的原值是0-255之间
test_y=test_data.test_labels[:2000].cuda()

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        #一般来说一个大卷积层包括卷积层，激活函数和池化层
        self.conv1=nn.Sequential(
            nn.Conv2d(
                in_channels=1, #表示原始图片有多少层，也就是有多少不同种类的特征值，如RGB图片，有红，绿，蓝三个值
                out_channels=16,#表示输出多少个不同种类的特征值；也就是对同一个图片块，有16个过滤器同时工作
                kernel_size=5, #一个过滤器的长和宽都是五个像素点
                stride=1, #相邻两次扫描的图片块之间相隔几个像素点
                padding=2, #在图片周围多出2圈0值，防止过滤器的某一边超过图片边界，如何计算：if stride=1,padding=(kernel_size-1)/2，保证提取出的新图片长宽和原图一样
            ),
            nn.ReLU(),
            #池化层向下筛选需要的部分
            nn.MaxPool2d(
                kernel_size=2, #使用一个长宽为2的池化过滤器
            ),
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(16,32,5,1,2), #输入的图片有16层，输出图片有32层
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out=nn.Linear(32*7*7,10) #输入的高度是32，长宽为7，因为经过两次池化；输出为10个不同的值，即0-9

    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x) #x中的数据有四个维度:(batch,32,7,7)
        x=x.view(x.size(0),-1) #保留batch,数据变为二维：(batch,32*7*7);因为输出层只接受一维数据作为输入
        output=self.out(x)
        return output

cnn = CNN()

#!!!cuda change here!!!
cnn.cuda()

#训练过程
optimizer=torch.optim.Adam(cnn.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss() #选择误差函数

if __name__ == '__main__':
    for epoch in range(EPOCH):
        for step,(x,y) in enumerate(train_loader):

            #!!!cuda在此做了改变！！！
            b_x=Variable(x).cuda()
            b_y=Variable(y).cuda()

            output=cnn(b_x)
            loss=loss_func(output,b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step%50 == 0:
                test_output=cnn(test_x)

                #!!! cuda change here!!!
                pred_y=torch.max(test_output,1)[1].cuda().data
                accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.item(), '| test accuracy: %.2f' % accuracy)
