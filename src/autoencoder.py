import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# 超参数
EPOCH = 10
BATCH_SIZE = 64
LR = 0.005
DOWNLOAD_MNIST = False   # 下过数据的话, 就可以设置成 False
N_TEST_IMG = 5          # 到时候显示 5张图片看效果, 如上图一

train_data = torchvision.datasets.MNIST(root='./mnist/',train=True,transform=torchvision.transforms.ToTensor(),download=DOWNLOAD_MNIST)
train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)

#plt one example
'''
print(train_data.data.size())
print(train_data.targets.size())
plt.imshow(train_data.data[2].numpy(),cmap='gray') #cmap设置colormap,即色图，gray表示灰度图
plt.title('%i'%train_data.targets[2])
plt.show()
'''

#build the autoencoder
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28,128),
            nn.Tanh(),
            nn.Linear(128,64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(), #把输出值变为0~1之间，因为原图是黑白
        )

    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

autoencoder = AutoEncoder()
optimizer = torch.optim.Adam(autoencoder.parameters(),lr=LR)
loss_func = nn.MSELoss()

# initialize figure
#subplot把多个图画到一个图中; a相当于是存储每个子图的数组
figure, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.ion()   # continuously plot

# original data (first row) for viewing
view_data = train_data.train_data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor)/255.
#a[0]表示第一行要画什么
for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray')
    #set_xsticks设置坐标轴
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())

for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):
        b_x = x.view(-1,28*28) #用同一份数据进行输入和对比
        b_y = x.view(-1,28*28) #reshape为2维数组，batch未知用-1，第二维长度是28*28；(batch, 28*28)；把图片摊平
        b_label = y

        encoded,decoded = autoencoder(b_x)
        loss = loss_func(decoded,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%100 == 0:
            print("Epoch: ",epoch, " | train loss: %.4f" % loss.item())
            _, decoded_data=autoencoder(view_data)
            for i in range(N_TEST_IMG):
                #第二行子图需要不断变化
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())
            plt.draw()
            plt.pause(0.05)

plt.ioff()
plt.show()

