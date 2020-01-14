import torch
import torch.utils.data as Data

BATCH_SIZE = 5 #一批训练5个数据

x = torch.linspace(1,10,10)
y = torch.linspace(10,1,10)

#构建训练数据集，data_tensor是输入，target_tensor是输出
torch_dataset = Data.TensorDataset(data_tensor=x,target_tensor=y)

#loader使数据分为一小批一小批的
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    #是否随机打乱数据
    shuffle=True,
    #两个线程
    num_workers=2
)

#多线程要加入
if __name__ == '__main__':
    for epoch in range(3):

        #enumerate函数可以给一个序列里的对象添加一个从0开始的索引，并返回新的有索引的序列
        for step,(batch_x,batch_y) in enumerate(loader):
            #training
            print("Epoch: ",epoch,"| Step: ",step," | batch_x:",batch_x.numpy()," | batch_y:",batch_y.numpy())











