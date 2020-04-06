import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

#把杆子立起来

# 超参数
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # 最优选择动作百分比
GAMMA = 0.9                 # 奖励递减参数
TARGET_REPLACE_ITER = 100   # Q 现实网络的更新频率
MEMORY_CAPACITY = 2000      # 记忆库大小
env = gym.make('CartPole-v0')   # 立杆子游戏
env = env.unwrapped
N_ACTIONS = env.action_space.n  # 杆子能做的动作
N_STATES = env.observation_space.shape[0]   # 杆子能获取的环境信息数
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

#对于动作价值的评估
class Net(nn.Module):
    def __init__(self,):
        super(Net,self).__init__()
        self.fc1=nn.Linear(N_STATES,10) #输入观测值
        self.fc1.weight.data.normal_(0,0.1) #初始化权重
        self.out=nn.Linear(10,N_ACTIONS) #输出对于观测值，做出每个动作的得分
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value=self.out(x)
        return actions_value

class DQN(object):
    def __init__(self):
        #eval和target这两个网络的结构相同，但参数不同
        self.eval_net, self.target_net=Net(),Net()

        self.learn_step_counter=0 #学习的步数
        self.memory_counter=0 #记忆的数量
        self.memory=np.zeros((MEMORY_CAPACITY,N_STATES*2+2)) #记忆库全为0；行数为可以存多少记忆，列为state的数量加action加reward
        self.optimizer=torch.optim.Adam(self.eval_net.parameters(),lr=LR)
        self.loss_func=nn.MSELoss()

    def choose_action(self,x):
        #x是观测值
        x = torch.unsqueeze(torch.FloatTensor(x),0)
        #使用随机数来生成一个概率
        if np.random.uniform() < EPSILON:
            #greedy: 选择价值最高的动作，EPSILON预设值为0.9，即90%的情况下，greedy
            actions_value=self.eval_net.forward(x)
            action = torch.max(actions_value,1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index

        else:#随机选择一个动作
            action = np.random.randint(0,N_ACTIONS)

        return action

    def store_transition(self,s,a,r,s_):
        #存储记忆库
        transition=np.hstack((s,[a,r],s_))
        #如果存储的记忆超过了存储总量，则进行替换
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index,:]=transition
        self.memory_counter+=1

    def learn(self):
        #学习过程，学习存储好的记忆
        #是否更新target_net; TARGET_REPLACE_ITER表示学习多少代之后进行更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        self.learn_step_counter+=1

        #训练eval_net

        #随机抽取一些记忆
        sample_index=np.random.choice(MEMORY_CAPACITY,BATCH_SIZE)
        b_memory = self.memory[sample_index,:]
        #把存储的memeory分开
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        q_eval = self.eval_net(b_s).gather(1,b_a) #输入现在的状态，得到所有动作的价值; 只要选择的动作b_a对应的价值

        #下一个状态，每个动作的对应价值；注意使用target_net!
        q_next = self.target_net(b_s_).detach()  # q_next 不进行反向传递误差, 所以 detach

        #“真实”收益是当前这个动作的收益再加上下一个动作的最大收益
        q_target=b_r+GAMMA*q_next.max(1)[0].view(BATCH_SIZE, 1)

        loss = self.loss_func(q_eval,q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn=DQN()

for i_episode in range(500):
    s = env.reset()

    ep_r = 0

    while True:
        env.render()    # 显示实验动画
        a = dqn.choose_action(s)

        # 选动作, 得到环境反馈
        s_, r, done, info = env.step(a)


        # 修改 reward, 使 DQN 快速学习
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        # 存记忆
        dqn.store_transition(s, a, r, s_)

        ep_r=ep_r+r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn() # 记忆库满了就进行学习
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))

        if done:    # 如果回合结束, 进入下回合
            break

        s = s_


























