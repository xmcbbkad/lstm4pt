import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pytz
import torch
from torch import nn
from torch.autograd import Variable

def str2ts(x):
    time_string = x + ' GMT-0500'
    dt = datetime.datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S GMT%z')
    ts = int(dt.timestamp())
    return ts


def create_dataset(dt, step):
    data_X = []
    data_Y = []
    for i in range(len(dt)- step):
        dt_continue = True
        line = '1'
        data_x = [[1.0],]
        data_y = 0
        for j in range(i+step-1, i-1, -1):
            #print(j)
            if dt.loc[j]['ts']-dt.loc[j+1]['ts'] != 60:
                dt_continue = False
                break
            
            value = dt.loc[j]['close']/dt.loc[j+1]['close']
            line += "\t{}".format(value)
            if j==i:
                data_y = value
            else:
                data_x.append([value])
        if not dt_continue:
            #print(i)
            #return
            continue
        data_X.append(data_x)
        data_Y.append(data_y)
        if(i%1000 == 0):
            print(line)
            print("==========={}".format(i))
        
    return np.array(data_X, dtype=np.float32), np.array(data_Y, dtype=np.float32)

class lstm(nn.Module):
    def __init__(self,input_size=2,hidden_size=4,output_size=1,num_layer=2):
        super(lstm,self).__init__()
        self.layer1 = nn.LSTM(input_size,hidden_size,num_layer)
        self.layer2 = nn.Linear(hidden_size,output_size)
    
    def forward(self,x):
        x,_ = self.layer1(x)
        s,b,h = x.size()
        x = x.view(s*b,h)
        x = self.layer2(x)
        x = x.view(s,b,-1)
        return x



dt = pd.read_csv('./ts_100k.csv',usecols=[0,1,2,3,4])

ts = dt['date'].apply(str2ts)
dt.insert(1, 'ts', ts)

#dt.head(10)

data_X, data_Y = create_dataset(dt=dt, step=5)

train_size = int(len(data_X) * 0.7)
train_X = data_X[:train_size]
train_Y = data_Y[:train_size]
test_X = data_X[train_size:]
test_Y = data_Y[train_size:]

train_X = train_X.reshape(-1, 1, 5)
train_Y = train_Y.reshape(-1, 1, 1)
test_X = test_X.reshape(-1, 1, 5)
test_Y = test_Y.reshape(-1, 1, 1)

train_x = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_Y)
test_x = torch.from_numpy(test_X)
test_y = torch.from_numpy(test_Y)

model = lstm(5, 8,1,1)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

for e in range(500):
    var_x = Variable(train_x)
    var_y = Variable(train_y)
    # 前向传播
    out = model(var_x)
    loss = criterion(out, var_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #print(e)
    #print(loss.data)
    
    #if (e + 1) % 20 == 0: # 每 100 次输出结果
    print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.data))
    
    
    model_a = model.eval() # 转换成测试模式

    var_data = Variable(test_x)
    pred_test = model_a(var_data) # 测试集的预测结果
    # 改变输出的格式
    pred_test = pred_test.view(-1).data.numpy()
    
    right = 0
    wrong = 0
    diff_random = 0
    diff_pred = 0
    for i in range(len(test_y)):
        y = test_y.view(-1).data.numpy()[i]
        pred_y = pred_test[i]
        #print(y, pred_y)
        if (y >1 and pred_y>1) or (y<1 and pred_y<1):
            right +=1
        else:
            wrong +=1
        diff_random += abs(y-1)
        diff_pred += abs(y-pred_y)

    print("all:{}  right:{}   wrong:{}   ratio:{}".format(len(test_y),right, wrong, right/len(test_y) ))
    print("diff_random:{}  diff_pred:{}".format(diff_random/len(test_y), diff_pred/len(test_y)))



model = model.eval() # 转换成测试模式

var_data = Variable(test_x)
pred_test = model(var_data) # 测试集的预测结果
# 改变输出的格式
pred_test = pred_test.view(-1).data.numpy()

right = 0
wrong = 0
diff_random = 0
diff_pred = 0
for i in range(len(test_y)):
    y = test_y.view(-1).data.numpy()[i]
    pred_y = pred_test[i]
    print(y, pred_y)
    if (y >1 and pred_y>1) or (y<1 and pred_y<1):
        right +=1
    else:
        wrong +=1
    diff_random += abs(y-1)
    diff_pred += abs(y-pred_y)

print("all:{}  right:{}   wrong:{}   ratio:{}".format(len(test_y),right, wrong, right/len(test_y) ))
print("diff_random:{}  diff_pred:{}".format(diff_random/len(test_y), diff_pred/len(test_y)))
