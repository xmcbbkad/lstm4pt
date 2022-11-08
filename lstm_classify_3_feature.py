import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader



def str2ts(x):
    time_string = x + ' GMT-0500'
    dt = datetime.datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S GMT%z')
    ts = int(dt.timestamp())
    return ts

def create_dataset_all(dt, step):
    data_X = []
    data_Y = []
    data_time_X = []
    data_time_Y = []
    for i in range(len(dt)-1, step-1, -1):
        dt_continue = True
        #print("i=",i)
        data_x = []
        data_y = None
        data_time_x = []
        data_time_y = None
        for j in range(i, i-step-1, -1):
            #print("-----j=",j)
            if j == i:   #序列中的第一个
                one = [dt.loc[j]['high']/dt.loc[j]['open'], dt.loc[j]['low']/dt.loc[j]['open'], dt.loc[j]['close']/dt.loc[j]['open']] 
                data_x.append(one)
                data_time_x.append([dt.loc[j]['date']])
            else:
                if dt.loc[j]['ts']-dt.loc[j+1]['ts'] != 60: #判断是否是连续的
                    dt_continue = False
                    break
            
                if j == i-step:  #要预测的
                    data_y = dt.loc[j]['close']/dt.loc[j+1]['close']
                    data_y = 1 if data_y>1 else 0
                    data_time_y = dt.loc[j]['date']         
                else:
                    one = [dt.loc[j]['high']/dt.loc[j+1]['close'], dt.loc[j]['low']/dt.loc[j+1]['close'], dt.loc[j]['close']/dt.loc[j+1]['close']]
                    data_x.append(one) #序列中不是第一个
                    data_time_x.append([dt.loc[j]['date']])
        
        if dt_continue:
            data_X.append(data_x)
            data_Y.append(data_y)
            data_time_X.append(data_time_x)
            data_time_Y.append(data_time_y)
        
        if(i%1000 == 0):
            print("==========={}".format(i))
                
            
    #return data_time, np.array(data_X, dtype=np.float32), np.array(data_Y, dtype=np.long)
    return data_X, data_Y, data_time_X, data_time_Y



dt = pd.read_csv('./ts_2k.csv',usecols=[0,1,2,3,4])
ts = dt['date'].apply(str2ts)
dt.insert(1, 'ts', ts)

data_X, data_Y, data_time_X, data_time_Y = create_dataset_all(dt=dt, step=10)

data_X = np.array(data_X, dtype=np.float32)
data_Y = np.array(data_Y, dtype=np.long)
per = np.random.permutation(data_X.shape[0])    #打乱后的行号
#print(per)
data_X = data_X[per, :, :]    #获取打乱后的训练数据
data_Y = data_Y[per]
#print(debug_all[1000])
#print(data_X)
#print(data_X.shape)
print(len(data_Y))
print(len(data_X))
print(data_X.shape)
#data_X[200], data_Y[200]

train_size = int(len(data_X) * 0.8)
train_X = data_X[:train_size]
train_Y = data_Y[:train_size]
test_X = data_X[train_size:]
test_Y = data_Y[train_size:]

#train_Y = train_Y.astype(np.long)
#test_Y = test_Y.astype(np.long)

train_X = train_X.reshape(-1,10,3)
train_Y = train_Y.reshape(-1, )
test_X = test_X.reshape(-1,10,3)
test_Y = test_Y.reshape(-1, )


for i in range(len(test_Y)):
    #if test_X[i][1][0] >=1:
    if test_Y[i] >=1:
        test_Y[i] = 1
    else:
        test_Y[i] = 0
    
for i in range(len(train_Y)):
    #if train_X[i][1][0] >=1:
    if train_Y[i] >=1:
        train_Y[i] = 1
    else:
        train_Y[i] = 0
        
train_x = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_Y)
test_x = torch.from_numpy(test_X)
test_y = torch.from_numpy(test_Y)


class TrainDataSet(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
        return
    
    def __getitem__(self,index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)

dt_train = TrainDataSet(train_X, train_Y)
#dt_test = TrainDataSet(test_X, test_Y)
train_loader = DataLoader(dt_train, batch_size = 64,shuffle = True, drop_last=True)


class lstm(nn.Module):
    def __init__(self,input_size=2,hidden_size=4,output_size=1,num_layer=2):
        super(lstm,self).__init__()
        self.layer1 = nn.LSTM(input_size,hidden_size,num_layer,batch_first=True)
        self.layer2 = nn.Linear(hidden_size,output_size)
        #input_size: embedding的维度，比如一个字由128维向量描述，则该值是128
        # hidden_size：lstm内部隐含层的维度
        # output_size：lstm输出向量的维度，比如一句话经过lstm变成一个64维的向量
        # lstm参数中并不定义序列的长度，就是不定义由几个lstm_cell串联
    
    def forward(self,x):
        # x的结构：[batch * seq_length * input_size]
        # seq_length：序列长度，比如要用前5个单词预测第6个，那么seq_length=5
        x,_ = self.layer1(x)
        #print("x.shape={}".format(x.shape))
        #b,s,h = x.size()
        x = x[:,-1,:]
        x = self.layer2(x)
        #print("before:",x)
        x = F.softmax(x,dim=1)
        #print("after:",x)
        #print("x.shape={}".format(x.shape))

        return x



model = lstm(3, 8, 2,1)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01)




def cal_acc(pred_y, true_y, threshold=0.5 ):
    #print("pred_y:",pred_y)
    #pre = torch.where(pred_y>threshold, 1, 0)
    pre = pred_y.clone()
    for i in range(len(pre)):
        if pre[i][1] >= threshold:
            pre[i][1] = 1
            pre[i][0] = 0
        else:
            pre[i][0] = 1
            pre[i][1] = 0
        
    #print("pre:", pre)
    #acc_all = sum(pre.indices == true_y)/len(true_y)
    TP = 0  # 1-->1
    FP = 0  # 0-->1
    FN = 0  # 1-->0
    TN = 0  # 1-->1
    ALL = 0
    for i in range(len(true_y)):
        #print(true_y[i], pre[i])
        if true_y[i] == 1 and pre[i][1] == 1:
            TP +=1
        elif true_y[i] == 0 and pre[i][1] == 1:
            FP += 1
        elif true_y[i] == 1 and pre[i][0] == 1:
            FN += 1
        elif true_y[i] == 0 and pre[i][0] == 1:
            TN += 1
        ALL += 1
    
    print("ALL:{}, TP:{}, FP:{}, FN:{}, TN:{}".format(ALL, TP, FP, FN, TN))
    
    precision = TP*1.0/(TP+FP+0.000000001)
    recall = TP*1.0/(TP+FN+0.000000001)
    print("threshold={}  precision={}  recall={}".format(threshold, precision, recall))

def eval(model, data_x, data_y):
    model_a = model.eval()
    var_data = Variable(data_x)
    #print(var_data.shape)
    pred_test = model_a(var_data) # 测试集的预测结果
    #print("pred_test.shape:{}".format(pred_test.shape))
    #print("data_y".shape)
    prediction = torch.max(pred_test, 1)
    #print(pred_test)
    #print(prediction.indices.shape)
    #print(data_y.shape)
    #print(pred_test)
    #accuracy = sum(prediction.indices == data_y)/len(data_y)
    #print("acc:{}".format(accuracy))
    
    cal_acc(pred_test, data_y, 0.5)
    cal_acc(pred_test, data_y, 0.6)
    cal_acc(pred_test, data_y, 0.7)
    cal_acc(pred_test, data_y, 0.8)
    cal_acc(pred_test, data_y, 0.9)
    print("------------------------")


    
for epoch in range(0):
    for i,data in enumerate(train_loader):
        var_x, var_y = data
        out = model(var_x)
        loss = criterion(out, var_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0: # 每 100 次输出结果
        print('Epoch: {}, Loss: {:.5f}'.format(epoch + 1, loss.data))
        eval(model, test_x, test_y)
        
    
for e in range(500000):
    var_x = Variable(train_x)
    var_y = Variable(train_y)
    #print("var_x.shape:", var_x.shape)
    #print("var_y.shape:", var_y.shape)
    #print("var_x:", var_x)
    #print("var_y:", var_y)
    # 前向传播
    out = model(var_x)
    #print(var_x.shape)
    #print(out.shape)
    #print(out.dtype)
    #print(var_y.shape)
    #print(var_y.dtype)
    #print(out)
    #print(var_y)
    loss = criterion(out, var_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #print(e)
    #print(loss.data)
    
    if (e + 1) % 1000 == 0: # 每 100 次输出结果
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.data))
        eval(model, test_x, test_y)
