import os
import time
from datetime import datetime
import torch
from torch import nn
from torch.autograd import Variable
import train
from Function.function import MAE, RMSE, get_cores, update_Ms, update_cores, get_fold_tensor
import numpy as np
import h5py
import tensorly as tl
from ConvGRU import ConvGRU
from Function.preprocess import process_IF

if __name__ == '__main__':

    # set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # detect if CUDA is available or not
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor  # computation in GPU
        device = torch.device("cuda")
    else:
        dtype = torch.FloatTensor
        device = torch.device("cpu")

    t=time.time()

    # data = np.load("data/datset_file_name")["volume"][:, :, :, 1]
    # data = data[:, :, :]
    # # data = data.reshape(-1, 1, 10, 20)
    # data = data.reshape(-1, 10, 20)

    filename = "data/BJ16_M32x32_T30_InOut.h5"
    f = h5py.File(filename)
    data = f["data"][:, 1, :, :]
    IF = process_IF(dir="data/TaxiBJ/")[:, :]

    data=np.array(data)

    batch_size = 48
    height = 10
    width = 10
    channels = 1
    hidden_dim = [8,1]
    kernel_size = (3,3) # kernel size for two stacked hidden layer
    num_layers = 2  # number of stacked hidden layer
    # 数据维度：(seq, input_size)
    seq = data.shape[0]
    input_size = data.shape[1]
    # 定义每个样本中的时间步数量和预测步数
    seq_length = 12  # 过去的时间步数量
    pred_length = 1  # 预测的时间步数量
    k=10
    Rs=[height,width]

    x=[ele for ele in data]


    model = ConvGRU(input_size=(height, width),  # (h,w)
                    input_dim=channels,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    num_layers=num_layers,
                    batch_first=True,
                    bias=True,
                    dtype=dtype,
                    return_all_layers=False,
                    )
    model.to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    Ms = [np.random.random([j, r]) for j, r in zip(list(x[0].shape), Rs)]
    for epoch in range(25):  # 进行模型训练
        # initilizer
        cores=get_cores(x, Ms)
        train_data = []
        factor = []
        for i in range(seq-seq_length-pred_length+1):
            train_data.append(cores[i:i + seq_length+1])
        for i in range(seq_length+pred_length-1,seq):
            factor.append(IF[i])
        train_data=np.array(train_data)
        train_data = torch.tensor(train_data, dtype=torch.float32)
        factor=np.array(factor)
        factor = torch.tensor(factor, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(train_data[:, :, :,:],factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False,drop_last=True)


        # factor=np.array(train_data)
        # factor = torch.tensor(train_data, dtype=torch.float32)
        # dataset2 = torch.utils.data.TensorDataset(factor[:, :, :,:])
        # dataloader2 = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False,drop_last=True)

        # get core_pred
        for i in range(30):
            loss_list=[]
            cores_pred_list=[]
            for ele in dataloader:
                inputs=ele[0]
                factor=ele[1]
                inputs=inputs[:,:-1,:,:].reshape(batch_size,seq_length,1,Rs[0],Rs[1])
                labels=inputs[:,-1,:,:].reshape(batch_size,pred_length,1,Rs[0],Rs[1])

                _, cores_pred = model(inputs.to(device),factor.to(device))
                cores_pred_list.append(cores_pred)
                loss = loss_function(cores_pred, labels.to(device))
                loss_list.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # if i==19:
                #     print(f"epoch={epoch}, labels={labels}, pred={cores_pred}")
            if epoch>=0:
                    # print('epoch: {} , loss: {:.4}'.format(i, loss.item()))
                print(f"epoch: {epoch} , loss: {np.array(loss_list).mean():.4}  \n")
            else:
                print(f" epoch={epoch}, i={i}\n")


        cores_pred_list=[ele.cpu().detach().numpy() for ele in cores_pred_list]
        cores_pred_list=np.array(cores_pred_list).reshape(-1,Rs[0],Rs[1])
        cores_pred_list = [ele for ele in cores_pred_list]
        for n in range(len(Rs)):
            unfold_cores=update_cores( n, Ms, x, cores, cores_pred_list,seq_length)
            cores = get_fold_tensor(unfold_cores, n, cores_pred_list[0].shape)
            Ms=update_Ms(Ms, x, unfold_cores, n,seq_length)


    cores = get_cores(x, Ms)
    train_data = []
    factor=[]
    for i in range(seq - seq_length - pred_length + 1):
        train_data.append(cores[i:i + seq_length + 1])
    for i in range(seq_length + pred_length - 1, seq):
        factor.append(IF[i])
    train_data = np.array(train_data)
    train_data = torch.tensor(train_data, dtype=torch.float32)
    factor = np.array(factor)
    factor = torch.tensor(factor, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(train_data[-48:, :, :, :], factor[-48:,:])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False, drop_last=True)

    model.eval()
    pred = []
    x=[]
    mae=[]
    rmse=[]
    with torch.no_grad():
        for ele in dataloader:
            inputs = ele[0][:, :-1, :, :]
            labels = ele[0][:, -1, :, :]
            factor = ele[1]
            inputs = inputs.reshape(batch_size,seq_length, 1, Rs[0],Rs[1])
            _, cores_pred = model(inputs.to(device),factor.to(device))
            cores_pred = list(cores_pred.cpu().reshape(-1,Rs[0],Rs[1]))
            for ele2,ele3 in zip(cores_pred,labels):
                a=tl.tenalg.multi_mode_dot(ele2.numpy(), Ms)
                b=tl.tenalg.multi_mode_dot(ele3.numpy(), Ms)
                pred.append(a)
                x.append(b)
                mae.append(MAE(a,b))
                rmse.append(RMSE(a, b))
    print(MAE(pred,x))
    print(RMSE(pred, x))
    t2 = time.time()
    T2 = time.time()
    print('程序运行时间:%s毫秒' % ((t2 - t) * 1000))
    print()




