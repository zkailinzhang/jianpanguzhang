from DPC import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


#计算保存记忆矩阵的Temp矩阵
def Temp_MemMat(memorymat,Temp_name):
    memorymat_row = memorymat.shape[0]
    Temp = np.zeros((memorymat_row, memorymat_row))
    for i in range(memorymat_row):
        for j in range(memorymat_row):
            Temp[i, j] = np.linalg.norm(memorymat[i] - memorymat[j])
    np.save(Temp_name,Temp)

#MSET计算，被MSETs调用
def MSET(memorymat,Kobs,Temp_name):#Temp为临时计算的矩阵
    memorymat_row=memorymat.shape[0]
    Kobs_row=Kobs.shape[0]
    Temp=np.load(Temp_name)
    Temp1=np.zeros((memorymat_row,Kobs_row))
    for m in range(memorymat_row):
        for n in range(Kobs_row):
            Temp1[m,n]=np.linalg.norm(memorymat[m] - Kobs[n])
    Kest=np.dot(np.dot(memorymat.T,(np.linalg.pinv(Temp))),Temp1)
    Kest=Kest.T
    return Kest

#判断输入的观测向量，再传到相应记忆矩阵中，得到估计值，调用MSET
def MSETs(memorymat1_name,memorymat2_name,memorymat3_name,Kobs):
    row_Kobs=Kobs.shape[0]
    col_Kobs = Kobs.shape[1]
    Kest=np.zeros((row_Kobs,col_Kobs))
    for t in range(row_Kobs):
        if Kobs[t,col_Kobs-1]<1/3:
            Kest[t] = MSET(memorymat1_name,Kobs[t:t+1,:],'Temp_low.npy')
        elif Kobs[t,col_Kobs-1]>2/3:
            Kest[t] = MSET(memorymat3_name, Kobs[t:t+1,:],'Temp_hig.npy')
        else:
            Kest[t] = MSET(memorymat2_name,Kobs[t:t+1,:],'Temp_med.npy')
    return Kest

#各变量及其误差的可视化
def pic_vars(label,Kobs,Kest):
    col_num=Kobs.shape[1]
    e=np.ones((Kobs.shape[0],Kobs.shape[1]))
    plt.ion()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 图片显示中文
    plt.rcParams['axes.unicode_minus'] = False
    for i in range(col_num):
        plt.figure()
        plt.subplot(211)
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        plt.plot(Kobs[:, i], 'steelblue', label='观测值', lw=1.5)
        plt.plot(Kest[:, i], 'indianred', label='估计值', lw=1.5)
        plt.legend(loc='upper right', fontsize=13)
        plt.xlabel('样本序号', fontsize=13)
        plt.ylabel(label[i], fontsize=13, verticalalignment='bottom')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.subplot(212)
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        e[:, i] = (np.abs(Kobs[:, i] - Kest[:, i]) / Kobs[:, i]) * 100
        plt.plot(e[:, i], 'peru', lw=1)  # 偏离度
        plt.xlabel('样本序号', fontsize=20)
        plt.ylabel('相对误差/%', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=18)
        plt.ioff()
        plt.show()
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
