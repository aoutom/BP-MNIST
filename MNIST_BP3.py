import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

#X=mnist.train.images[8].reshape(-1,28)
#imgplot = plt.imshow(X)
#plt.show()

#with open('结果存放.txt','a') as file_handle:   
#    x=mnist.train.images[8].reshape(-1,28)
#    for i in x:
#        for j in i:
#            file_handle.write(str(j)+" ") 
#        file_handle.write("\n")


if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    #MINBATCH
    Xc=mnist.train.images
    yc=np.zeros(len(mnist.train.labels),dtype='uint8')
    for i in range(len(mnist.train.labels)):
        for j in range(len(mnist.train.labels[i])):
            if mnist.train.labels[i][j]==1:
                yc[i]=j
    X=[Xc[i:i+5500] for i in range(0,len(Xc),5500)]
    y=[yc[i:i+5500] for i in range(0,len(yc),5500)]


    D = 784 # 样本维数（二维相当于横纵坐标）
    K = 10 # 样本种类数

    #两层神经网络尝试
    h1=100
    h2=100
    #初始化W和b
    W1=0.01*np.random.randn(D,h1)
    b1=np.zeros((1,h1))
    #print(b1)
    W2=0.01*np.random.randn(h1,h2)
    b2=np.zeros((1,h2))
    #print(b2)
    W3=0.01*np.random.randn(h2,K)
    b3=np.zeros((1,K))

    #设定步长（学习率）、和正则参数
    step=0.00001
    reg=1e-3

    
    #循环一万次
    for i in range(15000):
        j=i%10
        num_examples=X[j].shape[0]
        #每一千次步长缩小10%
        #if i%10000==0:
        #    step=0.9*step
        #使用的ReLU激活函数
        hiddenout1=np.maximum(0,np.dot(X[j],W1)+b1)
        hiddenout2=np.maximum(0,np.dot(hiddenout1,W2)+b2)
        scores=np.dot(hiddenout2,W3)+b3
        #计算每一样本的分类概率（softmax)
        scores_exp=np.exp(scores)
        prob=scores_exp/np.sum(scores_exp,axis=1,keepdims=True)
        #计算损失（交叉熵损失，加正则）
        cross_loss_prob=-np.log(prob[range(num_examples),y[j]])
        cross_loss=np.sum(cross_loss_prob)/num_examples
        reg_loss = 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2)+0.5*reg*np.sum(W3*W3)
        loss = cross_loss + reg_loss
        if i % 1000 == 0:
            print("循环次数: %d, 损失: %f" % (i, loss))

        #计算梯度
        #输出的梯度
        dprob=prob
        dprob[range(num_examples),y[j]]-=1
        #dprob=dprob/num_examples
        #scores=np.dot(hiddenout2,W3)+b3梯度
        dW3=np.dot(hiddenout2.T,dprob)
        db3=np.sum(dprob,axis=0,keepdims=True)
        dhiddenout2=np.dot(dprob,W3.T)
        dhiddenout2[hiddenout2<=0]=0
        #hiddenout2=np.maximum(0,np.dot(hiddenout1,W2)+b2)梯度
        dW2=np.dot(hiddenout1.T,dhiddenout2)
        db2=np.sum(dhiddenout2,axis=0,keepdims=True)
        dhiddenout1=np.dot(dhiddenout2,W2.T)
        dhiddenout1[hiddenout1<=0]=0
        #hiddenout1=np.maximum(0,np.dot(X,W1)+b1)梯度
        dW1=np.dot(X[j].T,dhiddenout1)
        db1=np.sum(dhiddenout1,axis=0,keepdims=True)
        #梯度正则
        dW3+=reg*W3
        dW2+=reg*W2
        dW1+=reg*W1

        #参数更新
        W1+=-step*dW1
        b1+=-step*db1
        W2+=-step*dW2
        b2+=-step*db2
        W3+=-step*dW3
        b3+=-step*db3

    #测试
    Xc=mnist.train.images
    yc=np.zeros(len(mnist.train.labels),dtype='uint8')
    for i in range(len(mnist.train.labels)):
        for j in range(len(mnist.train.labels[i])):
            if mnist.train.labels[i][j]==1:
                yc[i]=j
    hiddenout1=np.maximum(0,np.dot(Xc,W1)+b1)
    hiddenout2=np.maximum(0,np.dot(hiddenout1,W2)+b2)
    scores=np.dot(hiddenout2,W3)+b3
    predicted_class = np.argmax(scores, axis=1)
    print('训练集准确度: %.2f' % (np.mean(predicted_class == yc)))
    # 测试2
    X2=mnist.validation.images
    y2=np.zeros(len(mnist.validation.labels),dtype='uint8')
    for i in range(len(mnist.validation.labels)):
        for j in range(len(mnist.validation.labels[i])):
            if mnist.validation.labels[i][j]==1:
                y2[i]=j
    hiddenout1=np.maximum(0,np.dot(X2,W1)+b1)
    hiddenout2=np.maximum(0,np.dot(hiddenout1,W2)+b2)
    scores=np.dot(hiddenout2,W3)+b3
    predicted_class = np.argmax(scores, axis=1)
    print('验证集准确度: %.2f' % (np.mean(predicted_class == y2)))
    #测试1
    X1=mnist.test.images
    y1=np.zeros(len(mnist.test.labels),dtype='uint8')
    for i in range(len(mnist.test.labels)):
        for j in range(len(mnist.test.labels[i])):
            if mnist.test.labels[i][j]==1:
                y1[i]=j
    hiddenout1=np.maximum(0,np.dot(X1,W1)+b1)
    hiddenout2=np.maximum(0,np.dot(hiddenout1,W2)+b2)
    scores=np.dot(hiddenout2,W3)+b3
    predicted_class = np.argmax(scores, axis=1)
    print('测试集准确度: %.2f' % (np.mean(predicted_class == y1)))
