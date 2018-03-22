
# coding: utf-8

# In[8]:


import sys
import csv 
import math
import numpy as np
#import matplotlib.pyplot as plt


# In[9]:

disd=0
disd=int(disd)

data = []    
#一個維度儲存一種污染物的資訊
for i in range(18):
    data.append([])

n_row = 0
text = open('train.csv', 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
    # 第0列為header沒有資訊
    if n_row != 0:
        # 每一列只有第3-27格有值(1天內24小時的數值)
        for i in range(3,27):
            if r[i] != "NR" and float(r[i]) >= 0:
                data[(n_row-1)%18].append(float(r[i]))
                
               # if((n_row-1)%18==9 and float(r[i])>200):
                    #print('nrow={} i={} v={}'.format(n_row,i,r[i]))
            else:
                data[(n_row-1)%18].append(float(0))	
    n_row = n_row+1
text.close()


# In[10]:


print('data_shape:',np.shape(data))


# In[11]:


x = []
y = []

for i in range(12):
    # 一個月取連續10小時的data可以有471筆
    for j in range(471):
        x.append([])
        # 總共有18種污染物
        for t in range(18):
            # 連續9小時
            for s in range(9):
                x[471*i+j].append(data[t][480*i+j+s] )
        y.append(data[9][480*i+j+9])
x = np.array(x)
y = np.array(y)

#print('x_shape_bf:',np.shape(x))
#delete row 1
#x = np.delete( x, range(disd*9,disd*9+8), axis=1)
#print('x_shape_af:',np.shape(x))
#x = np.concatenate((x,x**3), axis=1)
# 三次方

#x = np.concatenate((x,x**2), axis=1)
# 增加平方項

x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
# 增加bias項                

x_raw = x
y_raw = y


#X = (x_raw - np.max(x_raw))/-np.ptp(x_raw)

data_len=x_raw.shape[0]
block=int(data_len/12)
print(block)

x = np.concatenate((x_raw[0*block:1*block,:],x_raw[4*block:7*block,:],x_raw[8*block:11*block,:]),axis=0)
xtest=x_raw[block+1:2*block,:]

#y = y_raw[4*block:11*block]
y = np.concatenate((y_raw[0*block:1*block],y_raw[4*block:7*block],y_raw[8*block:11*block]),axis=0)
ytest=y_raw[block+1:2*block]


#plt.plot(range(len(ytest)),ytest)
#plt.show()

print('x_shape:',np.shape(x))
print('y_shape:',np.shape(y))
print('xt_shape:',np.shape(xtest))
print('yt_shape:',np.shape(ytest))

#print('x:',x)
#print('x_test:',x_test)
#print('y:',y)




"""
print('x_raw:',x_raw[1:10])
print('X:',X[1:10])
plt.plot(range(len(x_raw[:,1])),x_raw[:,1])
plt.show()
plt.plot(range(len(X[:,1])),X[:,1])
plt.show()
"""


# In[12]:


w = np.zeros(len(x[0])) # initial weight vector
print('len_x[0]',len(x[0]))
lr = 1.5                        # learning rate
iter = 10000                 # iteration
print('w={},lr={},'.format(w[0],lr))

# In[13]:


x_t = x.transpose()
s_gra = np.zeros(len(x[0]))

for i in range(iter):
    hypo = np.dot(x,w)
    loss = hypo - y
    cost = np.sum(loss**2) / len(x)
    cost_a  = math.sqrt(cost)
    gra = np.dot(x_t,loss)
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    w = w - lr * gra/ada
    if(i%10000==0):
        print ('iteration: %d | Cost: %f  ' % ( i,cost_a))


#print ('iteration: %d | Cost: %f  ' % ( i,cost_a))
# In[14]:


#sdirpath = sys.argv[3]
# save model
np.save( 'model.npy',w)
# read model
w = np.load('model.npy')


# In[15]:



x_test=xtest
y_test=ytest

ans_t = []
for i in range(len(x_test)):
    #ans_t.append(["id_"+str(i)])
    a = np.dot(w,x_test[i])
    ans_t.append(a)

loss_t=y_test-ans_t
sqr=loss_t**2

#sqr=sqr[:1000]

#plt.plot(range(len(sqr)),sqr)
#plt.show()

yp=y_test
ap=ans_t
print('y')
#plt.plot(range(len(yp)),yp)
#plt.show()
print('ans')
#plt.plot(range(len(ap)),ap)
#plt.show()

rms=np.mean(sqr)
rms=np.sqrt(rms)
print('rms=',rms)

log = open("logfile", "a+")
mes=str(disd)+'. rms='+str(rms)
log.write(mes)

#                      #
#                      #
# In[16]: TESTING DATA #
#                      #
#                      # 
