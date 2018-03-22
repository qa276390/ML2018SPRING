
# coding: utf-8

# In[8]:


import sys
import csv 
import math
import numpy as np


#                      #
#                      #
# In[16]: TESTING DATA #
#                      #
#                      # 

test_x = []
n_row = 0
testfile = sys.argv[1]
text = open(testfile ,"r")
row = csv.reader(text , delimiter= ",")

for r in row:
    if n_row %18 == 0:
        test_x.append([])
        for i in range(2,11):
            test_x[n_row//18].append(float(r[i]) )
    else :
        for i in range(2,11):
            if r[i] !="NR" and float(r[i]) >= 0:
                test_x[n_row//18].append(float(r[i]))
            else:
                test_x[n_row//18].append(0)
    n_row = n_row+1
text.close()
test_x = np.array(test_x)
#test_x = np.delete(test_x, range(disd*9,disd*9+8), axis=1)

#test_x = np.concatenate((test_x,test_x**3), axis=1)
# 增加三次方

#test_x = np.concatenate((test_x,test_x**2), axis=1)
# 增加平方項

test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)
# 增加bias項  


# In[17]:

w = np.load('model.npy')

ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    if(a<0):
        a=0
    ans[i].append(a)

filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()

