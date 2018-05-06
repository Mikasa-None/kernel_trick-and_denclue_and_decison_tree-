import numpy as np
import csv
import pandas as pd

#remember to change the path
data=pd.read_csv('iris.csv')
data=np.array(data)
data=np.mat(data[:,0:4])
#print np.dot(data[0],data[1].T)
#length refers to the number of data
length=len(data)
#caiculate the kernel(k) using the kernel function
k=np.mat(np.zeros((length,length)))
for i in range(0,length):
    for j in range(i,length):
        k[i,j]=(np.dot(data[i],data[j].T))**2
        k[j,i]=k[i,j]
#print k
#save the kernel
name=range(length)
test=pd.DataFrame(columns=name,data=k)
test.to_csv('iris_k.csv')

len_k=len(k)
#centered kernel matrix
I=np.eye(len_k)
one=np.ones((len_k,len_k))
A=I-1.0/len_k*one
#print A
centered_k=np.dot(np.dot(A,k),A)
print centered_k
#save centered kernel matrix
test=pd.DataFrame(columns=name,data=centered_k)
test.to_csv('iris_ck.csv')

#normalized kernel matrix
W_2=np.zeros((len_k,len_k))
for i in range(0,len_k):
    W_2[i,i]=k[i,i]**(-0.5)
#print W_2
normalized_k=np.dot(np.dot(W_2,k),W_2)
#print normalized_k
# save normalized kernel matrix
test=pd.DataFrame(columns=name,data=normalized_k)
test.to_csv('iris_nk.csv')

#caiculate fai
fai=np.mat(np.zeros((length,10)))
for i in range(0,length):
    for j in range(0,4):
        fai[i,j]=data[i,j]**2
    for m in range(0,3):
        for n in range(m+1,4):
            j=j+1
            fai[i,j]=2**0.5*data[i,m]*data[i,n]
#print fai
#save fai
name_f=range(10)
test=pd.DataFrame(columns=name_f,data=fai)
test.to_csv('iris_fai.csv')

#calculate kernel through fai
k_f=np.mat(np.zeros((length,length)))
for i in range(0,length):
    for j in range(i,length):
        k_f[i,j]=(np.dot(fai[i],fai[j].T))
        k_f[j,i]=k_f[i,j]
test=pd.DataFrame(columns=name,data=k_f)
test.to_csv('iris_kf.csv')


#centered fai
rows=fai.shape[0]
cols=fai.shape[1]
centered_fai=np.mat(np.zeros((rows,cols)))
for i in range(0,cols):
    centered_fai[:,i]=fai[:,i]-np.mean(fai[:,i])
print centered_fai
test=pd.DataFrame(columns=name_f,data=centered_fai)
test.to_csv('iris_cf.csv')

#calculate centered kernel through centered fai
k_cf=np.mat(np.zeros((length,length)))
for i in range(0,length):
    for j in range(i,length):
        k_cf[i,j]=(np.dot(centered_fai[i],centered_fai[j].T))
        k_cf[j,i]=k_cf[i,j]
test=pd.DataFrame(columns=name,data=k_cf)
test.to_csv('iris_kcf.csv')

#normalized fai
normalized_fai=np.mat(np.zeros((rows,cols)))
for i in range(0,len(fai)):
    temp=np.linalg.norm(fai[i])
    normalized_fai[i]=fai[i]/np.linalg.norm(fai[i])
print normalized_fai
test=pd.DataFrame(columns=name_f,data=normalized_fai)
test.to_csv('iris_nf.csv')

#calculate normalized kernel through normalized fai
k_nf=np.mat(np.zeros((length,length)))
for i in range(0,length):
    for j in range(i,length):
        k_nf[i,j]=(np.dot(normalized_fai[i],normalized_fai[j].T))
        k_nf[j,i]=k_nf[i,j]
test=pd.DataFrame(columns=name,data=k_nf)
test.to_csv('iris_knf.csv')