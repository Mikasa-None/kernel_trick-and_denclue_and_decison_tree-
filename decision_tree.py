#-*-coding:utf-8-*-
import numpy as np
import pandas as pd
import math

#计算某一属性里面的能取得最大Gain的对应的split_point，以及其gain
def evaluate_numeric_attribute(D,attribute,sigma):
    featureList = [instance[attribute] for instance in D]
    uniqueVals = set(featureList)
    uniqueVals=sorted(list(uniqueVals))
    split_point=0.0
    BestGain=0.0
    for i in range(len(uniqueVals)-1):
        v=(uniqueVals[i]+uniqueVals[i+1])/2.0
        DY,DN=splitPointDataset(D,attribute,v)
        if len(DY)<=sigma or len(DN)<=sigma:
            continue
        baseEnt = calEnt(D)
        Gain=baseEnt-float(len(DY))/len(D)*calEnt((DY))-float(len(DN))/len(D)*calEnt(DN)
        if Gain>BestGain:
            BestGain=Gain
            split_point=v
    return split_point,BestGain

#计算数据的类别属性里面比例最大的类别的label以及其纯度
def majorityLabel_purity(classList):
    if len(classList)==0:
        return None,0.0,0
    classCount={}
    for n in classList:
        if n not in classCount.keys():
            classCount[n]=0
        classCount[n]+=1
    #print classCount
    sortedCC=sorted(classCount.iteritems(),key=lambda x:x[1],reverse=True)
    majorityLabel=sortedCC[0][0]
    purity=1.0*sortedCC[0][1]/len(classList)
    num=len(classList)
    return [majorityLabel,purity,num]

#calculate entropy
def calEnt(D):
    num=len(D)
    labelCount={}
    for instance in D:
        currentLabel=instance[-1]
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel]=0
        labelCount[currentLabel]+=1
    Ent=0.0
    for key in labelCount:
        prob=float(labelCount[key])/num
        Ent-=prob*math.log(prob,2)
    return Ent

#通过split_point对数据进行split处理
def splitPointDataset(D,attribute,value):
    DY=[]
    DN=[]
    for instance in D:
        if instance[attribute]<=value:
            DY.append(instance)
        if instance[attribute]>value:
            DN.append(instance)
    return DY,DN

#选择最优的属性
def chooseBestFeature(dataSet,sigma,pai):
    classList = [instance[-1] for instance in dataSet]
    majorityLabel,purity,num=majorityLabel_purity(classList)
    if len(set(classList))==1:     # 停止条件 1
        return None,majorityLabel,purity,num               # 返回标签的多数类别作为叶子节点
    if purity>=pai:               # 停止条件 2   如果所给的数据纯度达到标准，则停止划分
        return None,majorityLabel,purity,num
    featnum=len(dataSet[0])-1
    bestFeature=-1
    bestGain=0.0
    bestSplit_point=0.0
    for i in range(featnum):
        split_point,Gain = evaluate_numeric_attribute(dataSet,i,sigma)
        if Gain>bestGain:
            bestGain=Gain
            bestFeature=i
            bestSplit_point=split_point
    dY, dN = splitPointDataset(dataSet,bestFeature,bestSplit_point)
    if len(dY) < sigma or len(dN) < sigma:        # 停止条件 3
        return None,majorityLabel,purity,num
    return bestFeature,bestSplit_point,bestGain,num

#构造决策树
def createTree(D,sigma,pai):
    bestFeature, split_point,purity,num= chooseBestFeature(D,sigma,pai)
    if bestFeature == None: return {'label':split_point,'purity':purity,'number':num}
    DY, DN = splitPointDataset(D, bestFeature, split_point)
    Decision_tree = {}
    Decision_tree['attribute'] = bestFeature
    Decision_tree['split_point'] = split_point
    Decision_tree['Gain']=purity
    Decision_tree['left'] = createTree(DY, sigma, pai)
    Decision_tree['right'] = createTree(DN, sigma, pai)
    #print Decision_tree
    return Decision_tree


data=pd.read_csv('iris.csv')
data=np.array(data)
data=data[:,[0,1,4]]
data[data[:,2]=='Iris-setosa',2]='c1'
data[data[:,2]=='Iris-versicolor',2]='c2'
data[data[:,2]=='Iris-virginica',2]='c2'
#print data
data=data.tolist()
myTree = createTree(data,2,0.95)
print(myTree)
