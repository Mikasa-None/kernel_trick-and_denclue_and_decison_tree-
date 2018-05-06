#-*-coding:utf-8-*-
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
import networkx as nx
import pandas as pd

#爬坡
def _hill_climb(x_t, X, W=None, h=0.1, eps=1e-7):
    error = 99.
    prob = 0.
    x_l1 = np.copy(x_t)   #X(t+1)
    radius_new = 0.
    radius_old = 0.
    radius_twiceold = 0.
    iters = 0.
    while True:
        radius_thriceold = radius_twiceold
        radius_twiceold = radius_old
        radius_old = radius_new
        x_l0 = np.copy(x_l1)       #X(t)
        x_l1, density = _step(x_l0, X, W=W, h=h)
        error = density - prob
        prob = density
        radius_new = np.linalg.norm(x_l1 - x_l0)
        radius = radius_thriceold + radius_twiceold + radius_old + radius_new
        iters += 1
        if iters > 3 and error < eps:
            break
    return [x_l1, prob, radius]

#计算X(t+1)
def _step(x_l0, X, W=None, h=0.1):
    n = X.shape[0]
    d = X.shape[1]
    superweight = 0.  # superweight is the kernel X weight for each item
    x_l1 = np.zeros((1, d))
    if W is None:
        W = np.ones((n, 1))
    else:
        W = W
    for j in range(n):
        kernel = kernelize(x_l0, X[j], h, d)
        kernel = kernel * W[j] / (h ** d)
        superweight = superweight + kernel
        x_l1 = x_l1 + (kernel * X[j])
    x_l1 = x_l1 / superweight
    density = superweight / np.sum(W)
    return [x_l1, density]

#计算高斯核
def kernelize(x, y, h, degree):
    kernel = np.exp(-(np.linalg.norm(x - y) / h) ** 2. / 2.) / ((2. * np.pi) ** (degree / 2))
    return kernel
#bad code
'''def density(X,D,h,degree):
    sum1=0
    for i in range(D.shape[0]):
        k=kernelize(X,D[i],h,degree)
        sum1=sum1+k
    d=1./D.shape[0]/h**degree*sum1
    return d
def DENCLUE(D,h,xi,eps):
    rows=D.shape[0]
   A=np.zeros((rows,D.shape[1]))
    labels=-np.ones(rows)
    dens = np.zeros((rows, 1))
    degree = D.shape[1]
    for i in range(rows):
        x_mark=findattractor(D[i],D,h,eps)
        dens[i]=density(x_mark,D,h,degree)
        if dens[i]>=xi:
            A[i]=x_mark
    print A,dens
'''
class DENCLUE(BaseEstimator, ClusterMixin):
    def __init__(self, h=None, eps=1e-8, min_density=0., metric='euclidean'):
        self.h = h
        self.eps = eps
        self.min_density = min_density
        self.metric = metric

    def classify(self, X, y=None, sample_weight=None):
        if not self.eps > 0.0:
            raise ValueError("eps must be positive.")
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        density_attractors = np.zeros((self.n_samples, self.n_features))
        radii = np.zeros((self.n_samples, 1))
        density = np.zeros((self.n_samples, 1))

        # 构造初始值
        if self.h is None:
            self.h = np.std(X) / 5
        if sample_weight is None:
            sample_weight = np.ones((self.n_samples, 1))
        else:
            sample_weight = sample_weight
        # 初始化所有的点为noise点
        labels = -np.ones(X.shape[0])
        # 对每个样本点进行attractor和其相应密度的计算
        for i in range(self.n_samples):
            density_attractors[i], density[i], radii[i] = _hill_climb(X[i], X, W=sample_weight,
                                                                      h=self.h, eps=self.eps)
        # 构造链接图
        cluster_info = {}
        num_clusters = 0
        cluster_info[num_clusters] = {'instances': [0],
                                      'centroid': np.atleast_2d(density_attractors[0])}
        g_clusters = nx.Graph()
        for j1 in range(self.n_samples):
            g_clusters.add_node(j1, attr_dict={'attractor': density_attractors[j1], 'radius': radii[j1],
                                               'density': density[j1]})
        # 构造聚类图
        for j1 in range(self.n_samples):
            for j2 in (x for x in range(self.n_samples) if x != j1):
                if g_clusters.has_edge(j1, j2):
                    continue
                diff = np.linalg.norm(g_clusters.node[j1]['attractor'] - g_clusters.node[j2]['attractor'])
                if diff <= (g_clusters.node[j1]['radius'] + g_clusters.node[j1]['radius']):
                    g_clusters.add_edge(j1, j2)
        clusters = list(nx.connected_component_subgraphs(g_clusters))
        num_clusters = 0
        # 链接聚类
        for clust in clusters:
            # 得到attractors中的最大密度以及相应的点位信息
            max_instance = max(clust, key=lambda x: clust.node[x]['density'])
            max_density = clust.node[max_instance]['density']
            max_centroid = clust.node[max_instance]['attractor']
            complete = False
            c_size = len(clust.nodes())
            if clust.number_of_edges() == (c_size * (c_size - 1)) / 2.:
                complete = True
            # 构造聚类字典
            cluster_info[num_clusters] = {'instances': clust.nodes(),
                                          'size': c_size,
                                          'centroid': max_centroid,
                                          'density': max_density,
                                          'complete': complete}
            # 如果类的密度小于要求，则即为noise点
            if max_density >= self.min_density:
                labels[clust.nodes()] = num_clusters
            num_clusters += 1
        self.clust_info_ = cluster_info
        self.labels_ = labels
        return self


data = pd.read_csv('iris.csv')
data = np.array(data)
samples = np.mat(data[:,0:2])
true_labels=data[:,-1]
labels=list(set(true_labels))
true_ID=np.zeros((3,50))
index=range(len(true_labels))
for i in range(len(labels)):
    true_ID[i]=[j for j in index if true_labels[j]==labels[i]]
d = DENCLUE(0.25, 0.0001)
d.classify(samples)
right_num=0

for i in range(len(d.clust_info_)):
    bestlens=0
    clust_set = set(d.clust_info_[i]['instances'])
    for j in range(len(labels)):
        true_set=set(true_ID[j])
        and_set= clust_set&true_set
        if len(list(and_set))>bestlens:
            bestlens=len(list(and_set))
    right_num+=bestlens
#输出类的信息以及聚类的纯度
print d.clust_info_,float(right_num)/len(samples)
