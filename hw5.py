
# coding: utf-8

# In[427]:

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
import copy
import os
import sys
import math


def read_dataset(filename):
    dataset = []
    classlabels = []
    with open(filename) as f:
        for line in f:
            eachline = [float(i) for i in line.strip().split(',')]
            index = eachline[0]
            classlabel = int(eachline[1])
            x = eachline[2]
            y = eachline[3]
            dataset.append([x,y])
            classlabels.append(classlabel)
    return np.asarray(dataset),np.asarray(classlabels)

def initialize_centroids(points, k):
    centroids = list(points).copy()
    np.random.shuffle(centroids)
    return np.asarray(centroids[:k])

def closest_centroid(points, centroids):
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def move_centroids(points,k,old_centroids): ##points is dataset[0]
    index = closest_centroid(points, old_centroids)
    allcluster = []
    for i in range(k):
        eachcluster = points[index == i]
        allcluster.append(eachcluster)
    new_centroids = []
    for eachcluster in allcluster:
        new_centroid = eachcluster.sum(axis=0) / len(eachcluster)
        new_centroids.append(new_centroid)
    return np.asarray(new_centroids)


def KMEANS(points, numIterations,k):
    numIteration = 0
    old_centroids = initialize_centroids(points, k)
    while numIteration < numIterations:
        new_centroids = move_centroids(points, k,old_centroids)
        old_centroids = new_centroids
        numIteration += 1
    
    return new_centroids

def WithinDistance(points,centroid):
    distance=np.zeros((len(points),len(centroid)))
    for i in range(len(centroid)):
        distance[:,i] = ((points-centroid[i])**2).sum(axis=1)
    WithinD = sum(np.amin(distance, axis=1))
    return WithinD

def getcluster(points,labels,k):##data is ''digits-embedding.csv'' ,label=dataset[1]
    index = closest_centroid(points,  KMEANS(points, 50,k))
    allcluster = []
    allcluster_label = []
    for i in range(k):
        eachcluster = points[index == i]
        cluster_label = labels[index == i]
        allcluster.append(eachcluster)
        allcluster_label.append(cluster_label)
    return allcluster,allcluster_label

def SilhouetteCoefficient(points,labels,k) :
    allcluster = getcluster(points,labels,k)[0] ##allcluster 是list！！
    SC_allcluster=[]
    for i in range(k):
        Allcluster = copy.deepcopy(allcluster)
        del Allcluster[i]
        leftcluster = Allcluster
        SC=[]
        for eachexample in allcluster[i]:
            Cohesion = sum(np.sqrt(((allcluster[i]-eachexample)**2).sum(axis = 1))) / len(allcluster[i])
            seperations = []
            for eachcluster in leftcluster:
                seperation = sum(np.sqrt(((eachcluster-eachexample)**2).sum(axis = 1))) / len(eachcluster)
                seperations.append(seperation)
            Seperation = min(seperations)
            sc = (Seperation - Cohesion)/max(Seperation, Cohesion)
            SC.append(sc)
            sc_avg = sum(SC)/len(SC)
        SC_allcluster.append(sc_avg)
    SC_value = sum(SC_allcluster)/ k
    return SC_value

def NMI(points,labels,k):
    allcluster_labels = getcluster(points,labels,k)[1]
    N =len(points)
    mi_all = []
    H_cluster_all = []
    for eachcluster_labels in allcluster_labels:
        wk = len(eachcluster_labels)
        H_eachcluster = -(wk/N)*math.log(wk/N)
        mi_eachcluster=[]
        for j in range(10):
            wkcj = np.count_nonzero(eachcluster_labels==j)
            cj = np.count_nonzero(labels==j)
            if wkcj ==0:
                mi = 0
            else:
                mi=(wkcj/N)*math.log(N*wkcj/(wk*cj))
            mi_eachcluster.append(mi)
        mi_sum = sum(mi_eachcluster)
        mi_all.append(mi_sum)
        H_cluster_all.append(H_eachcluster)
    H_clusters = sum(H_cluster_all)
    MI = sum(mi_all)
    H_label_all =[]
    for i in range(10):
        if np.count_nonzero(labels==i) ==0:
            H_eachlabel = 0
        else:
            H_eachlabel = -np.count_nonzero(labels==i)/N*math.log(np.count_nonzero(labels==i)/N)
            H_label_all.append(H_eachlabel)
    H_labels = sum(H_label_all)
    NMI = MI/(H_clusters+H_labels)
    return NMI

def GetPlot(points,labels,k):
    allset = getcluster(points,labels,k)
    X1 = []
    X2 =[]
    LABEL= []
    for each in allset[0]:
        x1=each[:,0].tolist()
        X1+=x1
        x2=each[:,1].tolist()
        X2+=x2
    for eachlabel in allset[1]:
        label=eachlabel.tolist()
        LABEL+=label
    plt.scatter(np.asarray(X1),np.asarray(X2),c=np.asarray(LABEL),s=1)
    plt.show()
    return X1,X2,LABEL

def read_dataset2(filename):
    classlabels = []
    features = []
    allset  = []
    with open(filename) as f:
        for line in f:
            eachline = [float(i) for i in line.strip().split(',')]
            index = eachline[0]
            classlabel = int(eachline[1])
            feature = [int(i) for i in eachline[2:786]]
            features.append(feature)
            classlabels.append(classlabel)
    return np.asarray(features),np.asarray(classlabels)

def main():
    if len(sys.argv) == 3:
        data_file = sys.argv[1]
        k = int(sys.argv[2])
        numIterations = 50
        dataset = read_dataset(data_file)
        centroid = KMEANS(dataset[0], numIterations, k)
        WC_SSD = WithinDistance(dataset[0], centroid)
        print('WC-SSD ' + str(WC_SSD))
        SC = SilhouetteCoefficient(dataset[0], dataset[1], k)
        print('SC ' + str(SC))
        nmi = NMI(dataset[0], dataset[1], k)
        print('NMI ' + str(nmi))
    else:
        print('usage: python hw5.py dataFile K')
        print('exiting...')

main()
############################Code for PCA#############################################################################3
def PCA(points,n):
    mean_eachcolumn=np.mean(points,axis=0)
    new = points - mean_eachcolumn
    covMatrix = np.cov(new,rowvar=0)
    eigVals,eigVects = np.linalg.eig(np.mat(covMatrix))
    eigValIndice = np.argsort(eigVals)
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]
    n_eigVect=eigVects[:,n_eigValIndice]
    lowDdata=new*n_eigVect.real
    reconMat=lowDdata*n_eigVect.real.T+mean_eachcolumn
    return lowDdata,reconMat
############################Code for Part A###########################################################################
def read_datasetA(filename):
    classlabels = []
    features = []
    allset = []
    with open(filename) as f:
        for line in f:
            eachline = [float(i) for i in line.strip().split(',')]
            index = eachline[0]
            classlabel = int(eachline[1])
            feature = [int(i) for i in eachline[2:786]]
            allset.append((classlabel, feature))
    return allset

def mainA():
    plots = []
    dataset = read_datasetA1('digits-raw.csv')
    random.shuffle(dataset)
    allclass = list(range(10))
    for eachexample in dataset:
        for i in range(10):
            if eachexample[0] == i:
                image = np.asarray(eachexample[1])
                arr = np.reshape(image, (28, 28))
                plt.imshow(arr, cmap='gray')
                plots.append(plt)
                if i in allclass:
                    allclass = list(set(allclass) - set([i]))
                    plt.show()
                    if allclass == []:
                        break
                    else:
                        continue
                else:
                    continue

##########################Code for Hierarchy clustering############################
def read_datasetHC(filename):
    classlabels = []
    features = []
    allset  = []
    with open(filename) as f:
        for line in f:
            eachline = [float(i) for i in line.strip().split(',')]
            index = eachline[0]
            classlabel = int(eachline[1])
            feature = [int(i) for i in eachline[2:786]]
            #print(feature)
            features.append(feature)
            classlabels.append(classlabel)
    return np.asarray(features),np.asarray(classlabels)

def selectdata(points, labels, num):
    subsample_start = np.zeros((1, 784,), dtype=np.int)
    subsample = subsample_start
    for i in range(10):
        eachdataset = points[labels == i]
        np.random.shuffle(eachdataset)
        each = eachdataset[:num]
        subsample = np.vstack((subsample, each))
    return subsample[1:]

def getDenGraph():
    Z_single = linkage(selectdata(dataset[0], dataset[1],10), method='single')
    fcluster(Z_single, 10, criterion='maxclust')
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram Using Single Linkage')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(Z_single)
    plt.show()
    Z_complete = linkage(selectdata(dataset[0], dataset[1],10), method='complete')
    fcluster(Z_complete, 10, criterion='maxclust')
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram Using Complete Linkage')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(Z_complete)
    plt.show()
    Z_average = linkage(selectdata(dataset[0], dataset[1],10), method='average')
    fcluster(Z_average, 10, criterion='maxclust')
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram Using average Linkage')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(Z_average)
    plt.show()