import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Reading the dataset 
dataset = pd.read_csv("Mall_Customers.csv")

#the mistake was I did not convert it into a numpy array
X = dataset.iloc[:,1:].values

#Data Preprocessing
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
X[:,0] = lb.fit_transform(X[:,0])

#Checking Optimal no. of Clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("Elbow Method")
plt.xlabel("No. of clusters")
plt.ylabel("WCSS")
plt.show()


#Creating The Corect No. Of Clusters
kmeans = KMeans(n_clusters=6, init='k-means++')
y_means = kmeans.fit_predict(X)



#Hierarchial Clustering
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title("Dendogram")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()

#Optimal No. Of Clusters
from sklearn.cluster import AgglomerativeClustering
hc =  AgglomerativeClustering(n_clusters=6,affinity="euclidean",linkage="ward")
y_hc = hc.fit_predict(X)


res=0
for i in range(len(y_hc)):
    if y_hc[i]==y_means[i]:
        res+=1
        
print(res)