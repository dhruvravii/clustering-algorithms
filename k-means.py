from numpy import unique
from numpy import where
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
import pandas as pd

from google.colab import files
 
#for google colab 
uploaded = files.upload()
df=pd.read_csv('Salary Data.csv')
x = df["YearsExperience"].to_numpy()
y = df["Salary"].to_numpy()

print(x)
print(y)
x_ = x.reshape(-1, 1)

print(x_)
kmeans_model = KMeans(n_clusters=3, n_init = "auto").fit(x_)
print(kmeans_model.labels_)

x_class0 = [x[i] for i in range(len(x)) if kmeans_model.labels_[i] == 0]
x_class1 = [x[i] for i in range(len(x)) if kmeans_model.labels_[i] == 1]
x_class2 = [x[i] for i in range(len(x)) if kmeans_model.labels_[i] == 2]
y_class0 = [y[i] for i in range(len(x)) if kmeans_model.labels_[i] == 0]
y_class1 = [y[i] for i in range(len(x)) if kmeans_model.labels_[i] == 1]
y_class2 = [y[i] for i in range(len(x)) if kmeans_model.labels_[i] == 2]
plt.scatter(x_class0, y_class0, color='red', label='Class 0', s=5)
plt.scatter(x_class1, y_class1, color='blue', label='Class 1', s=5)
plt.scatter(x_class2, y_class2, color='green', label='Class 1', s=5)

plt.show()

