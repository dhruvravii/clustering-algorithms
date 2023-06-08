from numpy import unique
from numpy import where
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
import pandas as pd

from google.colab import files
 
 
uploaded = files.upload()
df=pd.read_csv('Coldplay.csv')

newdf1 = df.loc[(df.album_name == "Music Of The Spheres")|
                (df.album_name == "Everyday Life")|
                (df.album_name == "A Head Full Of Dreams")|
                (df.album_name == "Ghost Stories")|
                (df.album_name == "Mylo Xyloto")|
                (df.album_name == "X&Y")|
                (df.album_name == "Viva La Vida or Death and All His Friends")|
                (df.album_name == "Viva La Vida (Prospekt's March Edition)")|
                (df.album_name == "A Rush of Blood to the Head")|
                (df.album_name == "Parachutes")]
newdf2 = newdf1.drop(["release_date","loudness", "valence", "name", "explicit","album_name", "time_signature"], axis=1)
print(newdf1)
print(newdf2)
x_train = newdf2.to_numpy()
print(x_train)
gm_model = GaussianMixture(n_components=10).fit(x_train)
# print(gm_model.means_)
mks = ["o", "v", "^", "1", "s","P","*", "|", "D", "$f$"]
colors = ['r','g','b','c','m', 'y', 'k','#ff5733','#33fff9', '#111212']

m= ["null"]*119
n = newdf1.to_numpy()
for i in range(len(x_train)):
  # print(i)
  # print(n[i][2])
  if(n[i][3] == "Music Of The Spheres"):
       m[i]= mks[0]
  if(n[i][3] == "Everyday Life"):
       m[i]= mks[1]
  if(n[i][3] == "A Head Full Of Dreams"):
       m[i]= mks[2]
  if(n[i][3] == "Ghost Stories"):
       m[i]= mks[3]
  if(n[i][3] == "Mylo Xyloto"):
       m[i]= mks[4]
  if(n[i][3] == "X&Y"):
       m[i]= mks[5]
  if(n[i][3] == "Viva La Vida or Death and All His Friends"):
       m[i]= mks[6]
  if(n[i][3] == "Viva La Vida (Prospekt's March Edition)"):
       m[i]= mks[7]
  if(n[i][3] == "A Rush of Blood to the Head"):
       m[i]= mks[8]
  if(n[i][3] == "Parachutes"):
       m[i]= mks[9]

print(m)

labels = gm_model.predict(x_train)
print(labels)

for i in range(len(x_train)):
#   print(labels[i])
   plt.scatter(x_train[i][6], i, marker = m[i], c=colors[labels[i]])


plt.show()
