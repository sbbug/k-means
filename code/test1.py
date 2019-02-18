import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = pd.read_csv("../data/customer.txt",sep=',',header=None,)
print(data)
x = data.as_matrix()

# 传入要分类的数目
kms = KMeans(n_clusters=2)
y = kms.fit_predict(x)
print(y)
print(kms.labels_)
label=kms.labels_

pca = PCA(n_components=2)
new_pca = pd.DataFrame(pca.fit_transform(data))

d = new_pca[y == 0]
plt.plot(d[0], d[1], 'r.')
d = new_pca[y == 1]
plt.plot(d[0], d[1], 'go')

plt.gcf().savefig('kmeans.png')
plt.show()
