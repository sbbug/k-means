'''
先对数据进行预处理
'''
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
import numpy as np
#21列数据
col_names = [
            'cat_input1',
            'cat_input2',
            'demog_age',
            'demog_ho',
            'demog_homeval',
            'demog_inc',
            'demog_pr'
            'rfm1',
            'rfm2',
            'rfm3',
            'rfm4',
            'rfm5',
            'rfm6',
            'rfm7',
            'rfm8',
            'rfm9',
            'rfm10',
            'rfm11',
            'rfm12',
            'demog_gent',
            'demog_denm',
            'account',
            ]
data = pd.read_csv("../data/tests.csv",encoding="gbk")

res = pd.DataFrame()

data.drop(data.columns[[22]],axis=1,inplace=True)
clean_z = data['rfm3'].fillna(0)
clean_z[clean_z==''] = 0
data['rfm3'] = clean_z

input1_mapping = {'X':0.6, 'Y':0.3,'Z':0.1}
input2_mapping = {'A':0.5, 'B':0.3, 'C':0.15, 'D':0.05, 'E':0.0}
demog_ho_mapping = {'是':1, '否':0}


data['cat_input1'] = data['cat_input1'].map(input1_mapping)
data['cat_input2'] = data['cat_input2'].map(input2_mapping)



data['demog_ho'] = data['demog_ho'].map(demog_ho_mapping)
data['demog_age'] = data['demog_age'].where(data['demog_age'].notnull(), 0)

data['demog_inc'] = data['demog_inc'].str.replace('$', '')
data['demog_inc'] = data['demog_inc'].str.replace(',', '')
data['demog_inc'] = data['demog_inc'].astype(float)

data['demog_homeval'] = data['demog_homeval'].str.replace('$', '')
data['demog_homeval'] = data['demog_homeval'].str.replace(',', '')
data['demog_homeval'] = data['demog_homeval'].astype(float)

data['rfm1'] = data['rfm1'].str.replace('$', '')
data['rfm1'] = data['rfm1'].str.replace('(', '')
data['rfm1'] = data['rfm1'].str.replace(')', '')
data['rfm1'] = data['rfm1'].str.replace(',', '')
data['rfm1'] = data['rfm1'].astype(float)


data['rfm2'] = data['rfm2'].str.replace('$', '')
data['rfm2'] = data['rfm2'].str.replace(',', '')
data['rfm2'] = data['rfm2'].astype(float)


data['rfm3'] = data['rfm3'].str.replace('$', '')
data['rfm3'] = data['rfm3'].str.replace(',', '')
data['rfm3'] = data['rfm3'].astype(float)


data['rfm4'] = data['rfm4'].str.replace('$', '')
data['rfm4'] = data['rfm4'].str.replace(',', '')
data['rfm4'] = data['rfm4'].astype(float)

res['account'] = data['account']
data = data.drop(['account'], axis=1)

data = data.drop(['demog_ho'], axis=1)
data = data.drop(['rfm3'], axis=1)
print(np.isnan(data).any())

data = data.as_matrix()

pca = PCA(n_components=6)
new_pca = pd.DataFrame(pca.fit_transform(data))
X = new_pca.as_matrix()
print(X)


#调用kmeans，设置两个类，分别代表潜在客户与普通客户
kms = KMeans(n_clusters=6)
#获取类别标签
Y= kms.fit_predict(X)

res['class'] = Y

res.to_csv("../data/res.csv")

#使用PCA再次进行降维显示结果
pca = PCA(n_components=2)
new_pca = pd.DataFrame(pca.fit_transform(data))

#显示效果
d = new_pca[Y == 0]
plt.plot(d[0], d[1], 'r.')
d = new_pca[Y == 1]
plt.plot(d[0], d[1], 'g.')
d = new_pca[Y == 2]
plt.plot(d[0], d[1], 'b.')
d = new_pca[Y == 3]
plt.plot(d[0], d[1], 'y.')
d = new_pca[Y == 4]
plt.plot(d[0], d[1], 'c.')
d = new_pca[Y == 5]
plt.plot(d[0], d[1], 'k.')
plt.gcf().savefig('kmeans.png')
plt.show()