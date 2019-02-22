import xgboost as xgb
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn import metrics
from  sklearn.datasets  import  make_hastie_10_2
from  sklearn.ensemble  import  GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
warnings.filterwarnings("ignore")


#获取处理好的训练数据
def getTrain():
    # 读取数据,首先对训练数据进行处理
    data = pd.read_csv("../data/trains.csv", encoding="gbk")
    # 筛选掉训练数据的account列
    data.drop(['account'], axis=1, inplace=True)

    # rfm3 先处理美元符号，然后将空的列补填该列的平均值
    data['rfm3'] = data['rfm3'].str.replace('$', '')
    data['rfm3'] = data['rfm3'].str.replace(',', '')
    data['rfm3'] = data['rfm3'].astype(float)
    clean_z = data['rfm3'].fillna(data['rfm3'].mean())
    clean_z[clean_z == ''] = data['rfm3'].mean()
    data['rfm3'] = clean_z

    # 对input1数据编码
    input1_mapping = {'X': 0.6, 'Y': 0.3, 'Z': 0.1}
    input2_mapping = {'A': 0.5, 'B': 0.3, 'C': 0.15, 'D': 0.05, 'E': 0.0}
    demog_ho_mapping = {'是': 1, '否': 0}

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

    # 处理rfm1,rfm2,rfm4的美元符号
    data['rfm1'] = data['rfm1'].str.replace('$', '')
    data['rfm1'] = data['rfm1'].str.replace('(', '')
    data['rfm1'] = data['rfm1'].str.replace(')', '')
    data['rfm1'] = data['rfm1'].str.replace(',', '')
    data['rfm1'] = data['rfm1'].astype(float)

    data['rfm2'] = data['rfm2'].str.replace('$', '')
    data['rfm2'] = data['rfm2'].str.replace(',', '')
    data['rfm2'] = data['rfm2'].astype(float)

    data['rfm4'] = data['rfm4'].str.replace('$', '')
    data['rfm4'] = data['rfm4'].str.replace(',', '')
    data['rfm4'] = data['rfm4'].astype(float)

    data.drop(['demog_ho'], axis=1, inplace=True)
    data.drop(['rfm3'], axis=1, inplace=True)
    train_y = data['b_tgt']
    data.drop(['b_tgt'], axis=1, inplace=True)
    train_x = data

    print(train_y)
    print(train_x)
    return train_x,train_y

#获取处理好的测试数据
def getTest():
    '''
    然后开始处理测试数据
    '''
    # 读取数据,首先对训练数据进行处理
    data = pd.read_csv("../data/tests.csv", encoding="gbk")
    data.drop(data.columns[[22]], axis=1, inplace=True)
    # 筛选掉训练数据的account列
    res = pd.DataFrame()
    res['account'] = data['account']
    data.drop(['account'], axis=1, inplace=True)

    # rfm3 先处理美元符号，然后将空的列补填该列的平均值
    data['rfm3'] = data['rfm3'].str.replace('$', '')
    data['rfm3'] = data['rfm3'].str.replace(',', '')
    data['rfm3'] = data['rfm3'].astype(float)
    clean_z = data['rfm3'].fillna(data['rfm3'].mean())
    clean_z[clean_z == ''] = data['rfm3'].mean()
    data['rfm3'] = clean_z

    # 对input1数据编码
    input1_mapping = {'X': 0.6, 'Y': 0.3, 'Z': 0.1}
    input2_mapping = {'A': 0.5, 'B': 0.3, 'C': 0.15, 'D': 0.05, 'E': 0.0}
    demog_ho_mapping = {'是': 1, '否': 0}

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

    # 处理rfm1,rfm2,rfm4的美元符号
    data['rfm1'] = data['rfm1'].str.replace('$', '')
    data['rfm1'] = data['rfm1'].str.replace('(', '')
    data['rfm1'] = data['rfm1'].str.replace(')', '')
    data['rfm1'] = data['rfm1'].str.replace(',', '')
    data['rfm1'] = data['rfm1'].astype(float)

    data['rfm2'] = data['rfm2'].str.replace('$', '')
    data['rfm2'] = data['rfm2'].str.replace(',', '')
    data['rfm2'] = data['rfm2'].astype(float)

    data['rfm4'] = data['rfm4'].str.replace('$', '')
    data['rfm4'] = data['rfm4'].str.replace(',', '')
    data['rfm4'] = data['rfm4'].astype(float)

    data.drop(['demog_ho'], axis=1, inplace=True)
    data.drop(['rfm3'], axis=1, inplace=True)

    test_x = data
    print(test_x)

    return test_x,res

if __name__ == '__main__':

    '''
    先对训练数据进行处理
    '''
    train_x,train_y = getTrain()
    test_x,res = getTest()


    #开始训练模型
    clf = XGBClassifier(silent=0 ,#设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
                        #nthread=4,# cpu 线程数 默认最大
                        learning_rate= 0.3, # 如同学习率
                        min_child_weight=1,
                        # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
                        #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
                        #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
                        max_depth=6, # 构建树的深度，越大越容易过拟合
                        gamma=0,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
                        subsample=1, # 随机采样训练样本 训练实例的子采样比
                        max_delta_step=0,#最大增量步长，我们允许每个树的权重估计。
                        colsample_bytree=1, # 生成树时进行的列采样
                        reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
                        #reg_alpha=0, # L1 正则项参数
                        #scale_pos_weight=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
                        #objective= 'multi:softmax', #多分类的问题 指定学习任务和相应的学习目标
                        #num_class=10, # 类别数，多分类与 multisoftmax 并用
                        n_estimators=100, #树的个数
                        seed=1000 #随机种子
                        #eval_metric= 'auc'
                      )
    clf.fit(train_x, train_y)
    #测试数据
    y_pre = clf.predict(test_x)
    print(y_pre)
    #获取打分值
    y_pro = clf.predict_proba(test_x)[:, 1]

    res['score'] = y_pro

    res.to_csv("../data/score.csv")



