# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 14:15:29 2021

@author: Garry
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
import xgboost as xgb 
import lightgbm as lgb
from sklearn.metrics import roc_curve, auc 
import matplotlib.pyplot as plt

#读取文件
def read_data():
    df = pd.read_csv('./train_pro.csv',index_col=0)
    df.head()
    df.drop(['id'],axis=1)
    labels = df['label']
    df=df.drop(['label'],axis=1)
    # 数据集划分
    X_train,X_test,y_train,y_test=train_test_split(df,labels, random_state=2021, test_size=0.3)
    # 打印划分后的数据集大小
    print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
    
    return X_train,X_test,y_train,y_test


#使用XGBoost构建模型
def Xgb(X_train,X_test,y_train,y_test):
    
    dtrain=xgb.DMatrix(X_train,y_train)
    dtest=xgb.DMatrix(X_test)

    #参数
    params={ 'eta':0.1,
    'objective':'multi:softmax',
    'n_jobs':-1, 
    'learning_rate':0.01, 
    'num_class':4,
    'random_state':2021
     }
    
    #模型训练
    #watchlist = [(dtrain,'train')]
    Xgb=xgb.train(params,dtrain,num_boost_round=200)
    
    #模型预测
    #输出概率
    ypred=Xgb.predict(dtest)
    # print(ypred)
    
    print("XGBoost预测结果：")

    # 设置阈值, 输出一些评价指标，选择概率大于0.5的为1，其他为0类
    y_pred = (ypred >= 0.5)*1
    # print(y_pred)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test,y_pred))
    # print(type(y_pred))
    # print(y_pred)
    
    
    #XGBoost决策树的生长过程可视化
    image=xgb.to_graphviz(Xgb, num_trees=1)
    image.view()  #运行完后文件名为Source.gv.pdf便是此图
    return


#使用LightGBM 构建模型
def lg(X_train,X_test,y_train,y_test):
    train_data = lgb.Dataset(X_train, label=y_train)
    validation_data = lgb.Dataset(X_test, label=y_test)

    # 参数
    params = {'boosting_type':'goss',     
              'boosting_type': 'goss',
              'learning_rate': 0.01,
              'lambda_l1': 0.1,
              'lambda_l2': 0.2,
              'objective': 'multiclass', 
              'num_class': 4,
    }
    # 模型训练
    gbm = lgb.train(params, train_data, valid_sets=[validation_data],
                    num_boost_round=1000,
                    verbose_eval=100, 
                    early_stopping_rounds=200)

    # 模型预测
    y_pred = gbm.predict(X_test)  #输出的是一个概率值  
    y_pred = [list(x).index(max(x)) for x in y_pred]
    # y_pred = (y_pred >= 0.5)*1  #将概率转化为分类标签
    #print(y_pred)
    # 模型评估
    print("LightGBM预测结果：")

    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test,y_pred))
    
    #Lgbm决策树的生长过程可视化
    image=lgb.create_tree_digraph(gbm, tree_index=1)   
    image.view()   #运行完后文件名为Digraph.gv.pdf便是此图

    return 



if __name__ == "__main__":
    X_train,X_test,y_train,y_test=read_data()
    lg(X_train,X_test,y_train,y_test)
    Xgb(X_train,X_test,y_train,y_test)
