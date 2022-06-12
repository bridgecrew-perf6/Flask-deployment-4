# -*- coding: utf-8 -*-
# @Author : LuoXianan
# @File : model.py.py
# @Project: ML Deployment
# @CreateTime : 2022/5/12 22:25:49
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle

# Load the csv file
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('data.csv')

# 查看前五行数据
print(df.head())


# Select independent and dependent variable
X = df[["low_risk","medium_risk","high_risk","break_faith"]]
y = df["class"]

# Split the detaset into train and test  切分数据
# 0.3 意味着30个数据用于测试，还有一个随机状态，可以给任何数字
X_train,X_test,y_train,y_test = train_test_split(X , y, test_size=0.3)

# Feature scaling  进行特征缩放
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Instantiate the model
'''
 'LR' : LogisticRegression(max_iter=10000),  # 线性回归（LR）规定收敛次数为10000
        'LDA' : LinearDiscriminantAnalysis(),       # 线性判别分析（LDA）
        'KNN' : KNeighborsClassifier(),             # K近邻（KNN）
        'CART' : DecisionTreeClassifier(),          # 分类与回归树（CART）
        'NB' : GaussianNB(),                        # 贝叶斯分类器（NB）
        'SVM' : SVC(),                              # 支持向量机（SVM）
        'RFC': RandomForestClassifier()             # 随机森林分类(RFC)
'''
classifier = KNeighborsClassifier()               # K近邻（KNN）
# classifier = RandomForestClassifier()             # 随机森林分类(RFC)
# classifier = SVC()                                # 支持向量机（SVM）
# classifier = GaussianNB()                         # 贝叶斯分类器（NB）
# classifier = DecisionTreeClassifier()             # 分类与回归树（CART）
# classifier = LinearDiscriminantAnalysis()         # 线性判别分析（LDA）
# classifier = LogisticRegression(max_iter=10000)  # 线性回归（LR）规定收敛次数为10000

#  Fit the Model   拟合模型
classifier.fit(X_train,y_train)
# 深度学习模型建立完成
# 下一步是制作模型pickle对象,导入pickle需要的库
# Make pickle file of our model
pickle.dump(classifier,open("model.pkl","wb"))  # 将对象传递给这个分类器
# 使用方法dump,转储了使用模型分类器来模拟模型，然后给模型命名，扩展名为model.pkl

# 生成model.pkl文件，成功将模型转化为pickl文件

# 然后处理app.py文件
