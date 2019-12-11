# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import Imputer #为了填充
from collections import Counter
from sklearn import metrics
# ensemble model
from sklearn.ensemble import RandomForestRegressor


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import StandardScaler

################################################################### import
def load_housing_data():
    return pd.read_csv('train.csv')
def load_test_data():
    return pd.read_csv('test.csv')

def replace_number_data(data, judgement): #用众数来替换NA位仅对数值 axis=0按列处理， 1 按行
    imputer =Imputer(missing_values=judgement, strategy="most_frequent",axis=0 )
    return imputer.fit_transform(data)

'''#目前被鸽子项目
def replace_other_data(data):
    encoder = LabelEncoder()
    housing_cat_encoded = encoder.fit_transform(data.astype(str))
    find_nan=0
    for i in encoder.classes_:
        if (i=='nan'):
            housing_cat_encoded=replace_number_data(housing_cat_encoded.reshape(-1,1),find_nan)
            break
        else: find_nan+=1
        
    encoder = OneHotEncoder()
    housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))#返回的为稀疏矩阵
    return housing_cat_1hot
'''

def loop_check(title_list, data_ist):
    list=[]
    for i in title_list:
        encoder = LabelEncoder()
        housing_col = data_ist["%s"%(i)]
        encoder.fit_transform(housing_col.astype(str))
        if('nan'in encoder.classes_):
            list.append(i)
    return list

def check_type(check_list, data_list):
    wrong_number=[]
    wrong_data=[]
    for col in check_list:
        temporary = data_list[col]
        for i in range(len(temporary)):
            if (temporary[i]!='nan'):
                if (type(temporary[i]) is str):
                    wrong_data.append(col)
                else:wrong_number.append(col)
                break
    return wrong_number, wrong_data

def process_data(data_set):
    train_dummy = pd.get_dummies(data_set)
    numeric_cols = data_set.columns[data_set.dtypes != 'object'] #find numerical
    correct_number_means = train_dummy.loc[:, numeric_cols].mean()
    correct_number_std = train_dummy.loc[:, numeric_cols].std()
    train_dummy.loc[:, numeric_cols] = (train_dummy.loc[:, numeric_cols] - correct_number_means) / correct_number_std
    return train_dummy

################################################################## function
housing = load_housing_data() #获得训练集数据
#train_data=housing
correct_nan=('Alley','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
             'BsmtFinType2','FireplaceQu','GarageType', 'GarageFinish', 
             'GarageQual', 'GarageCond','PoolQC','Fence','MiscFeature')
#从描述中找出的NA合法项
check_list=[] #方便查看缺失项目

for col in correct_nan:
    housing[col] = housing[col].fillna('None')
check_list=loop_check(housing.columns, housing)#check NA data
#print (check_list)

#print (housing.describe())

total = housing.isnull().sum().sort_values(ascending=False)
#print(total) #显示缺失项
#print (housing)

#偏移值处理
#max 755000
ex_housing = housing.sort_values(by = 'GrLivArea', ascending = False)[:2] ### 沿着轴方向按指定值排序  
train= housing.drop(housing[housing['Id'] == 1299].index)
train= train.drop(train[train['Id'] == 524].index)
var = 'GrLivArea'
#data = pd.concat([train['SalePrice'], train[var]], axis=1)
#data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

#print (ex_housing)




#计算所有特征值每两个之间的相关系数，并作图表示。
corrmat = train.corr()#得到相关系数
#f,ax = plt.subplots(figsize = (12,9))
#sns.heatmap(corrmat, vmax = .8, square = True)#热点图

#取出相关性最大的前十个，做出热点图表示
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
#print (cols)
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',     annot_kws={'size': 10}, yticklabels=cols.values,    xticklabels=cols.values)
#plt.show()






wrong_number, wrong_data = check_type(check_list, housing)  #讲存在错误项区分类型
#print (wrong_number)
#print (wrong_data)
#wrong_number=('LotFrontage', 'MasVnrArea', 'GarageYrBlt')
#wrong_data=('MasVnrType', 'Electrical')

for number in wrong_number:  #对数字数据进行填补
    housing["%s"%number]=replace_number_data(housing[["%s"%number]],"NaN")

for data in wrong_data:      ##对非数字数据进行填补
    most_data_list = Counter(housing["%s"%data]).most_common(2)
    if (most_data_list[0][0]=='nan'):
        most_data = most_data_list[1][0]
    else: most_data = most_data_list[0][0]
    housing[data] = housing[data].fillna(most_data)

check_list=loop_check(housing.columns, housing)
#print (check_list)
################################################################## training
test=load_test_data()

for col in correct_nan:
    test[col] = test[col].fillna('None')
check_list=loop_check(test.columns, test)
#print (check_list)


total = test.isnull().sum().sort_values(ascending=False)
#print(total)








wrong_number, wrong_data = check_type(check_list, test)

for number in wrong_number:  #对数字数据进行填补
    test["%s"%number]=replace_number_data(test[["%s"%number]],"NaN")

for data in wrong_data:      ##对非数字数据进行填补，忘了丫的用本身数据进行填补了
    most_data_list = Counter(test["%s"%data]).most_common(2)
    if (most_data_list[0][0]=='nan'):
        most_data = most_data_list[1][0]
    else: most_data = most_data_list[0][0]
    test[data] = test[data].fillna(most_data)
    
check_list=loop_check(test.columns, test)
#print (check_list)
##################################################################  testing

housing['MSSubClass'] =housing['MSSubClass'].astype(str) #?????????????special

train_data = housing.drop('Id',axis=1)
train_data = train_data.drop('SalePrice',axis=1)       #数据处理d训练整合
train_result = housing["SalePrice"]
#train_data = train_data.values      数据类型转化成sklearn可用的numpy array
#print('存在' if any(housing.duplicated()) else '不存在', '重复观测值')

#数值数据标准化




#train_data= process_data(train_data)   ###使数据处理更平滑
#print(train_data)



train_dummy = pd.get_dummies(train_data)
numeric_cols = train_data.columns[train_data.dtypes != 'object']
scaler = StandardScaler()
scaler.fit(train_dummy.loc[:, numeric_cols])                               # 使用transfrom必须要用fit语句
train_dummy.loc[:, numeric_cols] = scaler.transform(train_dummy.loc[:, numeric_cols])          # transfrom通过找中心和缩放等实现标准化
train_dummy.loc[:, numeric_cols] = scaler.fit_transform(train_dummy.loc[:, numeric_cols]) 
print (train_dummy)


#train_data = train_data["MasVnrType"].values
#train_data_dum=pd.get_dummies(train_data)

#print (train_data_dum)



#################################################################  Data processing
#random_forest = RandomForestRegressor()
#random_forest.fit(train_data,train_result)

#test_data = test.drop('Id',axis=1)
#test_result = random_forest.predict(test_data)
#print(test_result)





#print(metrics.mean_squared_log_error(predicted_prices, train_y))    #使用(RMSE)均方对数误差是做评价指标
        
################################################################   submission
        
'''
Submission_File = pd.DataFrame({'Id': test['Id'], 'SalePrice': test_result})
# you could use any filename. We choose submission here
Submission_File.to_csv('teamname_submission.csv', index=False)

'''