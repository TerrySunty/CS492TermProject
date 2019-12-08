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

################################################################### import
def load_housing_data():
    return pd.read_csv('train.csv')

def replace_number_data(data, judgement): #用众数来替换NA位仅对数值 axis=0按列处理， 1 按行
    imputer =Imputer(missing_values=judgement, strategy="most_frequent",axis=0 )
    return imputer.fit_transform(data)

def replace_other_data(data):#目前被鸽子掉了
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

def loop_check(nameList):
    list=[]
    for i in nameList:
        encoder = LabelEncoder()
        housing_col = housing["%s"%(i)]
        housing_col_encoded = encoder.fit_transform(housing_col.astype(str))
        if('nan'in encoder.classes_):
            list.append(i)
    return list

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
check_list=loop_check(housing.columns)#check NA data
#print (check_list)

wrong_number=('LotFrontage', 'MasVnrArea', 'GarageYrBlt')
wrong_data=('MasVnrType', 'Electrical')

for number in wrong_number:  #对数字数据进行填补
    housing["%s"%number]=replace_number_data(housing[["%s"%number]],"NaN")

for data in wrong_data:      ##对非数字数据进行填补
    most_data_list = Counter(housing["%s"%data]).most_common(2)
    if (most_data_list[0][0]=='nan'):
        most_data = most_data_list[1][0]
    else: most_data = most_data_list[0][0]
    housing[data] = housing[data].fillna(most_data)

check_list=loop_check(housing.columns)
#print (check_list)