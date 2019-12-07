# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import Imputer #为了填充
#import xlrd # excel

################################################################### import
def load_housing_data():
    return pd.read_csv('train.csv')

def replace_number_data(data, judgement): #用众数来替换NA位仅对数值 axis=0按列处理， 1 按行
    imputer =Imputer(missing_values=judgement, strategy="most_frequent",axis=0 )
    return imputer.fit_transform(data)

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

def loop_check(nameList):
    for i in nameList:
        encoder = LabelEncoder()
        housing_col = housing["%s"%(i)]
        housing_col_encoded = encoder.fit_transform(housing_col.astype(str))
        if('nan'in encoder.classes_):
            print (i)

################################################################## function
housing = load_housing_data() #获得数据
#print (housing.iloc[0:10,0:1]) #get data in [excel]row and col
#housing=housing.drop('Id',axis=1) #delete feature
#housing=housing.drop([1,3]) drop rowfrom 1 to 3
correct_nan=('Alley','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','FireplaceQu','GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','PoolQC','Fence','MiscFeature')

encoder = LabelEncoder()
housing_col = housing["FireplaceQu"]
housing_col_encoded = encoder.fit_transform(housing_col.astype(str))
encoder.classes_
nameList=housing.iloc[:0,0:]
#print (nameList)

housing_num=housing.iloc[0:10,0:]
housing_num["LotFrontage"]=replace_number_data(housing_num[["LotFrontage"]],"NaN")
#print (housing_num["LotFrontage"]) # ? numpy array


for col in correct_nan:
    housing[col] = housing[col].fillna('None')

loop_check(housing)
    
housing_cat = housing["Alley"]
#print (housing_cat)
housing_cat_1hot=replace_other_data(housing_cat)
#print (housing_cat_1hot.toarray())


#encoder = LabelBinarizer()
#housing_cat_1hot=encoder.fit_transform(housing_cat.astype(str))
#print (housing_cat_1hot)
#housing_cat_1hot=replace_number_data(housing_cat_1hot)
#print (housing_cat_1hot.toarray())






