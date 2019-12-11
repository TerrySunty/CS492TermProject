# - SalePrice: 房产销售价格，以美元计价。所要预测的目标变量
# - MSSubClass: Identifies the type of dwelling involved in the sale 住所类型
# - MSZoning: The general zoning classification 区域分类
# - LotFrontage: Linear feet of street connected to property 房子同街道之间的距离
# - LotArea: Lot size in square feet 建筑面积
# - Street: Type of road access 主路的路面类型
# - Alley: Type of alley access 小道的路面类型
# - LotShape: General shape of property 房屋外形
# - LandContour: Flatness of the property 平整度
# - Utilities: Type of utilities available 配套公用设施类型
# - LotConfig: Lot configuration 配置
# - LandSlope: Slope of property 土地坡度
# - Neighborhood: Physical locations within Ames city limits 房屋在埃姆斯市的位置
# - Condition1: Proximity to main road or railroad 附近交通情况
# - Condition2: Proximity to main road or railroad (if a second is present) 附近交通情况（如果同时满足两种情况）
# - BldgType: Type of dwelling 住宅类型
# - HouseStyle: Style of dwelling 房屋的层数
# - OverallQual: Overall material and finish quality 完工质量和材料
# - OverallCond: Overall condition rating 整体条件等级
# - YearBuilt: Original construction date 建造年份
# - YearRemodAdd: Remodel date 翻修年份
# - RoofStyle: Type of roof 屋顶类型
# - RoofMatl: Roof material 屋顶材料
# - Exterior1st: Exterior covering on house 外立面材料
# - Exterior2nd: Exterior covering on house (if more than one material) 外立面材料2
# - MasVnrType: Masonry veneer type 装饰石材类型
# - MasVnrArea: Masonry veneer area in square feet 装饰石材面积
# - ExterQual: Exterior material quality 外立面材料质量
# - ExterCond: Present condition of the material on the exterior 外立面材料外观情况
# - Foundation: Type of foundation 房屋结构类型
# - BsmtQual: Height of the basement 评估地下室层高情况
# - BsmtCond: General condition of the basement 地下室总体情况
# - BsmtExposure: Walkout or garden level basement walls 地下室出口或者花园层的墙面
# - BsmtFinType1: Quality of basement finished area 地下室区域质量
# - BsmtFinSF1: Type 1 finished square feet Type 1完工面积
# - BsmtFinType2: Quality of second finished area (if present) 二次完工面积质量（如果有）
# - BsmtFinSF2: Type 2 finished square feet Type 2完工面积
# - BsmtUnfSF: Unfinished square feet of basement area 地下室区域未完工面积
# - TotalBsmtSF: Total square feet of basement area 地下室总体面积
# - Heating: Type of heating 采暖类型
# - HeatingQC: Heating quality and condition 采暖质量和条件
# - CentralAir: Central air conditioning 中央空调系统
# - Electrical: Electrical system 电力系统
# - 1stFlrSF: First Floor square feet 第一层面积
# - 2ndFlrSF: Second floor square feet 第二层面积
# - LowQualFinSF: Low quality finished square feet (all floors) 低质量完工面积
# - GrLivArea: Above grade (ground) living area square feet 地面以上部分起居面积
# - BsmtFullBath: Basement full bathrooms 地下室全浴室数量
# - BsmtHalfBath: Basement half bathrooms 地下室半浴室数量
# - FullBath: Full bathrooms above grade 地面以上全浴室数量
# - HalfBath: Half baths above grade 地面以上半浴室数量
# - Bedroom: Number of bedrooms above basement level 地面以上卧室数量
# - KitchenAbvGr: Number of kitchens 厨房数量
# - KitchenQual: Kitchen quality 厨房质量
# - TotRmsAbvGrd: Total rooms above grade (does not include bathrooms) 总房间数（不含浴室和地下部分）
# - Functional: Home functionality rating 功能性评级
# - Fireplaces: Number of fireplaces 壁炉数量
# - FireplaceQu: Fireplace quality 壁炉质量
# - GarageType: Garage location 车库位置
# - GarageYrBlt: Year garage was built 车库建造时间
# - GarageFinish: Interior finish of the garage 车库内饰
# - GarageCars: Size of garage in car capacity 车壳大小以停车数量表示
# - GarageArea: Size of garage in square feet 车库面积
# - GarageQual: Garage quality 车库质量
# - GarageCond: Garage condition 车库条件
# - PavedDrive: Paved driveway 车道铺砌情况
# - WoodDeckSF: Wood deck area in square feet 实木地板面积
# - OpenPorchSF: Open porch area in square feet 开放式门廊面积
# - EnclosedPorch: Enclosed porch area in square feet 封闭式门廊面积
# - 3SsnPorch: Three season porch area in square feet 时令门廊面积
# - ScreenPorch: Screen porch area in square feet 屏风门廊面积
# - PoolArea: Pool area in square feet 游泳池面积
# - PoolQC: Pool quality 游泳池质量
# - Fence: Fence quality 围栏质量
# - MiscFeature: Miscellaneous feature not covered in other categories 其它条件中未包含部分的特性
# - MiscVal: $Value of miscellaneous feature 杂项部分价值
# - MoSold: Month Sold 卖出月份
# - YrSold: Year Sold 卖出年份
# - SaleType: Type of sale 出售类型
# - SaleCondition: Condition of sale 出售条件
#####################################################################################

import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import norm, skew #统计常用函数

# 导入可视化工具
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pylab as pylab
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

data_train= pd.read_csv("./train.csv")
data_test= pd.read_csv("./test.csv")
# print("Size of training data set：{}\n Size of testing data set：{}".format(data_train.shape, data_test.shape))

data_train = data_train.drop(['Id'],axis=1)
data_test = data_test.drop(['Id'],axis=1)
# print(data_train.info())
data_clean = [data_train,data_test]
target = "SalePrice"


data_train = data_train.drop(data_train[(data_train['GrLivArea']>4000) & (data_train['SalePrice']<300000)].index)


(mu, sigma) = norm.fit(data_train['SalePrice'])





meaningful_zero=['PoolQC', 'FireplaceQu','GarageQual','GarageCond','FireplaceQu','GarageFinish','Fence','GarageType', 'GarageCars', 'GarageArea',
            'BsmtFinType2','BsmtQual','BsmtCond','BsmtFinType1','MasVnrType','MiscFeature','Alley','Fence','BsmtExposure']#合法的NA情况，比如没有车库，没有泳池

cont_NA=["LotFrontage ","LotFrontage "]
discrete_NA=['Electrical','MasVnrType', 'MSZoning','KitchenQual','Exterior1st','Exterior2nd','SaleType','Utilities'] # Utilities用allpub提升精度

for na in meaningful_zero:
    data_train[na] = data_train[na].fillna("None")
    data_test[na] = data_test[na].fillna("None")#给合法缺失情况直接赋值：无此项 none

data_train['LotFrontage'] = data_train['LotFrontage'].groupby(by=data_train['Neighborhood']).apply(lambda x: x.fillna(x.mean()))
data_train['MasVnrArea'] = data_train['MasVnrArea'].fillna(data_train['MasVnrArea'].mean())
data_test['LotFrontage'] = data_test['LotFrontage'].groupby(by=data_test['Neighborhood']).apply(lambda x: x.fillna(x.mean()))
data_test['MasVnrArea'] = data_test['MasVnrArea'].fillna(data_test['MasVnrArea'].mean())###用众数还是用平均值？

for na in discrete_NA:
    data_train[na] = data_train[na].fillna(data_train[na].mode()[0])
    data_test[na] = data_test[na].fillna(data_test[na].mode()[0])

data_train["Functional"] = data_train["Functional"].fillna("Typ") #因为描述中写的默认是typical情况
data_test["Functional"] = data_test["Functional"].fillna("Typ")


for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    data_train[col] = data_train[col].fillna(0)
    data_test[col] = data_test[col].fillna(0)

# 缺失GarageYrBlt，是因为不存在车库，这里由于年份是数值型，这里用较老年份1920代替
data_train['GarageYrBlt'] = data_train['GarageYrBlt'].fillna(1900)
data_test['GarageYrBlt'] = data_test['GarageYrBlt'].fillna(1900)

# missing_train_data = pd.DataFrame({'Missing Number': data_train.isnull().sum().sort_values(ascending=False)})
# missing_train_data = missing_train_data.drop(missing_train_data[missing_train_data['Missing Number']==0].index)
# missing_test_data = pd.DataFrame({'Missing Number':data_test.isnull().sum().sort_values(ascending=False)})
# missing_test_data = missing_test_data.drop(missing_test_data[missing_test_data['Missing Number']==0].index)
#
# print(missing_test_data)

data_train['TotalSF'] = data_train['TotalBsmtSF'] + data_train['1stFlrSF'] + data_train['2ndFlrSF']
data_test['TotalSF'] = data_test['TotalBsmtSF'] + data_test['1stFlrSF'] + data_test['2ndFlrSF']

for dataset in data_clean:
#MSSubClass
    dataset['MSSubClass'] = dataset['MSSubClass'].astype(str)

#OverallCond
    dataset['OverallCond'] = dataset['OverallCond'].astype(str)
    dataset['OverallQual'] = dataset['OverallQual'].astype(str)
#Year and month sold
    dataset['YrSold'] = dataset['YrSold'].astype(str)
    dataset['MoSold'] = dataset['MoSold'].astype(str)

train_dummy = pd.get_dummies(data_train)
test_dummy = pd.get_dummies(data_test)
numeric_train_cols = data_train.columns[data_train.dtypes != 'object']  # find numerical
numeric_test_cols = data_test.columns[data_test.dtypes != 'object'] #找到纯数值项的集合

# cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
#         'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
#         'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
#         'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
#         'YrSold', 'MoSold')
# from sklearn.preprocessing import LabelEncoder
# for c in cols:
#     label = LabelEncoder()
#     data_train[c] = label.fit_transform(data_train[c])
#     data_test[c] = label.fit_transform(data_test[c])
# 查看数据仍是字符串的特征有哪些
# data_train.dtypes[data_train.dtypes == 'object'].index

feats_numeric = data_train.dtypes[data_train.dtypes != "object"].index
data_skewness = pd.DataFrame({'Skew' :data_train[feats_numeric].apply(lambda x: skew(x)).sort_values(ascending=False)})
data_skewness = data_skewness[abs(data_skewness) > 0.75]

from scipy.special import boxcox1p
feats_skewed = data_skewness.index
lam = 0.15
for feat in feats_skewed:
    #all_data[feat] += 1
    data_train[feat] = boxcox1p(data_train[feat], lam)
# 对test测试集的偏差特征值进行变换
feats_numeric_test = data_test.dtypes[data_test.dtypes != "object"].index
data_skewness = pd.DataFrame({'Skew' :data_test[feats_numeric_test].apply(lambda x: skew(x)).sort_values(ascending=False)})
data_skewness = data_skewness[abs(data_skewness) > 0.75]
from scipy.special import boxcox1p
feats_skewed = data_skewness.index
lam = 0.15
for feat in feats_skewed:
    #all_data[feat] += 1
    data_test[feat] = boxcox1p(data_test[feat], lam)

