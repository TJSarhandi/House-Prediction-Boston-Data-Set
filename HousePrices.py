#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# In[39]:


test_df = pd.read_csv('HouseData/test.csv')


# In[40]:


train_df = pd.read_csv('HouseData/train.csv')


# In[42]:


full_df = pd.concat([train_df, test_df], sort=True).reset_index(drop=True)
full_df.shape


# In[37]:


sns.factorplot(x="LotArea", y ="SalePrice", data=train_df, kind="bar", size=5)
plt.show()


# In[ ]:


train_df.shape


# In[43]:


full_df.describe()


# In[48]:


corrMatrix = train_df.corr()
corrMatrix


# In[44]:


full_df.isnull().sum()


# In[17]:


sns.heatmap(train_df.isnull(), yticklabels=False, cbar=False)


# In[ ]:


train_df['MSZoning'].value_counts()


# In[ ]:


train_df.info()


# In[19]:


train_df.drop(['Alley'], axis=1, inplace=True)
train_df['BsmtCond'] = train_df['BsmtCond'].fillna(train_df['BsmtCond'].mode()[0])
train_df['LotFrontage'] = train_df['LotFrontage'].fillna(train_df['LotFrontage'].mode()[0])
train_df['BsmtQual'] = train_df['BsmtQual'].fillna(train_df['BsmtQual'].mode()[0])
train_df['FireplaceQu'] = train_df['FireplaceQu'].fillna(train_df['FireplaceQu'].mode()[0])
train_df['GarageType'] = train_df['GarageType'].fillna(train_df['GarageType'].mode()[0])
train_df.drop(['GarageYrBlt'], axis=1, inplace=True)
train_df['GarageFinish'] = train_df['GarageFinish'].fillna(train_df['GarageFinish'].mode()[0])
train_df['GarageQual'] = train_df['GarageQual'].fillna(train_df['GarageQual'].mode()[0])
train_df['GarageCond'] = train_df['GarageCond'].fillna(train_df['GarageCond'].mode()[0])
train_df.drop(['PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
train_df.drop(['Id'], axis =1, inplace =True)


# In[20]:


train_df.shape


# In[30]:


train_df.isnull().sum()


# In[27]:


train_df['BsmtExposure'] = train_df['BsmtExposure'].fillna(train_df['BsmtExposure'].mode()[0])
train_df['BsmtFinType1'] = train_df['BsmtFinType1'].fillna(train_df['BsmtFinType1'].mode()[0])
train_df['BsmtFinType2'] = train_df['BsmtFinType2'].fillna(train_df['BsmtFinType2'].mode()[0])
train_df['MasVnrType'] = train_df['MasVnrType'].fillna(train_df['MasVnrType'].mode()[0])
train_df['MasVnrArea'] = train_df['MasVnrArea'].fillna(train_df['MasVnrArea'].mode()[0])


# In[28]:


train_df.shape


# In[29]:


sns.heatmap(train_df.isnull(), yticklabels=False, cbar=False, cmap='coolwarm')


# In[ ]:


#Missing Values have been handled


# In[31]:


train_df.head()


# In[33]:


test_df.isnull().sum()


# In[ ]:


columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']


# In[ ]:


len(columns)


# In[ ]:


main_df = train_df.copy()


# In[ ]:


def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final


# In[ ]:


test_df.shape


# In[ ]:


test_df.head()


# In[ ]:


final_df=pd.concat([train_df,test_df],axis=0)


# In[ ]:


final_df.head()


# In[ ]:


final_df['SalePrice']


# In[ ]:


final_df.shape


# In[ ]:


final_df=category_onehot_multcols(columns)


# In[ ]:


final_df.shape


# In[ ]:


final_df = final_df.loc[:,~final_df.columns.duplicated()]


# In[ ]:


final_df


# In[ ]:


final_df.to_csv('finaldf.csv', index=False)


# In[ ]:


#New Train and Test data sets

df_train = final_df.iloc[:1422,:]
df_test = final_df.iloc[1422:,:]


# In[ ]:


df_train.shape


# In[ ]:


df_test.drop(['SalePrice'],axis=1, inplace=True)


# In[ ]:


x_train=df_train.drop(['SalePrice'], axis=1)
y_train=df_train['SalePrice']


# In[ ]:


import xgboost
classifier = xgboost.XGBRegressor()


# In[ ]:


#saving trained model
import pickle
filename = 'finalized_model.pkl'
pickle.dump(classifier,open(filename, 'wb'))


# In[ ]:


df_test.shape


# In[ ]:


classifier.fit(x_train, y_train)


# In[ ]:


y_pred=classifier.predict(df_test)


# In[ ]:


y_pred


# In[ ]:


##Create Sample Submission file and Submit using ANN
pred=pd.DataFrame(y_pred)
sub_df=pd.read_csv('sample_submission.csv')
datasets=pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns=['Id','SalePrice']
datasets.to_csv('sample_submission.csv',index=False)


# In[ ]:


pred.info


# In[ ]:


import xgboost
classifier=xgboost.XGBRegressor()


# In[ ]:


import xgboost
regressor=xgboost.XGBRegressor()


# In[ ]:


booster=['gbtree','gblinear']
base_score=[0.25,0.5,0.75,1]


# In[ ]:


## Hyper Parameter Optimization


n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
booster=['gbtree','gblinear']
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]

# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score
    }


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


# Set up the random search with 4-fold cross validation
random_cv = RandomizedSearchCV(estimator=regressor,
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state=42)


# In[ ]:


random_cv.fit(x_train,y_train)


# In[ ]:


random_cv.best_estimator_


# In[ ]:


regressor = xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.1, max_delta_step=0,
             max_depth=2, min_child_weight=2, missing=None, n_estimators=1000,
             n_jobs=2, nthread=None, objective='reg:linear', random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=1, verbosity=1)


# In[ ]:


regressor.fit(x_train, y_train)


# In[ ]:


y_pred = regressor.predict(df_test)


# In[ ]:


y_pred


# In[ ]:


#saving trained model
import pickle
filename = 'finalized_model.pkl'
pickle.dump(classifier,open(filename, 'wb'))


# In[ ]:


##Create Sample Submission file and Submit using ANN
pred=pd.DataFrame(y_pred)
sub_df=pd.read_csv('sample_submission.csv')
datasets=pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns=['Id','SalePrice']
datasets.to_csv('sample_submission.csv',index=False)

