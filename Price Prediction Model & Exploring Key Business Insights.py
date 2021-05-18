#!/usr/bin/env python
# coding: utf-8

# # Step-1 Reading and Understanding Data

# 1)Importing data using the pandas library
# 2)Understanding the structure of the data

# In[127]:


# !pip install numpy
# !pip install pandas
# !pip install matplotlib
# !pip install dataprep
# !pip install sklearn
# !pip install seaborn
# !pip install xgboost


# ## Importing the Libraries

# In[128]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataprep.eda import create_report


# ## Load data into the data frame

# In[129]:


data = pd.read_csv("C:\\Users\\vishw\\OneDrive\\Desktop\\vehicles.csv")
data.head()


# ## The above dataset is a Vehicle Dataset for 1 month . Main task in this project is to build a Price Prediction Model for various different cars

# ## No. of rows and columns in the dataframe intially

# In[130]:


data.shape


# ## Checking no. of rows and datatype of every column

# In[131]:


data.info()


# In[132]:


data.dtypes


# ## No. of Null Values in every column in the dataframe

# In[133]:


print('Null Values')
print(data.isnull().sum())


# # Step-2 Data Cleaning and Preparation

# ## Checking the duplicate rows in the dataframe

# In[134]:


data.loc[data.duplicated()]


# ## No. of Column with different features

# In[135]:


data.columns


# ## Missing percentage of data for various features

# In[136]:


miss_percent = (data.isnull().sum() / len(data)) * 100
missing = pd.DataFrame({"percent":miss_percent, 'count':data.isnull().sum()}).sort_values(by="percent", ascending=False)
missing.loc[missing['percent'] > 0]


# ## #Removing columns: Passengers, carfax_url, engine, vin, description, trim, province

# In[137]:


data.drop(['passengers', 'carfax_url','engine','vin','description','trim','province'], axis=1, inplace=True)


# ## Removing rows with missing price, year, mileage, drivetrain, transmission, fuel_type

# In[138]:


data.drop(data[data['price'].isna()].index, inplace = True)
data.drop(data[data['year'].isna()].index, inplace = True)
data.drop(data[data['mileage'].isna()].index, inplace = True)
data.drop(data[data['drivetrain'].isna()].index, inplace = True)
data.drop(data[data['transmission'].isna()].index, inplace = True)
data.drop(data[data['fuel_type'].isna()].index, inplace = True)


# ## No. of rows and columns in the dataset after cleaning

# In[139]:


data.shape


# In[140]:


data.head()


# ## Feature Engineering

# ## Calculating difference with first_date_seen and last_date_seen i.e. car arrived and sold/unsold

# In[141]:


data['first_date_seen'] = pd.to_datetime(data['first_date_seen'])
data['last_date_seen'] = pd.to_datetime(data['last_date_seen'])
data['Difference'] = abs(data['first_date_seen'] - data['last_date_seen']).dt.days


# ## Creating age of car from year i.e. car manufactured date

# In[142]:


data['age_of_car'] = 2021 - data['year']


# In[143]:


data.head()


# ## Car Situation i.e. whether the car is sold/unsold based on last_date_seen

# In[144]:


data['Car_situation'] = np.where(data['last_date_seen'] == '2021-05-03' , 'Unsold', 'Sold')


# ## Removing featuress: id, first_date_seen, last_date_seen, model, color, seller_name for model creation

# In[145]:


data.drop(['id', 'first_date_seen','last_date_seen','model','color','seller_name'], axis=1, inplace=True)


# ## Removing Outliers from the data on the basis of years (considering years from 2000 to 2021)

# In[146]:


data = data[data['year'] >= 2000]
data = data[data['year'] <= 2021]
data.shape


# ## Grouping transmission column into Automatic and Manual

# In[147]:


data['transmission'] = data['transmission'].str.replace(r'(^.*Automatic.*$)', 'Automatic')
data['transmission'] = data['transmission'].str.replace(r'(^.*CVT.*$)', 'Automatic')
data['transmission'] = data['transmission'].str.replace(r'(^.*Manual.*$)', 'Manual')
data['transmission'] = data['transmission'].str.replace(r'(^.*Sequential.*$)', 'Manual')


# ## Grouping Fuel Types into Gasoline, Hybrid, Electric, Diesel, Others

# In[148]:


data['fuel_type'] = data['fuel_type'].str.replace(r'(^.*Hybrid.*$)', 'Hybrid')
data['fuel_type'] = data['fuel_type'].str.replace(r'(^.*Gas.*$)', 'Gasoline')
data['fuel_type'] = data['fuel_type'].str.replace(r'(^.*Unleaded.*$)', 'Gasoline')
data['fuel_type'] = data['fuel_type'].str.replace(r'(^.*Flexible.*$)', 'Gasoline')
data['fuel_type'] = data['fuel_type'].str.replace(r'(^.*Other.*$)', 'Other')


# ## Grouping Seller Name into Private Seller and Other Seller

# In[149]:


data['seller_type'] = np.where(data['is_private'] == True, 'Private Seller', 'Other Seller')


# ## Dropping feature: is_private after classifying into binary

# In[150]:


data.drop(['is_private'], axis=1, inplace=True)


# ## Grouping various body types of cars

# In[151]:


data['body_type'] = data['body_type'].str.replace(r'(^.*Truck.*$)', 'Truck')
data['body_type'] = data['body_type'].str.replace(r'(^.*Cab.*$)', 'Cab')
data['body_type'] = data['body_type'].str.replace(r'(^.*Wagon.*$)', 'Wagon')
data[['body_type']] = data[['body_type']] .replace(['Compact','Sedan'],'Sedan')
data[['body_type']] = data[['body_type']] .replace(['Super Crew','Roadster','Avant','Cutaway'],'Others')


# ## Grouping drivetrains feature of cars

# In[152]:


data['drivetrain'] = data['drivetrain'].str.replace(r'(^.*4.*$)', '4WD')


# ## Finding no. of missing values in the dataset

# In[153]:


miss_percent = (data.isnull().sum() / len(data)) * 100
missing = pd.DataFrame({"percent":miss_percent, 'count':data.isnull().sum()}).sort_values(by="percent", ascending=False)
missing.loc[missing['percent'] > 0]


# ## Removing missing values from the dataset

# In[154]:


data = data.dropna(how='any',axis=0)


# ## No. of rows and columns in the dataset after cleaning

# In[155]:


data.shape


# ## Though there are 108 Duplicate rows but cannot be removed because there can be multiple cars with same features after cleaning the data or dealer might have multiple cars

# In[156]:


data.duplicated().sum()


# ## Calculating Price Percentiles to categorize in 'Low','Medium' and 'High' Groups

# In[157]:


print(data.price.describe(percentiles = [0.25,0.33,0.67,0.85,0.90,1]))


# ## Removing Outliers with price <= 1500

# In[158]:


data = data[data['price'] >= 1500]


# In[159]:


#Grouping Price
data['price_category'] = data['price'].apply(lambda x : "Low" if x < 15000 
                                                     else ("Medium" if 15000 <= x < 30000
                                                           else "High"))


# ## Calculating Mileage Percentiles to categorize in 'Low','Medium' and 'High' Groups

# In[160]:


print(data.mileage.describe(percentiles = [0.25,0.50,0.75,0.85,0.90,1]))


# ## Mileage data is divided into three category using priori

# In[161]:


data['mileage_category'] = data['mileage'].apply(lambda x : "Low" if x < 100000 
                                                     else ("Medium" if 100000 <= x < 200000
                                                           else "High"))


# ## No. of rows and columns in the dataset after adding Price_Category and Mileage_Category

# In[162]:


data.shape
data.head()


# ## Quick Exploratory Data Analysis Report with Overview, Interactions, Correlations, Missing Values

# #### Before applying models, looking at the features and also relationship with each other by visualization of data.

# In[163]:


create_report(data)


# ## Key Findings:
# 1)Correlation between Price and Mileage according to Pearson test is 0.37  i.e. moderate relationship.
# 2)Correlation between Price and Age of Car according to Pearson test is 0.39 i.e. moderate relationship.
# 3)Correlation between Age of Car and Mileage according to Pearson test is 0.65 i.e. Strong relationship.

# # Step-3 Exploratory Data Analysis

# ## Finding relationship between Price and Mileage in the dataset

# In[164]:


plt1 = sns.scatterplot(x = 'mileage', y = 'price', data = data)
plt1.set_xlabel('Mileage')
plt1.set_ylabel('Price of Car (Dollars)')
plt.show()


# # Visualizing the data

# ## Finding the distribution of Price and Mileage  by creating Distribution Plot

# In[165]:


plt.figure(figsize=(16,4))
plt.subplot(1, 3, 1)
#Distribution plot for Price
axis = sns.distplot(data['price'], color = 'red')
plt.title("Distribution plot for Price")
plt.subplot(1, 3, 2)
#Distribution plot for Mileage
axis = sns.distplot(data['mileage'], color='limegreen')
plt.title("Distribution plot for Mileage")


# #### Findings: 
# 1) The plot seems to be right skewed i.e. most prices in the dataset are low.
# 2) There is a significant difference between the mean and median of the price distribution.

# ## Creating Body Class v/s Price plot and Drivetrain v/s Price plot

# In[166]:


#Body Class v/s Price
axis = sns.barplot(x="body_type", y="price", data=data)
for i in axis.patches:
             axis.annotate("%.f" % i.get_height(), (i.get_x() + i.get_width() / 2., i.get_height()),
                 ha='right', va='center', fontsize=11, color='black', xytext=(0, 25), rotation = 90,
                 textcoords='offset points')
plt.xticks(rotation=45, horizontalalignment='right')
plt.xlabel('Body Classes')
plt.ylabel('Price')
plt.title('Average Price across various Body Types')
plt.show()
#Drive Train Class v/s Price
axis = sns.barplot(x="drivetrain", y="price", data=data)
for i in axis.patches:
             axis.annotate("%.f" % i.get_height(), (i.get_x() + i.get_width() / 2., i.get_height()),
                 ha='center', va='center', fontsize=11, color='black', xytext=(0, -50), 
                 textcoords='offset points')
plt.xlabel('Drive Train Classes')
plt.ylabel('Price')
plt.title('Average Price across various Drive Trains')
plt.show()


# ## Creating Transmission v/s Price plot and Sellers v/s Price plot

# In[167]:


#Transmission v/s Price
axis = sns.barplot(x="transmission", y="price", data=data)
for i in axis.patches:
             axis.annotate("%.f" % i.get_height(), (i.get_x() + i.get_width() / 2., i.get_height()),
                 ha='center', va='center', fontsize=11, color='black', xytext=(0, 7),
                 textcoords='offset points')
plt.xlabel('Transmission Classes')
plt.ylabel('Price')
plt.title('Transmission w.r.t Price')
plt.show()
#Sellers v/s Price
axis = sns.barplot(x="seller_type", y="price", data=data)
for i in axis.patches:
             axis.annotate("%.f" % i.get_height(), (i.get_x() + i.get_width() / 2., i.get_height()),
                 ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                 textcoords='offset points')
plt.xlabel('Seller Classes')
plt.ylabel('Price')
plt.title('Sellers w.r.t Price')
plt.show()


# ## Creating Body Class v/s Mileage and Drivetrain Class v/s Mileage plot

# In[168]:


#Body Class v/s Mileage
#plt.rcParams["figure.figsize"] = (20,3)
axis = sns.barplot(x="body_type", y="mileage", data=data)
for i in axis.patches:
             axis.annotate("%.0f" % i.get_height(), (i.get_x() + i.get_width() / 2., i.get_height()),
                 ha='center', va='center', fontsize=11, color='black', xytext=(0, -60), rotation = 90,
                 textcoords='offset points')
plt.xticks(rotation=45, horizontalalignment='right')
plt.xlabel('Body Classes')
plt.ylabel('Mileage')
plt.title('Mileage across various Body Types')
plt.show()
#Drive Train Class v/s Mileage
axis = sns.barplot(x="drivetrain", y="mileage", data=data)
for i in axis.patches:
             axis.annotate("%.0f" % i.get_height(), (i.get_x() + i.get_width() / 2., i.get_height()),
                 ha='center', va='center', fontsize=11, color='black', xytext=(0, -70), rotation = 90,
                 textcoords='offset points')
plt.xlabel('Drive Train Classes')
plt.ylabel('Mileage')
plt.title('Mileage across various Drive Trains')
plt.show()


# ## Creating Transmission v/s Mileage and Sellers v/s Mileage plot

# In[169]:


#Transmission v/s Mileage
axis = sns.barplot(x="transmission", y="mileage", data=data)
for i in axis.patches:
             axis.annotate("%.0f" % i.get_height(), (i.get_x() + i.get_width() / 2., i.get_height()),
                 ha='center', va='center', fontsize=11, color='black', xytext=(0, 10),
                 textcoords='offset points')
plt.xlabel('Transmission Classes')
plt.ylabel('Mileage')
plt.title('Transmission w.r.t Mileage')
plt.show()
#Sellers v/s Mileage
axis = sns.barplot(x="seller_type", y="mileage", data=data)
for i in axis.patches:
             axis.annotate("%.0f" % i.get_height(), (i.get_x() + i.get_width() / 2., i.get_height()),
                 ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                 textcoords='offset points')
plt.xlabel('Seller Classes')
plt.ylabel('Mileage')
plt.title('Sellers w.r.t Mileage')
plt.show()


# In[170]:


# Number of cars in different Price Category.
plt.figure(figsize=(16,4))
plt.subplot(1, 2, 1)
axis = sns.countplot(x="price_category", data=data)
for i in axis.patches:
    h = i.get_height()
    axis.text(i.get_x()+i.get_width()/2.,h+10,
              '{:1.0f}%'.format(h/len(data)*100),ha="center",fontsize=10) 
plt.xlabel('Price Classes')
plt.ylabel('Number of Cars')
plt.title('Number of cars in different Price Category')
plt.show()
# Number of cars in different Mileage Category
plt.subplot(1, 2, 2)
axis = sns.countplot(x="mileage_category", data=data)
for i in axis.patches:
    h = i.get_height()
    axis.text(i.get_x()+i.get_width()/2.,h+10,
              '{:1.0f}%'.format(h/len(data)*100),ha="center",fontsize=10) 
plt.xlabel('Mileage Classes')
plt.ylabel('Number of Cars')
plt.title('Number of cars in different Mileage Category')
plt.show()


# ## No. of cars w.r.t Fuel Classes

# In[171]:


#No. of cars w.r.t Fuel Classes
axis = sns.countplot(x="fuel_type", data=data)
for i in axis.patches:
    h = i.get_height()
    axis.text(i.get_x()+i.get_width()/2.,h+10,
              '{:1.2f}%'.format(h/len(data)*100),ha="center",fontsize=10) 
plt.xlabel('Fuel Classes')
plt.ylabel('No. of Cars with particular fuel types')
plt.title('Number of cars w.r.t Fuel Classes')
plt.show()


# ## #Top 20 cars w.r.t Car Types

# In[172]:


axis = sns.countplot(x="make", data=data,order=data.make.value_counts().iloc[:20].index)
for i in axis.patches:
    h = i.get_height()
    axis.text(i.get_x()+i.get_width()/2.,h+10,
              '{:1.0f}%'.format(h/len(data)*100),ha="center",fontsize=10,rotation = 45) 
plt.xlabel('Car Types')
plt.xticks(rotation=90)
plt.ylabel('No. of Cars')
plt.title('Number of cars w.r.t Car Types')
plt.show()


# ## Top 20 cities with most cars

# In[173]:


axis = sns.countplot(x="city", data=data,order=data.city.value_counts().iloc[:20].index)
for i in axis.patches:
    h = i.get_height()
    axis.text(i.get_x()+i.get_width()/2.,h+10,
              '{:1.0f}%'.format(h/len(data)*100),ha="center",fontsize=10, rotation = 15)
plt.xlabel('City Names')
plt.xticks(rotation=90)
plt.ylabel('No. of Cars')
plt.title('Number of cars w.r.t City Names')
plt.show()


# 

# In[174]:


#data.to_csv(r'C:/Users/vishw/OneDrive/Desktop/try.csv')


# # Model Building

# ## Importing sklearn Library for Model Building and applying functions for fitting the models

# In[175]:


import sklearn
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score,KFold
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

from sklearn.linear_model import LinearRegression

from sklearn import metrics, model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV


# ## Creating a dataframe by copying previous dataframe

# In[176]:


df = data.copy()


# ## Creating target and feature variables from the dataframe

# In[177]:


target = data['price']


# In[178]:


feature = df.drop(['year','mileage_category','price','city','longitude','latitude','Difference','price_category'], axis=1)


# ## Performing One Hot encoding on the feature variables for creating training dataset

# In[179]:


feature = pd.get_dummies(feature)
feature.head()


# In[180]:


features=feature.values
target=target.values


# ## Split data in training and testing data in 80-20 form by random allocation:

# In[181]:


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# ## RandomForesting

# In[182]:


RF_regressor=RandomForestRegressor(n_estimators=200,random_state=42,max_depth = 9)
RF_regressor.fit(X_train,y_train)
target_predict=RF_regressor.predict(X_test)


# In[183]:


def printAccuracy(regressor):
    print("R2 score of training data: ",r2_score(y_train,regressor.predict(X_train)))
    print("R2 score of testing data: ",r2_score(y_test, target_predict))
    print("Mean Square error of training data: ",mean_squared_error(y_train,regressor.predict(X_train)))
    print("Mean Square error of testing data: ",mean_squared_error(y_test, target_predict))
    print("Mean absolute error of training data: ",mean_absolute_error(y_train,regressor.predict(X_train)))
    print("Mean Absolute error of testing data: ",mean_absolute_error(y_test, target_predict))


# In[184]:


printAccuracy(RF_regressor)


# ## Checking important features generated in the dataset using Random Forest

# In[185]:


plt.figure(figsize=[12,6])
feat_importances = pd.Series(RF_regressor.feature_importances_, index=feature.columns)
feat_importances.nlargest(36).plot(kind='barh')
plt.show()


# ## XG Boost

# In[186]:


import xgboost as xgb


# In[187]:


xgBoost_regressor = xgb.XGBRegressor(colsample_bytree=1,              
                 learning_rate=0.01,
                 max_depth=9,
                 n_estimators=250,                                                                    
                 seed=42)
xgBoost_regressor.fit(X_train,y_train)
target_predict=xgBoost_regressor.predict(X_test)


# In[188]:


printAccuracy(xgBoost_regressor)


# ## Understanding Feature Importance for XG Boost

# In[189]:


plt.figure(figsize=[12,6])
feat_importances = pd.Series(xgBoost_regressor.feature_importances_, index=feature.columns)
feat_importances.nlargest(36).plot(kind='barh')
plt.show()


# # Conclusion

# #### 1) Applying Random Forest and XG Boost on the dataset because we are predicting price with nearly 86 features so other algorithms would not fulfil the purpose.
# #### 2) As the test accuracy and train accuracy are pretty similar for both the models so it is neither underfitting nor overfitting.

# # Future Scope

# #### 1) Perform Hyperparameter Tuning while selecting the parameter for Random Forest and XG Boost.
# #### 2) Train the model using Neural Networks.
# 
