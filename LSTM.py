#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


dfNoTransform = pd.read_csv(r'###.csv')
##path redacted for confidentiality


# In[3]:


dfNoTransform.info()


# In[4]:


dfNoTransform = dfNoTransform.drop(['Total'], axis=1)
dfNoTransform
#drop forecast target


# In[ ]:





# In[5]:


dfNoTransformNoDate = dfNoTransform.iloc[:,2:]
dfNoTransformNoDate
#Drop date


# In[6]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

corrNoTransformNoDate = dfNoTransformNoDate.corr()

# plot the variable correlatoin heatmap
corrGraph = sns.heatmap(corrNoTransformNoDate, 
        xticklabels=corrNoTransformNoDate.columns,
        yticklabels=corrNoTransformNoDate.columns)

corrGraph


# In[7]:


corrNoTransformNoDate


# In[8]:


corrNoTransformNoDate.to_csv("###.csv")


# In[9]:


df = pd.read_csv(r'###.csv')


# In[10]:


df.info()


# In[11]:


df.describe(include='all')


# In[12]:


dfNoDate = df.drop(['Total','Date'], axis=1)
dfNoDate


# In[13]:


lg=df['lg']


# In[14]:


# KPSS test
from statsmodels.tsa.stattools import kpss
#def kpss_test(series, **kw):    
 #   statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    # Format Output
 #   print(f'KPSS Statistic: {statistic}')
 #   print(f'p-value: {p_value}')
 #   print(f'num lags: {n_lags}')
 #   print('Critial Values:')
 #   for key, value in critical_values.items():
 #       print(f'   {key} : {value}')
 #   print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')

#kpss_test(lg)


# In[15]:


#ADF test
from statsmodels.tsa.stattools import adfuller


def adf_test(timeseries):
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)

adf_test(lg)
 
###p-value <0.05 meaning data is stationary


# In[16]:


lg.plot()


# In[17]:


corr = df.corr()
corr.to_csv("Variable Correlations.csv")


# In[18]:


#scatter point of variables
#df.plot(kind="scatter", x="subscriber",y="views")


# In[19]:


#Specifying X and y


# In[20]:


dfNoTransform


# In[21]:


#XwithDate = df.drop(['lg','ur','cpi','bbb','ys','yc','mor','prime','hp','cre','vix',
             #'three','dthree','dthreel1','dthreel2','dthreel3','dthreel4',
            #'five','dfive','dfivel1','dfivel2','dfivel3','dfivel4',
            #'ten','dten','dtenl1','dtenl2','dtenl3','dtenl4'], axis=1)

            #             'lg','rinc','ninc','three','five','ten','mor','cre','rgdp',
#             'rgdpl1','rgdpl2','rgdpl3','rgdpl4',
#             'rincl1',
#             'three','dthree','dthreel1','dthreel2','dthreel3','dthreel4',
#             'five','dfive','dfivel1','dfivel2','dfivel3','dfivel4',
#             'ten','dten','dtenl1','dtenl2','dtenl3','dtenl4',
#             'mor','dmor','dmorl1','dmorl2','dmorl3','dmorl4',
#             'cre','lcre','dlcre','dlcrel1','dlcrel2','dlcrel3','dlcrel4'
            
y = df[['lg','Date']]

X = df.drop(['lg','Total'], axis=1)
            
X = X.set_index('Date')
y = y.set_index('Date')

##Variable selected. Here variables are public economic data.


# In[22]:


X.head()


# In[23]:


y.head()


# In[24]:


from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
 
fields = X.columns    
    
#use linear regression as the model
lr = LinearRegression()
#rank all features, i.e continue the elimination until the last one
rfecv = RFECV(lr, step=1, cv=5)
#rfecv.fit(X_train_transformed,y_train)
rfecv.fit(X,y) 
    
print ("Features sorted by their rank:")
print (sorted(zip(map(lambda x: round(x, 4), rfecv.ranking_), fields)))


# In[25]:


print("Optimal number of features based on mean test score : %d" % rfecv.n_features_)

min_features_to_select = 1

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
#plt.plot(range(1,11), rfecv.cv_results_.get('mean_test_score')[:10])
plt.show()


# In[26]:


#rfecv.cv_results_.get('mean_test_score')


# In[27]:


###PCA Variable Selection
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import StandardScaler # for standardizing the Data
from sklearn.decomposition import PCA # for PCA calculation


# In[28]:


Xv = dfNoDate.values # getting all values as a matrix of dataframe 
sc = StandardScaler() # creating a StandardScaler object
X_std = sc.fit_transform(Xv) # standardizing the data


# In[29]:


pca = PCA()
X_pca = pca.fit(X_std)


# In[30]:


plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


# In[31]:


#num_components = 9
#pca = PCA(num_components)  
#X_pca = pca.fit_transform(X_std) # fit and reduce dimension


# In[32]:


pca = PCA(n_components = 0.99)
X_pca = pca.fit_transform(X_std) # this will fit and reduce dimensions
print(pca.n_components_) # one can print and see how many components are selected. In this case it is 4 same as above we saw in step 5


# In[33]:


pd.DataFrame(pca.components_, columns = dfNoDate.columns)


# In[34]:


n_pcs= pca.n_components_ # get number of component
# get the index of the most important feature on EACH component
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
initial_feature_names = dfNoDate.columns
# get the most important feature names
most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

most_important_names


# In[35]:


from collections import Counter
Counter(most_important_names)


# In[36]:


sort_orders = sorted(Counter(most_important_names).items(), key=lambda x: x[1], reverse=True)

for i in sort_orders:
    print(i[0],':', i[1])


# In[37]:


#most_important


# In[38]:


#pca.components_


# In[39]:


###Multivariate regression


# In[40]:


X


# In[41]:


X_subset=X[['dmorl1', 'dlhp','dlcrel2']]
X_subset


# In[42]:


###split training, validation and test dataset, it is stratified as category ID has imbalanced distribution

from sklearn.model_selection import train_test_split


X_train_subset, X_test_subset, y_train, y_test = train_test_split(
     X_subset, y, test_size=0.2, random_state=42, shuffle=False)

X_val_subset, X_test_subset, y_val, y_test = train_test_split(
     X_test_subset, y_test, test_size=0.5, random_state=42,  shuffle=False)


# In[43]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
        ('std_scaler', StandardScaler()) 
    ])

X_train_subset_transformed = pd.DataFrame(pipeline.fit_transform(X_train_subset),columns = X_train_subset.columns)


# In[44]:


X_train_subset_transformed.head()


# In[45]:


lr.fit(X_train_subset_transformed, y_train)


# In[46]:


import statsmodels.api as sm
from statsmodels.api import OLS

X_train_subset_transformed_with_const = sm.add_constant(X_train_subset_transformed)

OLS(y_train.reset_index(drop=True),X_train_subset_transformed_with_const).fit().summary()


# In[47]:


X_subset_with_const = sm.add_constant(X_subset.reset_index(drop=True))

OLS(y.reset_index(drop=True),X_subset_with_const).fit().summary()


# In[48]:


#scatter point of variables
df.plot(kind="scatter", x="dlhpl4",y="lg")


# In[49]:


df.plot(kind="scatter", x="dlcrel4",y="lg")


# In[50]:


df.plot(kind="scatter", x="dprimel1",y="lg")


# In[51]:


from sklearn.metrics import mean_squared_error, r2_score,  mean_absolute_error
def report_regression_metrics(model, X, y, plots=False, y_pred = []):
    if not len(y_pred):
        y_pred = model.predict(X)
    if plots:
        
        #plt.figure()
        #sns.distplot(y - y_pred) 
        #plt.figure()
        #sns.scatterplot(x=y, y=pd.DataFrame(y_pred,index=X.index))
        plt.plot(y, y_pred)
        
    print("MSE:", mean_squared_error(y, y_pred))
    print("R2 score:", r2_score(y, y_pred))
   # print("Max error:", max_error(y, y_pred))
    print("mean_absolute_error:", mean_absolute_error(y, y_pred))


# In[52]:


print("Train metrics:")
report_regression_metrics(lr, X_train_subset_transformed, y_train)

print("Val metrics:")
report_regression_metrics(lr, X_val_subset, y_val)

print("Test metrics:")
report_regression_metrics(lr, X_test_subset, y_test)


# In[53]:


y_pred = lr.predict(X_train_subset_transformed)
y_pred = pd.DataFrame(y_pred,index = X_train_subset.index)
y_pred.head()


# In[54]:


#y_train


# In[55]:


plt.plot(y_pred, color = "green")
plt.plot(y_train, color = "blue")
plt.title("Predict vs Actual (Training set)")
plt.xlabel("Quarter-Year")
plt.ylabel("LG")
plt.xticks(rotation = 90)
fig = plt.gcf()
fig.set_size_inches(24, 8)
plt.show()


# In[56]:


y_pred = lr.predict(X_val_subset)

y_pred = pd.DataFrame(y_pred,index = X_val_subset.index)


# In[57]:


y_pred


# In[58]:


index = X_val_subset.index
index


# In[59]:



y_val = pd.DataFrame(y_val,index = y.index)


# In[60]:


#y_val


# In[61]:


plt.plot(y_pred, color = "green")
plt.plot(y_val, color = "blue")
plt.title("Predict vs Actual (Validation set)")
plt.xlabel("Quarter-Year")
plt.ylabel("LG")
plt.xticks(rotation = 90)
fig = plt.gcf()
fig.set_size_inches(24, 8)
plt.show()


# In[62]:


y_pred = lr.predict(X_test_subset)

y_pred = pd.DataFrame(y_pred,index = X_test_subset.index)


# In[63]:


plt.plot(y_pred, color = "green")
plt.plot(y_test, color = "blue")
plt.title("Predict vs Actual (Test set)")
plt.xlabel("Quarter-Year")
plt.ylabel("LG")
plt.xticks(rotation = 90)
fig = plt.gcf()
fig.set_size_inches(24, 8)
plt.show()


# In[64]:


###An RNN Model for forecast


# In[65]:


dfNoTransform


# In[66]:


y = dfNoTransform[['lg']]

X = dfNoTransform.drop(['lg','Date'], axis=1)


# In[67]:


###split training, validation and test dataset, it is stratified as category ID has imbalanced distribution

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.3, random_state=42, shuffle=False)


# In[68]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
        ('std_scaler', StandardScaler()) 
    ])

X_train_transformed = pd.DataFrame(pipeline.fit_transform(X_train),columns = X_train.columns)
X_test_transformed = pd.DataFrame(pipeline.fit_transform(X_test),columns = X_train.columns)


# In[69]:


X_train_transformed


# In[70]:


X_train_transformed.shape


# In[71]:


X_train = X_train_transformed.values.reshape((X_train_transformed.shape[0], 1, X_train_transformed.shape[1]))
X_test = X_test_transformed.values.reshape((X_test_transformed.shape[0], 1, X_test_transformed.shape[1]))
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# In[72]:


X_test


# In[73]:


y_test


# In[74]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from matplotlib import pyplot


# In[75]:


# design network
model = Sequential()
model.add(LSTM(200, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit network
history = model.fit(X_train, y_train, epochs=10, batch_size=72, validation_data=(X_test, y_test), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# In[76]:


...
# make a prediction
yhat = model.predict(X_test)


# In[77]:


X_test.shape


# In[78]:


from keras.layers import concatenate


# In[79]:


X_test = X_test.reshape((X_test.shape[0], X_test.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, X_test[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
y_test = y_test.reshape((len(y_test), 1))
inv_y = concatenate((y_test, X_test[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


# In[ ]:




