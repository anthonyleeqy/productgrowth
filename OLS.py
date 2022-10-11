###split training, validation and test dataset, it is stratified as category ID has imbalanced distribution

from sklearn.model_selection import train_test_split


X_train_subset, X_test_subset, y_train, y_test = train_test_split(
     X_subset, y, test_size=0.2, random_state=42, shuffle=False)

X_val_subset, X_test_subset, y_val, y_test = train_test_split(
     X_test_subset, y_test, test_size=0.5, random_state=42,  shuffle=False)

###A data pipeline to standardize data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
        ('std_scaler', StandardScaler()) 
    ])

X_train_subset_transformed = pd.DataFrame(pipeline.fit_transform(X_train_subset),columns = X_train_subset.columns)


###Fit the data into an OLS model

lr.fit(X_train_subset_transformed, y_train)

import statsmodels.api as sm
from statsmodels.api import OLS

X_train_subset_transformed_with_const = sm.add_constant(X_train_subset_transformed)

OLS(y_train.reset_index(drop=True),X_train_subset_transformed_with_const).fit().summary()


###Evaluate model performance

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

print("Train metrics:")
report_regression_metrics(lr, X_train_subset_transformed, y_train)

print("Val metrics:")
report_regression_metrics(lr, X_val_subset, y_val)

print("Test metrics:")
report_regression_metrics(lr, X_test_subset, y_test)

###Print MSE, R^2, mean absolute error on training, validation and test datasets

y_pred = lr.predict(X_train_subset_transformed)
y_pred = pd.DataFrame(y_pred,index = X_train_subset.index)
y_pred.head()

plt.plot(y_pred, color = "green")
plt.plot(y_train, color = "blue")
plt.title("Predict vs Actual (Training set)")
plt.xlabel("Quarter-Year")
plt.ylabel("LG")
plt.xticks(rotation = 90)
fig = plt.gcf()
fig.set_size_inches(24, 8)
plt.show()

###Visualize the predicted value vs actual value