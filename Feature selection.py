##This is an automatic variable selection tool working like an elbow method.
##THe algorithm will fit into a linear regressor and pick the variables according to Recursive feature elimination with cross-validation
##Details: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html 

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

print("Optimal number of features based on mean test score : %d" % rfecv.n_features_)

##Show the optimal number of features based on mean of test score



min_features_to_select = 1

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
#plt.plot(range(1,11), rfecv.cv_results_.get('mean_test_score')[:10])
plt.show()

##Plot the line chart of number of features vs. cross-validation score. You will see there is an optimal number of features with the highest score. That would be the optimal number of features.