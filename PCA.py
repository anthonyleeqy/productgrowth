###PCA Variable Selection
%matplotlib inline

##A Principal Component Analysis Framework

from sklearn.preprocessing import StandardScaler # for standardizing the Data
from sklearn.decomposition import PCA # for PCA calculation

Xv = dfNoDate.values # getting all values as a matrix of dataframe 
sc = StandardScaler() # creating a StandardScaler object
X_std = sc.fit_transform(Xv) # standardizing the data

pca = PCA()
X_pca = pca.fit(X_std)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

pca = PCA(n_components = 0.99)
X_pca = pca.fit_transform(X_std) # this will fit and reduce dimensions
print(pca.n_components_) # one can print and see how many components are selected. In this case it is 4 same as above we saw in step 5

print(pd.DataFrame(pca.components_, columns = dfNoDate.columns))
##A quick view of all principal components


n_pcs= pca.n_components_ # get number of component
# get the index of the most important feature on EACH component
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
initial_feature_names = dfNoDate.columns
# get the most important feature names
most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

##We got the most important variables according to PCA, now we print out all the selected features and their frequency

sort_orders = sorted(Counter(most_important_names).items(), key=lambda x: x[1], reverse=True)

for i in sort_orders:
    print(i[0],':', i[1])