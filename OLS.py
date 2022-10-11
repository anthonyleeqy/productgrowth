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
