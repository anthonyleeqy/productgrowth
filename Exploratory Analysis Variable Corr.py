##A quick exploratory analysis on correlations between each variables
##When there are many available variable choices, a quick understanding of correlations can help to choose relevant variables and prevent overfitting

##import necessary packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

##a simple data ingestion, from csv file
dfNoTransform = pd.read_csv(r'###.csv')
##path redacted for confidentiality

dfNoTransform = dfNoTransform.drop(['Target'], axis=1)
dfNoTransform
#drop forecast target

dfNoTransformNoDate = dfNoTransform.iloc[:,2:]
dfNoTransformNoDate
#Drop date

import matplotlib.pyplot as plt
%matplotlib inline

corrNoTransformNoDate = dfNoTransformNoDate.corr()

# plot the variable correlatoin heatmap
corrGraph = sns.heatmap(corrNoTransformNoDate, 
        xticklabels=corrNoTransformNoDate.columns,
        yticklabels=corrNoTransformNoDate.columns)

corrGraph

##above show heat map of each variables

##Output correlation to csv
corrNoTransformNoDate.to_csv("###.csv")