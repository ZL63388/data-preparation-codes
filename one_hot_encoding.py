# import packages
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder



# create a sample dataframe
X = pd.DataFrame({"input1": [1,2,3,4,5],
                  "input2": ["A","A","B","B","C"],
                  "input3": ["X","X","X","Y","Y"]})


# put categorical variables in a list 
categorical_vars = ["input2", "input3"]


# instantiate the one hot encoder
one_hot_encoder = OneHotEncoder(sparse=False, drop = "first")


# apply the one hot encoder logic 
encoder_vars_array = one_hot_encoder.fit_transform(X[categorical_vars])


# create object for the feature names using the categorical variables
encoder_feature_names = one_hot_encoder.get_feature_names(categorical_vars)


# create a dataframe to hold the one hot encoded variables
encoder_vars_df = pd.DataFrame(encoder_vars_array, columns = encoder_feature_names)


# concatenate the new dataframe back to the original input variables dataframe
X_new = pd.concat([X.reset_index(drop=True), encoder_vars_df.reset_index(drop=True)], axis = 1)


# drop the orignal input 2 and input 3 as it is not needed anymore
X_new.drop(categorical_vars, axis = 1, inplace = True)