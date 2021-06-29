# import packages
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


# create a sample dataframe
my_df = pd.DataFrame({"A": [1,4,7,10,13],
                      "B": [3,6,9,np.nan,15],
                      "C": [2,5,np.nan,11,np.nan]})


# define the package into a variable
imputer = SimpleImputer()


# train the imputer with your dataframe using the .fit
imputer.fit(my_df)


# apply the imputer using the .tranform to your dataframe
imputer.transform(my_df)


# print results (results will be in array form)
my_df1 = imputer.transform(my_df)


# run again
imputer.fit_transform(my_df)


# to change the datatype to dataframe 
my_df2 = pd.DataFrame(imputer.transform(my_df), columns = my_df.columns)


# impute only a specific column
imputer.fit_transform(my_df[["B"]])
my_df["B"] = imputer.fit_transform(my_df[["B"]])