# import packages
import pandas as pd
import numpy as np


# create a sample dataframe
my_df = pd.DataFrame({"A": [1,2,3,np.nan,5,np.nan,7],
                      "B": [4,np.nan,7,np.nan,1,np.nan,2]})


# finding missing values with pandas
my_df.isna()
my_df.isna().sum()


# dropping missing values with pandas
my_df.dropna()
my_df.dropna(how = "any")
