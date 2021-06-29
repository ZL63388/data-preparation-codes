# import packages
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# create a sample dataframe
my_df = pd.DataFrame({"Height": [1.98,1.77,1.76,1.80,1.64],
                      "Weight": [99,81,70,86,82]})


# Standardisation
scale_standard = StandardScaler()

scale_standard.fit_transform(my_df)
my_df_standardised = pd.DataFrame(scale_standard.fit_transform(my_df), columns = my_df.columns)


# Normalisation
scale_norm = MinMaxScaler()
scale_norm.fit_transform(my_df)
my_df_normalised = pd.DataFrame(scale_norm.fit_transform(my_df), columns = my_df.columns)