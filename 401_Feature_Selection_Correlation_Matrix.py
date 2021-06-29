# import packages
import pandas as pd


# import data
my_df = pd.read_csv("feature_selection_sample_data.csv")

# run correlation matrix
correlation_matrix = my_df.corr()