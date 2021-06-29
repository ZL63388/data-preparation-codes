# import packages
import pandas as pd


# create a sample dataframe
my_df = pd.DataFrame({"input1": [15,41,44,47,50,53,56,59,99],
                      "input2": [29,41,44,47,50,53,56,59,66]})

my_df.plot(kind = "box", vert = False)

outlier_columns = ["input1", "input2"]


# Boxplot Approach

for column in outlier_columns:
    
    lower_quartile = my_df[column].quantile(0.25)
    upper_quartile = my_df[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * 1.5
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    
    outliers = my_df[(my_df[column] < min_border) | (my_df[column] > max_border)].index
    print(f"{len(outliers)} outliers detected in column {column}")
    
    my_df.drop(outliers, inplace = True)
    


# Standar Deviation Approach

my_df = pd.DataFrame({"input1": [15,41,44,47,50,53,56,59,99],
                      "input2": [29,41,44,47,50,53,56,59,66]})

for column in outlier_columns:
    
    mean = my_df[column].mean()
    std_dev = my_df[column].std()
    
    min_border = mean - std_dev * 3
    max_border = mean + std_dev * 3
    
    outliers = my_df[(my_df[column] < min_border) | (my_df[column] > max_border)].index
    print(f"{len(outliers)} outliers detected in column {column}")
    
    my_df.drop(outliers, inplace = True)
