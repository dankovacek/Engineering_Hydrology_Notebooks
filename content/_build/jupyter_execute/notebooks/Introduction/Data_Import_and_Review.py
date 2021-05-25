# Example: Data Import and Review

When starting out learning to code in Python, there may be a few simple tasks that seem unneccessarily complex.  Importing data is one such task, where in Excel we click on a file to open it and most of the time the program already knows whether a cell contains a number or text.  In Python we don't have to declare the type of data explicitly, but the example below demonstrates the process of importing a file of time series data where the raw data format requires an extra step before we can work with the data.

There is a limit to the size of dataset we can manually review for anomalies.  As data gets bigger, it becomes necessary to evaluate our dataset in ways that don't require scanning tens or hundreds of thousands of rows.  In this example, we import a comma-separated-value (csv) file from a measurement device.  This notebook provides an introduction to opening and reviewing files programmatically.  

One final note, the '#' symbol used extensively below is used for adding comments that the code interpreter ignores.  The interpreter ignores all lines starting with '#', as well as anything after a '#' symbol on a line that doesn't start with '#'.

# import libraries
# pandas is a library that manipulates "dataframes", table-like structures,
# to efficiently run operations on large datasets
import pandas as pd
# numpy is a fundamental library for scientific computing
# based on the representation of data as arrays
import numpy as np
# matplotlib is a plotting library
from matplotlib import pyplot as plt

# this is a command to tell the notebook to display figures
# within the notebook instead of creating new windows.
%matplotlib inline

## Import Data

Now we want to import data from a folder.  Another common difficulty is in recognizing how the program interprets file paths.  If we want to open a data file from within this notebook, it is important to know that Jupyter interprets file paths **relative** to the location of the Jupyter notebook file.

In this case, the file we are working with is "Data_Import_and_Review.ipynb", and it's saved under "*(something)*/Engineering_Hydrology_Notebooks/Introduction/Data_Import_and_Review.ipynb", where *(something)* is wherever you saved and unpacked the main repository file on your computer.  

If we had a datafile (let's call it *CIVL418_Hidden_Creek_H_data.csv*) in the same folder as our notebook file, we can simply call the filename.  With pandas, it would be like so:

`df = pd.read_csv('CIVL418_Hidden_Creek_H_data.csv')`

Looking at our file structure, there is a folder called *data_review* that contains a number of csv files.  If we try and import one of these files using strictly the filename, we'll see an error like:

```
FileNotFoundError                         Traceback (most recent call last)
<ipython-input-5-b872b6bb0ac8> in <module>
      3 # in this case  is located in the
      4 # Open the folder and navigate to where
----> 5 df = pd.read_csv('CIVL418_Hidden_Creek_H_data.csv')
```

This error occurs because Jupyter looks for the file in the same level directory as our file.  We can tell it to look up one level in the file directory by starting the path string with double ellipsis, `..`.   

If we try the script below, our notebook will look for *CIVL418_Hidden_Creek_H_data.csv* one level up from the *Introduction* folder, which puts us at the level of `Engineering_Hydrology_Notebooks/`.  Before moving on, ask yourself if this path will work?  

`df = pd.read_csv('../CIVL418_Hidden_Creek_H_data.csv')`

The path won't work because there is no file named *CIVL418_Hidden_Creek_H_data.csv* found in the folder `Engineering_Hydrology_Notebooks/`.  The file is saved in the `data_review` folder that is however in this folder, so we construct the path as follows:

`../data_review/CIVL418_Hidden_Creek_H_data.csv`

Let's import our data now.

# import data from the 
# note the file path is RELATIVE to the location of the notebook file we're working in
df = pd.read_csv('../../Project_Data/Hidden_Creek_stage_data.csv')
df.describe(include='all')

Looking at the stats of the above dataframe, it appears as though the 'Value' column has non-numeric values.  We can tell this by trying to perform a calculation on the column.  If calculating the mean doesn't work, there could be a few explanations.

df['Value'].mean()

We can see the mean calculation throws a big, ugly error:

    ValueError: could not convert ... to numeric:

When you import a csv with pandas, it will try to figure out the type of each column.  If pandas can't figure it out, it will leave the values as strings.  In this case, and this is very common when working with files coming from different places (other software, hardware systems, etc.), some values which are non-numeric are included somewhere in the file.  We don't want to go through a big csv and manually change the files, because what if we have to deal with hundreds of files, or millions of rows?

We can verify the type of values in the column by calling type, (and we need to check one row, so I've indexed row 0 below).

print(type(df['Value'][0]))

Indeed, the data type of the elements of the `Value` column are string.

We can change the type of values in a few ways.  Let's try using the `.astype()` function from pandas.

df['Value'] = df['Value'].astype(float)
print(type(df['Value'][0]))

We can see the type is now float.  Let's try calculating a mean again.  

df['Value'].mean()

Now if we plot the value, we can see there are a couple of breaks in the series.

df.plot('Date', 'Value')

We can also see what periods correspond to gaps by filtering for all rows the `nan` values.

gaps = df[df['Value'].isnull()]

gaps

If we know we'll be dealing with more files from this source in the future, we could begin by addressing it at the source (change how values are recorded and saved).  

print(len(gaps))

The `len` function tells us how many non-numeric values (`NaN`) there are in our dataset by checking the length of the filtered dataframe called `gaps` .  


df.mean()

The mean function will skip over the non-numeric values and not include them in the calculation of summary statistics.

This may seem like a lot of work for not much value, but you quickly become proficient at basic data review using programmatic methods over manually (visually) walking through the entire dataset.  This skill is especially important as datasets grow in size.

