# Data Import and Review

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
df = pd.read_csv('../data_review/CIVL418_Hidden_Creek_H_data.csv')
df.describe(include='all')

Looking at the stats of the above dataframe, it appears as though the 'Value' column has non-numeric values.  We can tell this by trying to perform a calculation on the column.  If calculating the mean doesn't work, there could be a few explanations.

df['Value'].mean()

We can see the mean calculation throws an informative error:

    ValueError: could not convert string to float:

When you import a csv with pandas, it will try to figure out the type of each column.  If pandas can't figure it out, it will leave the values as strings.  In this case, and this is very common when working with files coming from different places (other software, hardware systems, etc.), some values which are non-numeric are included somewhere in the file.  We don't want to go through a big csv and manually change the files, because what if we have to deal with hundreds of files, or millions of rows?

We can verify the type of values in the column by calling type, (and we need to check one row, so I've indexed row 0 below).  Indeed, the type is string.

print(type(df['Value'][0]))

We can change the type of values in a few ways.  Let's try using the `.astype()` function from pandas.

df['Value'] = df['Value'].astype(float)
print(type(df['Value'][0]))

We can see the type is now float.  Let's try calculating a mean again.  

df['Value'].mean()

Now if we plot the value, we can see there are a couple of breaks in the series.

df.plot('Date', 'Value')

We can also see what periods correspond to gaps by filtering for the `nan` values.

gaps = df[df['Value'].isnull()]

gaps

What else can we do to avoid all this work?

Now that we know how the `nan` values are represented in the file, if we know we'll be dealing with more files from this source in the future, we can either address it at the source (change how values are recorded and saved).  Usually we don't have control over this, but we can address it at the point of import.

df = pd.read_csv('data_review/CIVL418_2019_Hidden_Creek_H_data.csv', na_values=['NaN'])
df['Value'].mean()

Hmmm.  That didn't work.  Why not?  Let's try and figure it out.

df[df['Value'] == 'NaN']

Why didn't that work?  We can see above in the error there is `NaN` before our very eyes...  Is it because Excel hates us and wants to fill us with the rage of a thousand suns?  The answer is yes.  

This is where the creative aspect of looking at data comes in. 

Let's get all the `NaN` values in an array.

all_values = df['Value'].to_numpy()
# show a sample of the first ten values
all_values[:10]

Here we can see that not only are the values strings, but in a few cases, we can even see there are spaces included!  How annoying!

Let's check the entire list for any strings containing `NaN`.


# start off with an empty array.  This is where we'll store any strings that contain `NaN`
nan_values = []
# iterate through each of the values
for value in all_values:
    # check if the string contains the substring `Nan`
    # if so, append it to the `nan_values` array.
    if 'NaN' in value:
        nan_values.append(value)
           

The `len` function tells us how many values have `NaN` in the string.  The `set` function reduces the `nan_values` array to eliminate all duplicates -- i.e. `set` returns the *unique* values in an array.

print(len(nan_values))
set(nan_values)

Aha!!  There are a bunch of blank spaces in the 'NaN' string!  We can now use this at the point of import.

nan_string = '  NaN'
new_df = pd.read_csv('data_review/CIVL418_2019_Hidden_Creek_H_data.csv', na_values=[nan_string])

new_df.mean()

Hooray, it worked!

That seemed like a lot of work, but when you see it a few times, you quickly become proficient at basic data review using programmatic methods rather than manually (visually) walking through the entire dataset.

new_df.plot('Date', 'Value')

