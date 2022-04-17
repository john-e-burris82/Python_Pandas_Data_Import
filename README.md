# Importing Data in Python

<p>There are numerous different ways to import data into Python. Below are some of the most common. These include flatfiles such .txt and .csv. Other common data types that an analyst may run into include Excel sheets, MATLAB, SAS, and relational database data. More recently for me, I came across importing pickle data. For a brief intro, click here <https://wiki.python.org/moin/UsingPickle>. Below are the main modules that will be needed for this overview. I will caveat that I am not providing the data files to complete this; however, most can be found on Kaggle or GitHub.

# Required packages

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import scipy.io  
import h5py  
from sas7bdat import SAS7BDAT  
from sqlalchemy import create_engine  
import pickle  

# This tutorial will cover the following:
## Importing Files

Python has the ability to import files for reading and writing. The main concept to remember when importing files is wether or not you want to read or write to the file. This will change how you open the file. In order to never forget to close a file, a context manager can be utilized. You can take this one step further and build a function to read in files faster. Code examples to read and write to files will be provided except in a function format. Caution is needed, this code can permenately erase the data in the file you choose. To avoid that, I commented out the write example and context editor.

## Importing in Numpy

As most of use know, Numpy is a powerful number crunching module. For more on Numpy, if unfamiliar, visit <Numpy.org>. The following will look at importing flatfiles into Numpy. Make sure that you imported numpy as np. As .csv files are rather common, the code below will focus on those. As most data files have columns and rows, telling the import what seperates those is important. The most common seperators are ',' and tabs. These are called delimiters. Ensure you specify the correct delimetor during import. A comma will be indicated as delimiter=',' and a tab will be indicated as delimiter='\t'. The two most common ways to get data into a Numpay array are np.loadtxt() <https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html> and np.genfromtext() <https://numpy.org/doc/stable/reference/generated/numpy.genfromtxt.html>.

## Importing Flat Files with Pandas

Having originally worked with R, Pandas' dataframe is a known entity for me. Importing into a dataframe makes exploratory analysis and munging much simplier in my opinion. To demonstrate the task of importing flat files, I will utilize the famous 'Titanic' dataset, the 'Seaslug' dataset from a CSV, and 'Battledeath' from Excel. The same command pd.read_csv() can be utilized on .csv and .txt files; however, to import data from Excel pd.read_excel() is needed. 

## Importing SAS Data
    
Statistical Analysis System or SAS is one of the most popular systems for statistical analysis; therefore, coming across a SAS dataset is probable.

## Importing MATLAB

When importing MATLAB data, you are basically importing a dictionary. From there, convert the dictionary to DataFrame.

## Importing Pickle

Pickle files are Python's native file type. Pickled files are serialized or converted to bytestream. Everything I have come across indicates users should be weary of importing pickle files from an unknown source as it could contain malicious code. I currently do not have a pickle file, so any code is commented out.

## Importing HDF5 Data
    
HDF5 is new to me. It apparently is utilized to store gigabytes, even terabytes, of numerical data. Once HDF5 files are imported to Python, it will act as a dictionary. Like the basic importing a flat file, we will want to specify 'r' during the import to indicate that we only want to read the file.

## Importing Relational Database Data

Relation databases are rather common. Being able to connect to the database, query the data, and import into Python is a task worth knowing.
    
## Importing Flat Files From the Web

Importing files directly from a website is another common means to import files. We could just download the data to our local computer or cloud storage and import it there; but, why waste storage space. Plus, downloading directly from the source allows any updates to be ingested when the code is ran again.
    
## Data Utilized
    
All files were downloaded from DataCamp. These include:
    
battledeath.xlsx  
disarea.dta  
ja_data2.mat  
mnist_kaggle_some_row.csv  
notebook.ipynb  
sales.sas7bat  
seaslug.txt  
titanic_sub.csv  
