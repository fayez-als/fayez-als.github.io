---
layout:     post
title:      So, What is Jekyllolo?
date:       2020-06-09 12:32:18
summary:    test test test
---

my first post
================

### Customers Data segmentation using PCA and K-means

This is a quick project where I'll demonstrate the correct way to segment customers data for further analysis

The purpose of customers analytics is to provide new usefull insights, lets see how we can approach this task using python.

``` python
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
print("hello")
```

    ## hello

``` python
df = pd.read_csv('dataset.csv')
print(df.head())
```

    ##           ID  Sex  Marital status  ...  Income  Occupation  Settlement size
    ## 0  100000001    0               0  ...  124670           1                2
    ## 1  100000002    1               1  ...  150773           1                2
    ## 2  100000003    0               0  ...   89210           0                0
    ## 3  100000004    0               0  ...  171565           1                1
    ## 4  100000005    0               0  ...  149031           1                1
    ## 
    ## [5 rows x 8 columns]

First we load and examine the dataset
-------------------------------------

``` python

print('hello world')
```

    ## hello world
