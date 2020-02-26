---
layout:     post
title:      My First Post
date:       2020-02-25 11:21:29
summary:    This is an empty post to test my blog.
categories: jekyll mixyll
---

Customer Analytics
================
2020-02-25 11:21:29

### Customers Data segmentation using PCA and K-means

This is a quick project where I'll demonstrate the correct way to segment customers data for further analysis. The purpose of customers analytics is to provide new usefull insights.The necessary columns for any customers analysis are the customers personal informations -which can be surveyed through different method such as loyality subscriptions- and a detailed purchase history for each customer. I'll be using python for this analysis, let's start.

#### Here I will load the necessary packages and import the dataset

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
full = pd.read_csv('dataset.csv')
print(full.head())
```

    ##           ID  Sex  Marital status  ...  Income  Occupation  Settlement size
    ## 0  100000001    0               0  ...  124670           1                2
    ## 1  100000002    1               1  ...  150773           1                2
    ## 2  100000003    0               0  ...   89210           0                0
    ## 3  100000004    0               0  ...  171565           1                1
    ## 4  100000005    0               0  ...  149031           1                1
    ## 
    ## [5 rows x 8 columns]

``` python
print(full.columns)
```

    ## Index(['ID', 'Sex', 'Marital status', 'Age', 'Education', 'Income',
    ##        'Occupation', 'Settlement size'],
    ##       dtype='object')

``` python
df = full[['Sex', 'Marital status', 'Age', 'Education', 'Income','Occupation', 'Settlement size']]
```

#### This is the customers informations data set that contains basic informations... We'll plot the correlations between each column using a Heat Map. Heat Maps are a great way to visualize correlations using color coding.

    ## [Text(0, 0.5, 'Sex'), Text(0, 1.5, 'Marital status'), Text(0, 2.5, 'Age'), Text(0, 3.5, 'Education'), Text(0, 4.5, 'Income'), Text(0, 5.5, 'Occupation'), Text(0, 6.5, 'Settlement size')]

    ## [Text(0.5, 0, 'Sex'), Text(1.5, 0, 'Marital status'), Text(2.5, 0, 'Age'), Text(3.5, 0, 'Education'), Text(4.5, 0, 'Income'), Text(5.5, 0, 'Occupation'), Text(6.5, 0, 'Settlement size')]

<img src="/asas_files/figure-markdown_github/unnamed-chunk-2-1.png" width="672" angle=90 style="display: block; margin: auto;" /> This heatmap shows basic correlations informations: 1. Age and Education are positively correlated 2. Occupation and income are correlated 3.Women are slightly more educated then men

... non of these can be considered as an insight... it can be deducted based on intuition.

yet it's good to know that our data is based on real observations

### Visualizing Raw Data

We have 2000 data points, which we'll scatter acrros Age and Income.

``` python
plt.figure(figsize = (12, 9))
plt.scatter(df.iloc[:, 2], df.iloc[:, 4])
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Visualization of raw data')
```

<img src="/_posts/asas_files/figure-markdown_github/unnamed-chunk-3-1.png" width="1152" />

### Standardization

Standardizing data, so that all features have equal weight. This is important for modelling. Otherwise, in our case Income would be considered much more important than Education for Instance.

``` python
scaler = StandardScaler()
df_std = scaler.fit_transform(df)
```

### Hierarchical Clustering

``` python
hier_clust = linkage(df_std, method = 'ward')
plt.figure(figsize = (12,9))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Observations')
plt.ylabel('Distance')
dendrogram(hier_clust,
           truncate_mode = 'level', 
           p = 5, 
           show_leaf_counts = False, 
           no_labels = True)
```

    ## {'icoord': [[5.0, 5.0, 15.0, 15.0], [25.0, 25.0, 35.0, 35.0], [10.0, 10.0, 30.0, 30.0], [45.0, 45.0, 55.0, 55.0], [65.0, 65.0, 75.0, 75.0], [50.0, 50.0, 70.0, 70.0], [20.0, 20.0, 60.0, 60.0], [85.0, 85.0, 95.0, 95.0], [105.0, 105.0, 115.0, 115.0], [90.0, 90.0, 110.0, 110.0], [125.0, 125.0, 135.0, 135.0], [145.0, 145.0, 155.0, 155.0], [130.0, 130.0, 150.0, 150.0], [100.0, 100.0, 140.0, 140.0], [40.0, 40.0, 120.0, 120.0], [165.0, 165.0, 175.0, 175.0], [185.0, 185.0, 195.0, 195.0], [170.0, 170.0, 190.0, 190.0], [205.0, 205.0, 215.0, 215.0], [225.0, 225.0, 235.0, 235.0], [210.0, 210.0, 230.0, 230.0], [180.0, 180.0, 220.0, 220.0], [245.0, 245.0, 255.0, 255.0], [265.0, 265.0, 275.0, 275.0], [250.0, 250.0, 270.0, 270.0], [285.0, 285.0, 295.0, 295.0], [305.0, 305.0, 315.0, 315.0], [290.0, 290.0, 310.0, 310.0], [260.0, 260.0, 300.0, 300.0], [200.0, 200.0, 280.0, 280.0], [80.0, 80.0, 240.0, 240.0], [325.0, 325.0, 335.0, 335.0], [345.0, 345.0, 355.0, 355.0], [330.0, 330.0, 350.0, 350.0], [365.0, 365.0, 375.0, 375.0], [385.0, 385.0, 395.0, 395.0], [370.0, 370.0, 390.0, 390.0], [340.0, 340.0, 380.0, 380.0], [405.0, 405.0, 415.0, 415.0], [425.0, 425.0, 435.0, 435.0], [410.0, 410.0, 430.0, 430.0], [445.0, 445.0, 455.0, 455.0], [465.0, 465.0, 475.0, 475.0], [450.0, 450.0, 470.0, 470.0], [420.0, 420.0, 460.0, 460.0], [360.0, 360.0, 440.0, 440.0], [485.0, 485.0, 495.0, 495.0], [505.0, 505.0, 515.0, 515.0], [490.0, 490.0, 510.0, 510.0], [525.0, 525.0, 535.0, 535.0], [545.0, 545.0, 555.0, 555.0], [530.0, 530.0, 550.0, 550.0], [500.0, 500.0, 540.0, 540.0], [565.0, 565.0, 575.0, 575.0], [585.0, 585.0, 595.0, 595.0], [570.0, 570.0, 590.0, 590.0], [605.0, 605.0, 615.0, 615.0], [625.0, 625.0, 635.0, 635.0], [610.0, 610.0, 630.0, 630.0], [580.0, 580.0, 620.0, 620.0], [520.0, 520.0, 600.0, 600.0], [400.0, 400.0, 560.0, 560.0], [160.0, 160.0, 480.0, 480.0]], 'dcoord': [[0.0, 2.1223540150144173, 2.1223540150144173, 0.0], [0.0, 4.3155745482676915, 4.3155745482676915, 0.0], [2.1223540150144173, 5.589695377405012, 5.589695377405012, 4.3155745482676915], [0.0, 6.791198451937796, 6.791198451937796, 0.0], [0.0, 7.803972378606145, 7.803972378606145, 0.0], [6.791198451937796, 12.047948567816455, 12.047948567816455, 7.803972378606145], [5.589695377405012, 14.091240220841797, 14.091240220841797, 12.047948567816455], [0.0, 2.4758879971699788, 2.4758879971699788, 0.0], [0.0, 4.009067557513786, 4.009067557513786, 0.0], [2.4758879971699788, 5.927859602489558, 5.927859602489558, 4.009067557513786], [0.0, 7.846037798790616, 7.846037798790616, 0.0], [0.0, 11.1835606844946, 11.1835606844946, 0.0], [7.846037798790616, 12.048396735493654, 12.048396735493654, 11.1835606844946], [5.927859602489558, 21.733799440726337, 21.733799440726337, 12.048396735493654], [14.091240220841797, 30.442238972320503, 30.442238972320503, 21.733799440726337], [0.0, 5.630547575057578, 5.630547575057578, 0.0], [0.0, 12.516887065383933, 12.516887065383933, 0.0], [5.630547575057578, 15.016219162404724, 15.016219162404724, 12.516887065383933], [0.0, 5.448239753967315, 5.448239753967315, 0.0], [0.0, 5.671253618131798, 5.671253618131798, 0.0], [5.448239753967315, 18.86452742462555, 18.86452742462555, 5.671253618131798], [15.016219162404724, 27.52233083936995, 27.52233083936995, 18.86452742462555], [0.0, 5.414998042947461, 5.414998042947461, 0.0], [0.0, 10.236526006642633, 10.236526006642633, 0.0], [5.414998042947461, 13.39749201434294, 13.39749201434294, 10.236526006642633], [0.0, 4.277655804201464, 4.277655804201464, 0.0], [0.0, 8.472874000215867, 8.472874000215867, 0.0], [4.277655804201464, 16.554609278021523, 16.554609278021523, 8.472874000215867], [13.39749201434294, 31.588168944768274, 31.588168944768274, 16.554609278021523], [27.52233083936995, 39.12397586244344, 39.12397586244344, 31.588168944768274], [30.442238972320503, 56.73375166963763, 56.73375166963763, 39.12397586244344], [0.0, 3.4574207514663065, 3.4574207514663065, 0.0], [0.0, 3.9896328524137576, 3.9896328524137576, 0.0], [3.4574207514663065, 9.097400845886968, 9.097400845886968, 3.9896328524137576], [0.0, 6.884958800671696, 6.884958800671696, 0.0], [0.0, 7.077343782971088, 7.077343782971088, 0.0], [6.884958800671696, 13.706423543547206, 13.706423543547206, 7.077343782971088], [9.097400845886968, 17.550420382920624, 17.550420382920624, 13.706423543547206], [0.0, 6.543311758583991, 6.543311758583991, 0.0], [0.0, 9.045264300731061, 9.045264300731061, 0.0], [6.543311758583991, 11.18575572620523, 11.18575572620523, 9.045264300731061], [0.0, 7.714779507745513, 7.714779507745513, 0.0], [0.0, 9.075960069579184, 9.075960069579184, 0.0], [7.714779507745513, 13.669297772312323, 13.669297772312323, 9.075960069579184], [11.18575572620523, 20.569854345075264, 20.569854345075264, 13.669297772312323], [17.550420382920624, 25.719802545357478, 25.719802545357478, 20.569854345075264], [0.0, 2.43171745867978, 2.43171745867978, 0.0], [0.0, 5.128124118168716, 5.128124118168716, 0.0], [2.43171745867978, 8.45013452100761, 8.45013452100761, 5.128124118168716], [0.0, 5.326338957512825, 5.326338957512825, 0.0], [0.0, 8.413061089449444, 8.413061089449444, 0.0], [5.326338957512825, 14.04107999544911, 14.04107999544911, 8.413061089449444], [8.45013452100761, 16.760620468263898, 16.760620468263898, 14.04107999544911], [0.0, 4.532363113975979, 4.532363113975979, 0.0], [0.0, 9.83116762656914, 9.83116762656914, 0.0], [4.532363113975979, 12.292237821984994, 12.292237821984994, 9.83116762656914], [0.0, 6.729198790405168, 6.729198790405168, 0.0], [0.0, 11.565541418045406, 11.565541418045406, 0.0], [6.729198790405168, 18.204501006713418, 18.204501006713418, 11.565541418045406], [12.292237821984994, 24.406924804054864, 24.406924804054864, 18.204501006713418], [16.760620468263898, 32.36297994757603, 32.36297994757603, 24.406924804054864], [25.719802545357478, 63.06917547917128, 63.06917547917128, 32.36297994757603], [56.73375166963763, 77.34958547428063, 77.34958547428063, 63.06917547917128]], 'ivl': ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''], 'leaves': [3402, 3699, 3734, 3869, 3864, 3879, 3890, 3929, 3697, 3712, 3833, 3859, 3919, 3923, 3917, 3936, 3873, 3924, 3945, 3952, 3839, 3850, 3846, 3925, 3871, 3874, 3921, 3956, 3606, 3841, 3854, 3950, 3589, 3842, 3834, 3880, 3933, 3941, 3896, 3912, 3855, 3915, 3928, 3947, 3899, 3910, 3931, 3954, 3732, 3794, 3823, 3832, 3789, 3901, 3905, 3926, 3845, 3904, 3894, 3960, 3906, 3920, 3943, 3955], 'color_list': ['g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'b', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'b', 'b']}

``` python
plt.show()
```

<img src="/asas_files/figure-markdown_github/unnamed-chunk-5-1.png" width="1152" />

### K-means Clustering with PCA

``` python
pca = PCA()
pca.fit(df_std)
```

    ## PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
    ##     svd_solver='auto', tol=0.0, whiten=False)

``` python
plt.figure(figsize = (12,9))
plt.plot(range(1,8), pca.explained_variance_ratio_.cumsum(), marker = 'o', linestyle = '--')
plt.title('Explained Variance by Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()
```

<img src="/_posts/asas_files/figure-markdown_github/unnamed-chunk-6-1.png" width="1152" />

Generally, we want to keep around 80-90% of the explained variance, here I chose 3 components

``` python
pca = PCA(n_components = 3)
pca.fit(df_std)
```

    ## PCA(copy=True, iterated_power='auto', n_components=3, random_state=None,
    ##     svd_solver='auto', tol=0.0, whiten=False)

``` python
df_pca_comp = pd.DataFrame(data = pca.components_,
                           columns = df.columns.values,
                           index = ['Component 1', 'Component 2', 'Component 3'])
df_pca_comp
```

    ##                   Sex  Marital status  ...  Occupation  Settlement size
    ## Component 1 -0.314695       -0.191704  ...    0.492059         0.464789
    ## Component 2  0.458006        0.512635  ...    0.014658        -0.069632
    ## Component 3 -0.293013       -0.441977  ...   -0.395505        -0.295685
    ## 
    ## [3 rows x 7 columns]

### Heat Map for Principal Components against original features

``` python

sns.heatmap(df_pca_comp,
            vmin = -1, 
            vmax = 1,
            cmap = 'RdBu',
            annot = True)
plt.yticks([0, 1, 2], 
           ['Component 1', 'Component 2', 'Component 3'],
           rotation = 45,
           fontsize = 9)
```

    ## ([<matplotlib.axis.YTick object at 0x00000000651024E0>, <matplotlib.axis.YTick object at 0x00000000650E3358>, <matplotlib.axis.YTick object at 0x000000006644F3C8>], <a list of 3 Text yticklabel objects>)

``` python
plt.show()
```

<img src="/_posts/asas_files/figure-markdown_github/unnamed-chunk-8-1.png" width="1152" />

### Applying our PCA model to the dataset and choosing the correct number of kmean clusters

There are many methods for choosing the number of clusters, the quickest one is the "Elbow Methode".

``` python


pca.transform(df_std)
```

    ## array([[ 2.51474593,  0.83412239,  2.1748059 ],
    ##        [ 0.34493528,  0.59814564, -2.21160279],
    ##        [-0.65106267, -0.68009318,  2.2804186 ],
    ##        ...,
    ##        [-1.45229829, -2.23593665,  0.89657125],
    ##        [-2.24145254,  0.62710847, -0.53045631],
    ##        [-1.86688505, -2.45467234,  0.66262172]])

``` python
scores_pca = pca.transform(df_std)

wcss = []
for i in range(1,11):
    kmeans_pca = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans_pca.fit(scores_pca)
    wcss.append(kmeans_pca.inertia_)
    
```

    ## KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    ##        n_clusters=1, n_init=10, n_jobs=None, precompute_distances='auto',
    ##        random_state=42, tol=0.0001, verbose=0)
    ## KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    ##        n_clusters=2, n_init=10, n_jobs=None, precompute_distances='auto',
    ##        random_state=42, tol=0.0001, verbose=0)
    ## KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    ##        n_clusters=3, n_init=10, n_jobs=None, precompute_distances='auto',
    ##        random_state=42, tol=0.0001, verbose=0)
    ## KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    ##        n_clusters=4, n_init=10, n_jobs=None, precompute_distances='auto',
    ##        random_state=42, tol=0.0001, verbose=0)
    ## KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    ##        n_clusters=5, n_init=10, n_jobs=None, precompute_distances='auto',
    ##        random_state=42, tol=0.0001, verbose=0)
    ## KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    ##        n_clusters=6, n_init=10, n_jobs=None, precompute_distances='auto',
    ##        random_state=42, tol=0.0001, verbose=0)
    ## KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    ##        n_clusters=7, n_init=10, n_jobs=None, precompute_distances='auto',
    ##        random_state=42, tol=0.0001, verbose=0)
    ## KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    ##        n_clusters=8, n_init=10, n_jobs=None, precompute_distances='auto',
    ##        random_state=42, tol=0.0001, verbose=0)
    ## KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    ##        n_clusters=9, n_init=10, n_jobs=None, precompute_distances='auto',
    ##        random_state=42, tol=0.0001, verbose=0)
    ## KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    ##        n_clusters=10, n_init=10, n_jobs=None, precompute_distances='auto',
    ##        random_state=42, tol=0.0001, verbose=0)

``` python
plt.figure(figsize = (10,8))
plt.plot(range(1, 11), wcss, marker = 'o', linestyle = '--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('K-means with PCA Clustering')
plt.show()
```

<img src="/_posts/asas_files/figure-markdown_github/unnamed-chunk-9-1.png" width="960" /> 4 clusters seems to be the right number to choose

``` python
kmeans_pca = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
kmeans_pca.fit(scores_pca)
```

    ## KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    ##        n_clusters=4, n_init=10, n_jobs=None, precompute_distances='auto',
    ##        random_state=42, tol=0.0001, verbose=0)

``` python
df_segm_pca_kmeans = pd.concat([df.reset_index(drop = True), pd.DataFrame(scores_pca)], axis = 1)
df_segm_pca_kmeans.columns.values[-3: ] = ['Component 1', 'Component 2', 'Component 3']
```

The last column we add contains the pca k-means clustering labels.

``` python
df_segm_pca_kmeans['Segment K-means PCA'] = kmeans_pca.labels_

df_segm_pca_kmeans
```

    ##       Sex  Marital status  Age  ...  Component 2  Component 3  Segment K-means PCA
    ## 0       0               0   67  ...     0.834122     2.174806                    0
    ## 1       1               1   22  ...     0.598146    -2.211603                    2
    ## 2       0               0   49  ...    -0.680093     2.280419                    1
    ## 3       0               0   45  ...    -0.579927     0.730731                    3
    ## 4       0               0   53  ...    -0.440496     1.244909                    3
    ## ...   ...             ...  ...  ...          ...          ...                  ...
    ## 1995    1               0   47  ...     0.298330     1.438958                    1
    ## 1996    1               1   27  ...     0.794727    -1.079871                    2
    ## 1997    0               0   31  ...    -2.235937     0.896571                    1
    ## 1998    1               1   24  ...     0.627108    -0.530456                    2
    ## 1999    0               0   25  ...    -2.454672     0.662622                    1
    ## 
    ## [2000 rows x 11 columns]

``` python
df_segm_pca_kmeans_freq = df_segm_pca_kmeans.groupby(['Segment K-means PCA']).mean()
df_segm_pca_kmeans_freq
```

    ##                           Sex  Marital status  ...  Component 2  Component 3
    ## Segment K-means PCA                            ...                          
    ## 0                    0.503788        0.689394  ...     2.029427     0.841953
    ## 1                    0.305011        0.095861  ...    -0.904856     1.005493
    ## 2                    0.900576        0.963977  ...     0.705300    -0.776925
    ## 3                    0.027444        0.168096  ...    -1.046172    -0.248046
    ## 
    ## [4 rows x 10 columns]

``` python
df_segm_pca_kmeans_freq['N Obs'] = df_segm_pca_kmeans[['Segment K-means PCA','Sex']].groupby(['Segment K-means PCA']).count()
df_segm_pca_kmeans_freq['Prop Obs'] = df_segm_pca_kmeans_freq['N Obs'] / df_segm_pca_kmeans_freq['N Obs'].sum()
```

Cluster 0 seems to be mostly married females with middle range income and with basic education and age arround 28, we will call this cluster millenials.

Cluster 1 is mostly single middle aged men with low education and high income, we will call it Gentles.

Cluster 2 mostly single middle aged men with low education and low income, we will call it Unlucky.

Finally Cluster 3 mostly High educated with the highest income with age around 55, we will call it seniors,

``` python
df_segm_pca_kmeans_freq = df_segm_pca_kmeans_freq.rename({0:'Mellineals', 
                                                          1:'Gentles',
                                                          2:'Unlucky', 
                                                          3:'Seniors'})
df_segm_pca_kmeans_freq
```

    ##                           Sex  Marital status  ...  N Obs  Prop Obs
    ## Segment K-means PCA                            ...                 
    ## Mellineals           0.503788        0.689394  ...    264    0.1320
    ## Gentles              0.305011        0.095861  ...    459    0.2295
    ## Unlucky              0.900576        0.963977  ...    694    0.3470
    ## Seniors              0.027444        0.168096  ...    583    0.2915
    ## 
    ## [4 rows x 12 columns]

``` python
df_segm_pca_kmeans['Legend'] = df_segm_pca_kmeans['Segment K-means PCA'].map({0:'Nellineals', 
                                                          1:'Gentles',
                                                          2:'Unlucky', 
                                                          3:'Seniors'})
                                                          
x_axis = df_segm_pca_kmeans['Component 2']
y_axis = df_segm_pca_kmeans['Component 1']
plt.figure(figsize = (10, 8))
sns.scatterplot(x_axis, y_axis, hue = df_segm_pca_kmeans['Legend'], palette = ['g', 'r', 'c', 'm'])
plt.title('Clusters by PCA Components')
plt.show()
```

<img src="/_posts/asas_files/figure-markdown_github/unnamed-chunk-12-1.png" width="960" />

Final note
----------

The final clustered data set can be further analysed with descriptive statisticals methods, but I will stop here. I hope this mini project was informative and showed the correct way in clustering customers data.

### Fayez Alshehri

