# -TLAB-Music-Recommendation-Algorithm

Within this project, we will take a look at a real dataset of songs from 1950 to 2011. 

This dataset contains a mix of lyrical and continuous variables pulled from a 2020 research paper titled Music Dataset: Lyrics and Metadata from 1950 to 2019. 

Your mission is to build, from the ground up, an unsupervised recommendation engine:

Data: an unlabeled music dataset of lyrical and metadata features
Approach: use scikit-learn to preprocess, cluster, and evaluate song groupings
Goal: derive thematic clusters that power personalized song recommendations for users based on their listening history

01_Initial EDA
Perform univariate, bivariate, and multivariate exploratory analyses.
Create visualizations to uncover patterns and formulate hypotheses—no target variable, just exploring feature interactions.

02_Data Cleaning & Dimensionality Reduction
Clean and wrangle the raw data (drop nulls, fix formats, remove outliers).
Drop irrelevant or highly correlated columns (e.g., the full lyrics text).
Scale and reduce dimensionality (e.g., via PCA).
Save the processed dataset as a new CSV for modeling.

03_Modeling & Hyperparameter Search
Train an unsupervised clustering model (e.g., K-Means, hierarchical).
Use techniques like the Elbow Method or Silhouette Analysis to pick the optimal number of clusters.
Assign each training sample a cluster label and save that labeled DataFrame.

04_New Sample Prediction
Load the test dataset and apply the same preprocessing and clustering pipeline.
Generate cluster labels for each test sample.
Save the test set with its new cluster column for downstream analysis.