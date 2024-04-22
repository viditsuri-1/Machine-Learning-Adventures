#!/usr/bin/env python
# coding: utf-8

# # Task 7: AutoFeatureSelector Tool
# ## This task is to test your understanding of various Feature Selection methods outlined in the lecture and the ability to apply this knowledge in a real-world dataset to select best features and also to build an automated feature selection tool as your toolkit
# 
# ### Use your knowledge of different feature selector methods to build an Automatic Feature Selection tool
# - Pearson Correlation
# - Chi-Square
# - RFE
# - Embedded
# - Tree (Random Forest)
# - Tree (Light GBM)

# ### Dataset: FIFA 19 Player Skills
# #### Attributes: FIFA 2019 players attributes like Age, Nationality, Overall, Potential, Club, Value, Wage, Preferred Foot, International Reputation, Weak Foot, Skill Moves, Work Rate, Position, Jersey Number, Joined, Loaned From, Contract Valid Until, Height, Weight, LS, ST, RS, LW, LF, CF, RF, RW, LAM, CAM, RAM, LM, LCM, CM, RCM, RM, LWB, LDM, CDM, RDM, RWB, LB, LCB, CB, RCB, RB, Crossing, Finishing, Heading, Accuracy, ShortPassing, Volleys, Dribbling, Curve, FKAccuracy, LongPassing, BallControl, Acceleration, SprintSpeed, Agility, Reactions, Balance, ShotPower, Jumping, Stamina, Strength, LongShots, Aggression, Interceptions, Positioning, Vision, Penalties, Composure, Marking, StandingTackle, SlidingTackle, GKDiving, GKHandling, GKKicking, GKPositioning, GKReflexes, and Release Clause.

# In[1]:


#%matplotlib inline
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
from collections import Counter
import math
from scipy import stats


# In[3]:


player_df = pd.read_csv(r"C:\Users\vidit\Desktop\AI Development\ML 1\Task 7\fifa19.csv")


# In[4]:


numcols = ['Overall', 'Crossing','Finishing',  'ShortPassing',  'Dribbling','LongPassing', 'BallControl', 'Acceleration','SprintSpeed','Agility','Stamina','Volleys','FKAccuracy','Reactions','Balance','ShotPower','Strength','LongShots','Aggression','Interceptions']

# Removing nationality on request during lectures
catcols = ['Preferred Foot','Position','Body Type','Weak Foot'] # 'Nationality'


# In[5]:


player_df = player_df[numcols+catcols]


# In[6]:


traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])],axis=1)
features = traindf.columns

traindf = traindf.dropna()


# In[7]:


features


# In[8]:


traindf = pd.DataFrame(traindf,columns=features)


# In[9]:


y = traindf['Overall']>=87
X = traindf.copy()
del X['Overall']


# In[10]:


X.head()


# In[11]:


len(X.columns)


# ### Set some fixed set of features

# In[12]:


feature_name = list(X.columns)
# no of maximum features we need to select
num_feats=30


# ## Filter Feature Selection - Pearson Correlation

# ### Pearson Correlation function

# In[105]:


def cor_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    
    """
    Perform feature selection using Pearson correlation with the target variable.

    Parameters:
    - X: DataFrame of features
    - y: Series of target variable
    - num_feats: Number of features to select

    Returns:
    - cor_support: Boolean mask indicating selected features
    - cor_feature: List of selected feature names
    """
    
    # Concatenate features and target variable into a single DataFrame
    all_data = pd.concat([X, y], axis=1)
    
    # Calculate Pearson correlation coefficients
    cor_matrix = all_data.corr()
    
    # Calculate the absolute correlation coefficients with the target variable
    cor_with_target = cor_matrix.iloc[:-1, -1].abs()
    
    # Select the top 'num_feats' features with the highest correlation with the target variable
    cor_feature = cor_with_target.nlargest(num_feats).index.values
    cor_support = X.columns.isin(cor_feature)
    
    # Your code ends here
    return cor_support, cor_feature


# In[106]:


cor_support, cor_feature = cor_selector(X, y, num_feats)
print(str(len(cor_feature)), 'selected features')


# ### List the selected features from Pearson Correlation

# In[15]:


cor_feature


# ## Filter Feature Selection - Chi-Sqaure

# In[16]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler


# ### Chi-Squared Selector function

# In[17]:


def chi_squared_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    
    """
    Perform feature selection using the chi-square test after scaling features.

    Parameters:
    - X: DataFrame of features
    - y: Series of target variable (categorical)
    - num_feats: Number of features to select

    Returns:
    - chi_support: Boolean mask indicating selected features
    - chi_feature: List of selected feature names
    """
    
    # Scale the features using MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # SelectKBest with chi2 as the score function
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X_scaled, y)
    
    # Boolean mask indicating selected features
    chi_support = chi_selector.get_support()
    
    # List of selected feature names
    chi_feature = X.columns[chi_support]
    
    # Your code ends here
    return chi_support, chi_feature


# In[18]:


chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
print(str(len(chi_feature)), 'selected features')


# ### List the selected features from Chi-Square 

# In[19]:


chi_feature


# ## Wrapper Feature Selection - Recursive Feature Elimination

# In[20]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


# ### RFE Selector function

# In[21]:


def rfe_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    
    """
    Perform feature selection using Recursive Feature Elimination (RFE) with MinMaxScaler.

    Parameters:
    - X: DataFrame of features
    - y: Series of target variable
    - num_feats: Number of features to select

    Returns:
    - rfe_support: Boolean mask indicating selected features
    - rfe_feature: List of selected feature names
    """
    
    # Scale the features using MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Create an estimator (e.g., Logistic Regression)
    estimator = LogisticRegression()
    
    # RFE with the chosen estimator
    rfe_selector = RFE(estimator, n_features_to_select=num_feats)
    rfe_selector = rfe_selector.fit(X_scaled, y)
    
    # Boolean mask indicating selected features
    rfe_support = rfe_selector.support_
    
    # List of selected feature names
    rfe_feature = X.columns[rfe_support]
    
    # Your code ends here
    return rfe_support, rfe_feature


# In[22]:


rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
print(str(len(rfe_feature)), 'selected features')


# ### List the selected features from RFE

# In[23]:


rfe_feature


# ## Embedded Selection - Lasso: SelectFromModel

# In[24]:


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


# In[66]:


def embedded_log_reg_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    
    """
    Perform embedded feature selection using Logistic Regression with L1 regularization.

    Parameters:
    - X: DataFrame of features
    - y: Series of target variable
    - num_feats: Number of features to select

    Returns:
    - embedded_lr_support: Boolean mask indicating selected features
    - embedded_lr_feature: List of selected feature names
    """
    
    # Scale the features using MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Create Logistic Regression model with L1 regularization
    lasso_selector = LogisticRegression(penalty='l1', solver='liblinear', C=2.8)
    
    # SelectFromModel with the chosen estimator
    embedded_lr_selector = SelectFromModel(lasso_selector, max_features=num_feats)
    embedded_lr_selector.fit(X_scaled, y)
    
    # Boolean mask indicating selected features
    embedded_lr_support = embedded_lr_selector.get_support()
    
    # List of selected feature names
    embedded_lr_feature = X.columns[embedded_lr_support]
    
    # Your code ends here
    return embedded_lr_support, embedded_lr_feature


# In[67]:


embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
print(str(len(embedded_lr_feature)), 'selected features')


# In[68]:


embedded_lr_feature


# ## Tree based(Random Forest): SelectFromModel

# In[28]:


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


# In[73]:


def embedded_rf_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    
    """
    Perform embedded feature selection using Random Forest.

    Parameters:
    - X: DataFrame of features
    - y: Series of target variable
    - num_feats: Number of features to select

    Returns:
    - embedded_rf_support: Boolean mask indicating selected features
    - embedded_rf_feature: List of selected feature names
    """
    
    # Scale the features using MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Create Random Forest model
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
    
    rf_selector.fit(X_scaled, y)
    
    feature_importances = rf_selector.feature_importances_
    sorted_feature_indices = feature_importances.argsort()[::-1]
    selected_feature_indices = sorted_feature_indices[:num_feats]
    
    embedded_rf_support = np.zeros(X.shape[1], dtype=bool)
    embedded_rf_support[selected_feature_indices] = True
    
    embedded_rf_feature = X.columns[embedded_rf_support]
    
    return embedded_rf_support, embedded_rf_feature


# In[74]:


embedder_rf_support, embedder_rf_feature = embedded_rf_selector(X, y, num_feats)
print(str(len(embedder_rf_feature)), 'selected features')


# In[75]:


embedder_rf_feature


# ## Tree based(Light GBM): SelectFromModel

# In[32]:


#pip install lightgbm


# In[33]:


from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier


# In[81]:


def embedded_lgbm_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    
    """
    Perform embedded feature selection using LightGBM.

    Parameters:
    - X: DataFrame of features
    - y: Series of target variable
    - num_feats: Number of features to select

    Returns:
    - embedded_lgbm_support: Boolean mask indicating selected features
    - embedded_lgbm_feature: List of selected feature names
    """
    
    # Scale the features using MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Create LightGBM model
    lgbm_selector = LGBMClassifier(n_estimators=100, random_state=42)  # Adjust parameters as needed
    lgbm_selector.fit(X_scaled, y)
    
    feature_importances = lgbm_selector.feature_importances_
     
    sorted_feature_indices = np.argsort(feature_importances)[::-1]
    selected_feature_indices = sorted_feature_indices[:num_feats]
    
    # Boolean mask indicating selected features
    embedded_lgbm_support = np.zeros(X.shape[1], dtype=bool)
    embedded_lgbm_support[selected_feature_indices] = True
    
    # List of selected feature names
    embedded_lgbm_feature = X.columns[embedded_lgbm_support]
    
    # Your code ends here
    return embedded_lgbm_support, embedded_lgbm_feature


# In[82]:


embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
print(str(len(embedded_lgbm_feature)), 'selected features')


# In[83]:


embedded_lgbm_feature


# ## Putting all of it together: AutoFeatureSelector Tool

# In[111]:


pd.set_option('display.max_rows', None)
# put all selection together

feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embedded_lr_support,
                                    'Random Forest':embedder_rf_support, 'LightGBM':embedded_lgbm_support})

numeric_columns = ['Pearson', 'Chi-2', 'RFE', 'Logistics', 'Random Forest', 'LightGBM']
feature_selection_df[numeric_columns] = feature_selection_df[numeric_columns].astype(int)

feature_selection_df['Total'] = np.sum(feature_selection_df[numeric_columns], axis=1)

# Convert the 'Total' column to string before concatenating
feature_selection_df['Total'] = feature_selection_df['Total'].astype(str)

feature_selection_df['Display'] = feature_selection_df['Feature'] + ' (' + feature_selection_df['Total'] + ')'


feature_selection_df = feature_selection_df.sort_values(['Total', 'Feature'], ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
feature_selection_df.head(num_feats)[['Display']]


# ## Can you build a Python script that takes dataset and a list of different feature selection methods that you want to try and output the best (maximum votes) features from all methods?

# In[112]:


def preprocess_dataset(dataset_path):
    """
    Preprocess the dataset for a machine learning task.

    Parameters:
    - dataset_path: Path to the CSV file containing the dataset.

    Returns:
    - X: DataFrame of features after preprocessing.
    - y: Series representing the target variable after preprocessing.
    - num_feats: Number of features to consider.
    """

    # Read the dataset from the specified path
    player_df = pd.read_csv(dataset_path)

    # Define numerical and categorical columns of interest
    numcols = ['Overall', 'Crossing', 'Finishing', 'ShortPassing', 'Dribbling', 'LongPassing', 'BallControl',
               'Acceleration', 'SprintSpeed', 'Agility', 'Stamina', 'Volleys', 'FKAccuracy', 'Reactions', 'Balance',
               'ShotPower', 'Strength', 'LongShots', 'Aggression', 'Interceptions']
    catcols = ['Preferred Foot', 'Position', 'Body Type', 'Weak Foot']

    # Select relevant columns from the dataset
    player_df = player_df[numcols + catcols]

    # Create a new DataFrame with numerical columns and one-hot encoded categorical columns
    traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])], axis=1)

    # Extract feature names
    features = traindf.columns

    # Drop rows with missing values
    traindf = traindf.dropna()

    # Reconstruct DataFrame with cleaned data and specified column order
    traindf = pd.DataFrame(traindf, columns=features)

    # Define the target variable 'y' based on a condition (Overall >= 87)
    y = traindf['Overall'] >= 87

    # Define the feature matrix 'X' by excluding the target variable
    X = traindf.copy()
    del X['Overall']

    # Specify the desired number of features to consider
    num_feats = 30

    return X, y, num_feats


# In[118]:


def autoFeatureSelector(dataset_path, methods=[]):
    # Parameters
    # data - dataset to be analyzed (csv file)
    # methods - various feature selection methods we outlined before, use them all here (list)
    
    # preprocessing
    X, y, num_feats = preprocess_dataset(dataset_path)
    
    # Run every method we outlined above from the methods list and collect returned best features from every method
    if 'pearson' in methods:
        cor_support, cor_feature = cor_selector(X, y,num_feats)
    if 'chi-square' in methods:
        chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
    if 'rfe' in methods:
        rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
    if 'log-reg' in methods:
        embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
    if 'rf' in methods:
        embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
    if 'lgbm' in methods:
        embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
    
    # Combine all the above feature list and count the maximum set of features that got selected by all methods
    #### Your Code starts here (Multiple lines)
    
    feature_dict = {}
    
    if 'pearson' in methods:
        feature_dict['pearson'] = cor_feature
    if 'chi-square' in methods:
        feature_dict['chi-square'] = chi_feature
    if 'rfe' in methods:
        feature_dict['rfe'] = rfe_feature
    if 'log-reg' in methods:
        feature_dict['log-reg'] = embedded_lr_feature
    if 'rf' in methods:
        feature_dict['rf'] = embedded_rf_feature
    if 'lgbm' in methods:
        feature_dict['lgbm'] = embedded_lgbm_feature
    
    best_features = set.intersection(*map(set, feature_dict.values()))
    
    #### Your Code ends here
    return best_features


# In[119]:


best_features = autoFeatureSelector(dataset_path="./fifa19.csv", methods=['pearson', 'chi-square', 'rfe', 'log-reg', 'rf', 'lgbm'])
best_features


# ### Last, Can you turn this notebook into a python script, run it and submit the python (.py) file that takes dataset and list of methods as inputs and outputs the best features

# In[125]:


import argparse
def preprocess_dataset(dataset_path):
    """
    Preprocess the dataset for a machine learning task.

    Parameters:
    - dataset_path: Path to the CSV file containing the dataset.

    Returns:
    - X: DataFrame of features after preprocessing.
    - y: Series representing the target variable after preprocessing.
    - num_feats: Number of features to consider.
    """

    # Read the dataset from the specified path
    player_df = pd.read_csv(dataset_path)

    # Define numerical and categorical columns of interest
    numcols = ['Overall', 'Crossing', 'Finishing', 'ShortPassing', 'Dribbling', 'LongPassing', 'BallControl',
               'Acceleration', 'SprintSpeed', 'Agility', 'Stamina', 'Volleys', 'FKAccuracy', 'Reactions', 'Balance',
               'ShotPower', 'Strength', 'LongShots', 'Aggression', 'Interceptions']
    catcols = ['Preferred Foot', 'Position', 'Body Type', 'Weak Foot']

    # Select relevant columns from the dataset
    player_df = player_df[numcols + catcols]

    # Create a new DataFrame with numerical columns and one-hot encoded categorical columns
    traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])], axis=1)

    # Extract feature names
    features = traindf.columns

    # Drop rows with missing values
    traindf = traindf.dropna()

    # Reconstruct DataFrame with cleaned data and specified column order
    traindf = pd.DataFrame(traindf, columns=features)

    # Define the target variable 'y' based on a condition (Overall >= 87)
    y = traindf['Overall'] >= 87

    # Define the feature matrix 'X' by excluding the target variable
    X = traindf.copy()
    del X['Overall']

    # Specify the desired number of features to consider
    num_feats = 30

    return X, y, num_feats


    # Define the feature matrix 'X' by excluding the target variable
    X = traindf.copy()
    del X['Overall']

    # Specify the desired number of features to consider
    num_feats = 30

    return X, y, num_feats


def autoFeatureSelector(dataset_path, methods=[]):
    # Parameters
    # data - dataset to be analyzed (csv file)
    # methods - various feature selection methods we outlined before, use them all here (list)
    
    # preprocessing
    X, y, num_feats = preprocess_dataset(dataset_path)
    
    # Run every method we outlined above from the methods list and collect returned best features from every method
    if 'pearson' in methods:
        cor_support, cor_feature = cor_selector(X, y,num_feats)
    if 'chi-square' in methods:
        chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
    if 'rfe' in methods:
        rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
    if 'log-reg' in methods:
        embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
    if 'rf' in methods:
        embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
    if 'lgbm' in methods:
        embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
    
    # Combine all the above feature list and count the maximum set of features that got selected by all methods
    #### Your Code starts here (Multiple lines)
    
    feature_dict = {}
    
    if 'pearson' in methods:
        feature_dict['pearson'] = cor_feature
    if 'chi-square' in methods:
        feature_dict['chi-square'] = chi_feature
    if 'rfe' in methods:
        feature_dict['rfe'] = rfe_feature
    if 'log-reg' in methods:
        feature_dict['log-reg'] = embedded_lr_feature
    if 'rf' in methods:
        feature_dict['rf'] = embedded_rf_feature
    if 'lgbm' in methods:
        feature_dict['lgbm'] = embedded_lgbm_feature
    
    best_features = set.intersection(*map(set, feature_dict.values()))
    
    #### Your Code ends here
    return best_features


def main():
    dataset_path = input("Enter the path to the fifa dataset (CSV file): ")
    methods_str = input("Enter a space-separated list of feature selection methods: ")
    methods = methods_str.split()

    if not dataset_path or not methods:
        print("Please provide both dataset path and feature selection methods.")
        return

    best_features = autoFeatureSelector(dataset_path=dataset_path, methods=methods)
    print("Best Features:", best_features)

if __name__ == "__main__":
    main()


# In[ ]:




