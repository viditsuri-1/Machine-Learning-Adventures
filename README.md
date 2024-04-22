**Machine-Learning-Adventures**

This repository contains various tasks and personal projects that I completed while practicing ML.

**1. Web Scraping:**

   This task helped me understand web scraping using the BeautifulSoup library.
   
 _**Key Info:**_
 
     It contains a python script to scrape an article and store the extracted text in a file.
     
     Url of the article scraped: https://medium.com/@subashgandyer/papa-what-is-a-neural-network-c5e5cc427c7
     
**2. Practicing Pandas:**

   Understanding the usefulness of pandas in analysis and visualization of data.


**3. Real Estate Housing Prediction:**

   Using scikit-library and understanding its application of building a simple Linear Regression model on a real estate dataset.

**4. Auto Feature Selector:**

   Creating an automated feature selection tool with various methods.

**5. Data Imputation (Imputters):**

   Using different data imputation stratergies on different algorithms.

**6. Best Random Forest:**

   Tasks Completed:

   Build a RandomForest for the above dataset (not one but many with different sets of parameters)
   
   Explore RandomizedSearchCV in Scikit-learn documentation
   
   Create a parameter grid with these values
   
   n_estimators : between 10 and 200
   
   max_depth : choose between 3 and 20
   
   max_features : ['auto', 'sqrt', None] + list(np.arange(0.5, 1, 0.1))
   
   max_leaf_nodes : choose between 10 to 50
   
   min_samples_split : choose between 2, 5, or 10
   
   bootstrap : choose between True or False
   
   Create the estimator (RandomForestClassifier)
   
   Create the RandomizedSearchCV with estimator, parameter grid, scoring on roc auc, n_iter = 10, random_state=RSEED(50) for same reproducible results
   
   Fit the model
   
   Explore the best model parameters
   
   Use the best model parameters to predict
   
   Plot the best model ROC AUC Curve
   
   Plot the Confusion Matrix
   
   Write any insights or observations you found in the last

   
