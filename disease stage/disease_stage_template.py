import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split, ShuffleSplit, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix



# Load the dataset into a pandas DataFrame
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


train = train_df.iloc[: ,1:]
test = test_df.iloc[: ,1:]


# Check for missing values
print(train.isna().sum())
print('------------------')
print(test.isna().sum())


#Created a boxplot for each numerical column to check for outliers
fig, ax = plt.subplots(figsize=(18,10))
sns.boxplot(data=train, ax=ax)
plt.show()

print(train['Stage'].value_counts())


# One Hot encoding manually in python to showcase the understanding

# Convert binary variables to numerical values
binary_vars = ["Ascites", "Hepatomegaly", "Spiders", "Edema"]
train[binary_vars] = train[binary_vars].replace({"N": 0, "Y": 1})

# Convert ordinal variable "Edema" to numerical values
edema_map = {"N": 0, "S": 1, "Y": 2}
train["Edema"] = train["Edema"].replace(edema_map)

# Convert "Status" to numerical values
status_map = {"C": 0, "CL": 1, "D": 2}
train["Status"] = train["Status"].replace(status_map)

# Convert "Drug" to numerical values
drug_map = {"Placebo": 0, "D-penicillamine": 1}
train["Drug"] = train["Drug"].replace(drug_map)

# Convert "Sex" to numerical values
sex_map = {"F": 0, "M": 1}
train["Sex"] = train["Sex"].replace(sex_map)



# Convert binary variables to numerical values
binary_vars = ["Ascites", "Hepatomegaly", "Spiders", "Edema"]
test[binary_vars] = test[binary_vars].replace({"N": 0, "Y": 1})

# Convert ordinal variable "Edema" to numerical values
edema_map = {"N": 0, "S": 1, "Y": 2}
test["Edema"] = test["Edema"].replace(edema_map)

# Convert "Status" to numerical values
status_map = {"C": 0, "CL": 1, "D": 2}
test["Status"] = test["Status"].replace(status_map)

# Convert "Drug" to numerical values
drug_map = {"Placebo": 0, "D-penicillamine": 1}
test["Drug"] = test["Drug"].replace(drug_map)

# Convert "Sex" to numerical values
sex_map = {"F": 0, "M": 1}
test["Sex"] = test["Sex"].replace(sex_map)



# Fill missing values with the median for the respective feature
train.fillna(train.median(), inplace=True)
test.fillna(test.median(), inplace=True)


# Calculate the Z-scores for all the columns
z_scores = np.abs(stats.zscore(train))
# Set a threshold for the Z-score to identify outliers
threshold = 3.5
# Get the indices of the rows that contain outliers
outliers_indices = np.where(np.any(z_scores > threshold, axis=1))
# Remove the outliers from the dataset
train_clean = train.drop(outliers_indices[0], axis=0).reset_index(drop=True)


print(f'Shape after cleaning: {train_clean.shape}')


# Check for missing values after filling missing values 
# Check for distrubution of the "Stage" after cleaning
print(train_clean.isna().sum())
print('------------------')
print(test.isna().sum())
print(train_clean['Stage'].value_counts())



#Created a boxplot for each numerical column to check for outliers
fig, ax = plt.subplots(figsize=(18,10))
sns.boxplot(data=train_clean, ax=ax)
plt.show()



#Defining the target value for the modelling
X = train_clean.drop('Stage', axis=1)
y = train_clean['Stage']



# Split the data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)


print(y_train.value_counts())
print("\n")
print(y_test.value_counts())



## DATA PIPELINE WITH KERNEL ##

# Define the data preprocessing steps as a pipeline
preprocessing_pipeline = Pipeline([
    ('scaling', StandardScaler()),
    ('lda', LinearDiscriminantAnalysis())
])


# Combine the preprocessing pipeline and the logistic regression classifier into a single pipeline
pipeline = Pipeline([
    ('preprocessing', preprocessing_pipeline),
    ('classifier', SVC())
])

# Defining the hyperparameters
hyperparameters = {
    'classifier__C': [00.1, 0.01, 0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf', 'poly'],
    'classifier__gamma': ['scale', 'auto', 00.1, 0.1, 1, 10],
    'classifier__class_weight': ['balanced'],
    'preprocessing__lda__n_components': [3]
}


# Define stratified k-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# Perform Grid search with cross-validation
random_search_svm = RandomizedSearchCV(pipeline, hyperparameters, 
                                   cv=cv,
                                   n_iter=20,
                                   random_state=42, 
                                   return_train_score=True,
                                   error_score='raise')

random_search_svm.fit(X_train, y_train)

print("Best hyperparameters:", random_search_svm.best_params_)

y_pred = random_search_svm.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# Get the training and test scores
train_scores = random_search_svm.cv_results_['mean_train_score']
test_scores = random_search_svm.cv_results_['mean_test_score']


# Plot the training and test scores
plt.figure(figsize=(10, 5))

plt.plot(train_scores, color = 'blue', marker='o', markersize=5 ,label='Training Accuracy')
plt.plot(test_scores, color = 'green', linestyle='--', marker='s', markersize=5, label='Validation Accuracy')

plt.fill_between(np.arange(len(train_scores)),
                 train_scores - random_search_svm.cv_results_['std_train_score'],
                 train_scores + random_search_svm.cv_results_['std_train_score'],
                 alpha=0.15, color='blue')

plt.fill_between(np.arange(len(test_scores)),
                 test_scores - random_search_svm.cv_results_['std_test_score'],
                 test_scores + random_search_svm.cv_results_['std_test_score'],
                 alpha=0.15, color='green')

plt.xlabel('Iteration')
plt.xticks(range(0, 21, 2))
plt.ylabel('Accuracy')
plt.title('Support Vector Machine Visualization')
plt.legend()
plt.show


# Apply the best parameters for prediction
y_pred_test = random_search_svm.best_estimator_.predict(test)



## DATA PIPELINE WITH REGULARIZATION ##

# Define the data preprocessing steps as a pipeline
preprocessing_pipeline = Pipeline([
    ('scaling', StandardScaler()),
    ('pca', PCA()),
    ('rfe', RFE(LogisticRegression()))
])

# Combine the preprocessing pipeline and the logistic regression classifier into a single pipeline
pipeline = Pipeline([
    ('preprocessing', preprocessing_pipeline),
    ('classifier', LogisticRegression())
])

# Defining the hyperparameters
hyperparameters = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2'],
    'preprocessing__pca__n_components': [2, 4, 6, 8, 10, 12, 14],
    'classifier__solver': ['liblinear', 'saga'],
    'classifier__max_iter': [5000],
    'preprocessing__rfe__n_features_to_select': [2, 4, 6, 8, 10, 12, 14]
}

# Define stratified k-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform Grid search with cross-validation
random_search_lr = RandomizedSearchCV(pipeline, hyperparameters, 
                                      cv=cv,
                                      n_iter=20,
                                      random_state=42, 
                                      return_train_score=True,
                                      error_score='raise')

random_search_lr.fit(X_train, y_train)

print("Best hyperparameters:", random_search_lr.best_params_)

y_pred = random_search_lr.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=1))
print(confusion_matrix(y_test, y_pred))



# Get the training and test scores
train_scores = random_search_lr.cv_results_['mean_train_score']
test_scores = random_search_lr.cv_results_['mean_test_score']


# Plot the training and test scores
plt.figure(figsize=(10, 5))
plt.plot(train_scores, color = 'blue', marker='o', markersize=5 ,label='Training Accuracy')
plt.plot(test_scores, color = 'green', linestyle='--', marker='s', markersize=5, label='Validation Accuracy')

plt.fill_between(np.arange(len(train_scores)),
                 train_scores - random_search_lr.cv_results_['std_train_score'],
                 train_scores + random_search_lr.cv_results_['std_train_score'],
                 alpha=0.15, color='blue')

plt.fill_between(np.arange(len(test_scores)),
                 test_scores - random_search_lr.cv_results_['std_test_score'],
                 test_scores + random_search_lr.cv_results_['std_test_score'],
                 alpha=0.15, color='green')


plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Logistic Regression Visualization')
plt.legend()
plt.show


# Apply the best parameters for prediction
y_pred_test = random_search_lr.best_estimator_.predict(test)


## OTHER MODELS ##

# Define the data preprocessing steps as a pipeline
preprocessing_pipeline = Pipeline([
    ('scaling', StandardScaler()),
    ('lda', LinearDiscriminantAnalysis())
])
     
# Combine the preprocessing pipeline and the KNN classifier into a single pipeline
pipeline = Pipeline([
    ('preprocessing', preprocessing_pipeline),
    ('classifier', KNeighborsClassifier())
])


# Defining the hyperparameters
hyperparameters = {
    'classifier__n_neighbors': list(range(1, 31)),
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
    'preprocessing__lda__n_components': [3],
    'classifier__algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
    'classifier__leaf_size': [10, 20, 30, 40, 50],
    'classifier__p': [1, 2, 3, 4, 5, 6]
}



# Define stratified k-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# Perform Grid search with cross-validation
random_search_knn = RandomizedSearchCV(pipeline, hyperparameters, 
                                       cv=cv,
                                       n_iter=20,
                                       random_state=42, 
                                       return_train_score=True,
                                       error_score='raise')

random_search_knn.fit(X_train, y_train)

print("Best hyperparameters:", random_search_knn.best_params_)

y_pred = random_search_knn.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Get the training and test scores
train_scores = random_search_knn.cv_results_['mean_train_score']
test_scores = random_search_knn.cv_results_['mean_test_score']


# Plot the training and test scores
plt.figure(figsize=(10, 5))
plt.plot(train_scores, color = 'blue', marker='o', markersize=5 ,label='Training Accuracy')
plt.plot(test_scores, color = 'green', linestyle='--', marker='s', markersize=5, label='Validation Accuracy')

plt.fill_between(np.arange(len(train_scores)),
                 train_scores - random_search_knn.cv_results_['std_train_score'],
                 train_scores + random_search_knn.cv_results_['std_train_score'],
                 alpha=0.15, color='blue')

plt.fill_between(np.arange(len(test_scores)),
                 test_scores - random_search_knn.cv_results_['std_test_score'],
                 test_scores + random_search_knn.cv_results_['std_test_score'],
                 alpha=0.15, color='green')


plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('KNN visualization')
plt.legend()
plt.show

# Apply the best parameters for prediction
y_pred_test = random_search_knn.best_estimator_.predict(test)


