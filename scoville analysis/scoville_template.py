import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Reading the data with pandas
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Selecting all columns without the first column
train = train_df.iloc[: ,1:]
test = test_df.iloc[:, 1:]


# Check for missing values
print(train.isna().sum())
print('------------------')
print(test.isna().sum())

#Created a boxplot for each numerical column to check for outliers
fig, ax = plt.subplots(figsize=(18,10))
sns.boxplot(data=train, ax=ax)
plt.show()



# Calculate the Z-scores for all the columns
z_scores = np.abs(stats.zscore(train))
# Set a threshold for the Z-score to identify outliers
threshold = 5
# Get the indices of the rows that contain outliers
outliers_indices = np.where(np.any(z_scores > threshold, axis=1))
# Remove the outliers from the dataset
train_clean = train.drop(outliers_indices[0], axis=0).reset_index(drop=True)
print(train_clean.shape)


#Created a boxplot for each numerical column to check for outliers
fig, ax = plt.subplots(figsize=(20,10))
sns.boxplot(data=train_clean, ax=ax)
plt.show()


#Defining the target value for the modelling
X = train_clean.drop('Scoville score', axis=1)
y = train_clean['Scoville score']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)



## DATA PIPELINE WITH REGRESSION MODEL ##
# Create a preprocessing and model pipeline
preprocessor = StandardScaler()
model = RandomForestRegressor(random_state=42)

pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

# Set up the hyperparameters for the random forest regressor
params = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [10, 15, 20],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [2, 4, 8],
    "model__max_features": ['sqrt', 'log2'],
}

# Use cross-validation and hyperparameter tuning to find the best parameters
cv = KFold(n_splits=10, shuffle=True, random_state=42)

gs = GridSearchCV(pipeline,
                  param_grid=params,
                  scoring="neg_mean_squared_error",
                  cv=cv, n_jobs=-1)

gs.fit(X_train, y_train)

# Best model
best_model = gs.best_estimator_

# Make predictions
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)


print('Root mean squared error (RMSE) - train: %.3f, test: %.3f' % (
        np.sqrt(mean_squared_error(y_train, y_pred_train)),
        np.sqrt(mean_squared_error(y_test, y_pred_test))))

print('Mean squared error (MSE)  - train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_pred_train),
        mean_squared_error(y_test, y_pred_test)))

print('R^2                       - train:  %.3f, test: %.3f' % (
        r2_score(y_train, y_pred_train),
        r2_score(y_test, y_pred_test)))

print('Mean absolute error (MAE) - train: %.3f,  test:  %.3f' % (
        mean_absolute_error(y_train, y_pred_train),
        mean_absolute_error(y_test, y_pred_test)))


# Make predictions on the actual test data
test_predictions = best_model.predict(test)

# Scatter plot of true vs predicted target values
plt.scatter(y_test, y_pred_test, alpha=0.3)
plt.xlabel('True Target Values')
plt.ylabel('Predicted Target Values')
plt.title('True vs Predicted Target Values')

# Add a reference line for perfect predictions
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', lw=2)

plt.show()



## BINNING TRAIN TARGET VALUES ##
n_bins = 2
y_train_binned = pd.cut(y_train, n_bins, labels=False)

# Preprocessor
preprocessor = StandardScaler()

# Base model
base_model = KNeighborsClassifier()

# The Bagging classifier
bagging_classifier = BaggingClassifier(estimator=base_model, random_state=42)

# Create a pipeline
pipeline = Pipeline([('preprocessor', preprocessor), ('model', bagging_classifier)])

# Parameters for the ensemble model
params = {
    "model__estimator__n_neighbors": range(1, 31),
    "model__estimator__weights": ['uniform', 'distance'],
    "model__estimator__algorithm": ['ball_tree', 'kd_tree', 'brute', 'auto'],
    "model__estimator__p": [1, 2, 3, 4],
    "model__n_estimators": range(10, 101, 10),
    "model__max_samples": np.arange(0.1, 1.1, 0.1),
    "model__max_features": np.arange(0.1, 1.1, 0.1),
    "model__bootstrap": [True, False],
    "model__bootstrap_features": [True, False]
}

cv = KFold(n_splits=10, shuffle=True, random_state=42)

rs = RandomizedSearchCV(pipeline,
                        params,
                        n_iter = 30,
                        scoring='neg_mean_squared_error',
                        cv=cv,
                        n_jobs=-1)

rs.fit(X_train, y_train_binned)

# Best model
best_model = rs.best_estimator_

# Make predictions
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)


print('Root mean squared error (RMSE) - train: %.3f, test: %.3f' % (
        np.sqrt(mean_squared_error(y_train, y_pred_train)),
        np.sqrt(mean_squared_error(y_test, y_pred_test))))

print('Mean squared error (MSE)  - train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_pred_train),
        mean_squared_error(y_test, y_pred_test)))

print('R^2                       - train:  %.3f, test: %.3f' % (
        r2_score(y_train, y_pred_train),
        r2_score(y_test, y_pred_test)))

print('Mean absolute error (MAE) - train: %.3f,  test:  %.3f' % (
        mean_absolute_error(y_train, y_pred_train),
        mean_absolute_error(y_test, y_pred_test)))


# Make predictions on the actual test data
test_predictions = best_model.predict(test)

# Scatter plot of true vs predicted target values
plt.scatter(y_test, y_pred_test, alpha=0.3)
plt.xlabel('True Target Values')
plt.ylabel('Predicted Target Values')
plt.title('True vs Predicted Target Values')

# Add a reference line for perfect predictions
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', lw=2)

plt.show()

# The issues with binning continous values to categorical is that we can lose a lot of information
# The more you increase the bins, the more information you can lose
# Which affects the performance of your model
# Thats why it is preferable to use regression models for continous tasks


## OTHER MODELS ##
from sklearn.linear_model import Ridge, Lasso

# Preprocessor
preprocessor = StandardScaler()

# Base models
rf_regressor = RandomForestRegressor(random_state=42)
ada_regressor = AdaBoostRegressor(random_state=42)
bagging_regressor = BaggingRegressor(estimator=LinearRegression(), random_state=42)


# Assigning a meta model for the stacking regressor
#meta_model=LinearRegression()
#meta_model = Ridge(alpha=1.0, random_state=42)
meta_model = Lasso(random_state=42)


# Stacking ensemble model
base_models = [("random_forest", rf_regressor), ("adaboost", ada_regressor), ("bagging", bagging_regressor)]
stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=meta_model)

# Create a pipeline
pipeline = Pipeline([("preprocessor", preprocessor), ("model", stacking_regressor)])

# Set up the hyperparameters for the ensemble model
params = {
    "model__random_forest__n_estimators": [50, 100, 200],
    "model__random_forest__max_depth": [10, 15, 20],
    "model__adaboost__n_estimators": [50, 100, 200],
    "model__adaboost__learning_rate": [0.001, 0.01, 0.1, 1],
    "model__bagging__n_estimators": [50, 100, 200],
    "model__bagging__max_samples": [0.5, 1.0],
    "model__bagging__max_features": [0.5, 1.0],
    "model__final_estimator__alpha": [0.01, 0.1, 1, 10]
}

# Perform hyperparameter tuning
cv = KFold(n_splits=10, shuffle=True, random_state=42)

rs = RandomizedSearchCV(pipeline,
                        params,
                        n_iter = 30,
                        scoring="neg_mean_squared_error",
                        cv=cv,
                        n_jobs=-1)

rs.fit(X_train, y_train)


print("Best parameters found:")
print(rs.best_params_)


# Best model
best_model = rs.best_estimator_

# Make predictions
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)


print('Root mean squared error (RMSE) - train: %.3f, test: %.3f' % (
        np.sqrt(mean_squared_error(y_train, y_pred_train)),
        np.sqrt(mean_squared_error(y_test, y_pred_test))))

print('Mean squared error (MSE)  - train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_pred_train),
        mean_squared_error(y_test, y_pred_test)))

print('R^2                       - train:  %.3f, test: %.3f' % (
        r2_score(y_train, y_pred_train),
        r2_score(y_test, y_pred_test)))

print('Mean absolute error (MAE) - train: %.3f,  test:  %.3f' % (
        mean_absolute_error(y_train, y_pred_train),
        mean_absolute_error(y_test, y_pred_test)))


# Make predictions on the actual test data
test_predictions = best_model.predict(test)

# Scatter plot of true vs predicted target values
plt.scatter(y_test, y_pred_test, alpha=0.3)
plt.xlabel('True Target Values')
plt.ylabel('Predicted Target Values')
plt.title('True vs Predicted Target Values')

# Add a reference line for perfect predictions
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', lw=2)

plt.show()


