import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load CSV file into a Pandas DataFrame
train_df = pd.read_csv('train.csv') 
test_df = pd.read_csv('test.csv')


# Select all features except the first one called ['Unnamed:0']
train = train_df.iloc[:, 1:]
test = test_df.iloc[:, 1:]


#Created a boxplot for each numerical column to check for outliers
fig, ax = plt.subplots(figsize=(18,10))
train.plot(kind='box', ax=ax)
plt.show()


# Fill missing values with the median for the respective feature
train.fillna(train.median(), inplace=True)
test.fillna(test.median(), inplace=True)


# Define the Tukey method to remove outliers
def remove_outliers_tukey(train):
    Q1 = train.quantile(0.25)
    Q3 = train.quantile(0.75)
    IQR = Q3 - Q1
    return train[~((train < (Q1 - 1.5 * IQR)) | (train > (Q3 + 1.5 * IQR))).any(axis=1)]

# Apply the Tukey method to remove outliers from the whole dataframe
train_clean = remove_outliers_tukey(train)
train_clean.shape
#After cleaning, went from 2040 to 1799 rows


# Created a boxplot for each numerical column after cleaning the data from possible outliers
fig, ax = plt.subplots(figsize=(18,10))
train_clean.plot(kind='box', ax=ax)
plt.show()


# Split the train.csv into X and y
X = train_clean.drop('Drinkable', axis=1)
y = train_clean['Drinkable']


print(X.shape)
print(y.shape)


# Set the parameters for Random Forest
n_estimators_list = [100, 200, 500]
max_depth_list = [20, 30, 40]
min_samples_leaf_list = [2, 4, 10]
min_samples_split_list = [2, 5, 10]
max_features_list = ['sqrt', 'log2']


best_params = {'n_estimators': None,
               'max_depth': None,
               'min_samples_leaf': None,
               'min_samples_split': None,
               'max_features': None}


# No need to scale when using random forrest 
# Random Forest is a tree-based model and hence does not require feature scaling

# The model training is done without using built in methods

# Lists where we want to store the accuracies
train_accuracies = []
val_accuracies = []

# Create a loop for multiple train-test splits
for _ in range(20):
    
    # Reset the best accuracy for each iteration
    best_accuracy = 0
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
                                                      random_state=42, stratify=y)

    # Train the Random Forest model with different parameters
    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            for min_samples_leaf in min_samples_leaf_list:
                for min_samples_split in min_samples_split_list:
                    for max_features in max_features_list:
                        model = RandomForestClassifier(n_estimators=n_estimators,
                                                       max_depth=max_depth,
                                                       min_samples_leaf=min_samples_leaf,
                                                       min_samples_split=min_samples_split,
                                                       max_features=max_features,
                                                       n_jobs=-1)
                                                      
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_val)

                        # Evaluate the model's performance
                        accuracy = accuracy_score(y_val, predictions)

                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_params['n_estimators'] = n_estimators
                            best_params['max_depth'] = max_depth
                            best_params['min_samples_leaf'] = min_samples_leaf
                            best_params['min_samples_split'] = min_samples_split
                            best_params['max_features'] = max_features



    # Calculate training accuracy
    train_predictions = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(best_accuracy)


# Print the best parameters
print('Best parameters:', best_params)


# Plot the training and validation accuracy
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')


plt.xticks(range(0, 21, 2))
plt.xlabel('Iterations')


plt.fill_between(np.arange(len(train_accuracies)),
                 np.mean(train_accuracies) - np.std(train_accuracies),
                 np.mean(train_accuracies) + np.std(train_accuracies),
                 alpha=0.1)

plt.fill_between(np.arange(len(val_accuracies)),
                 np.mean(val_accuracies) - np.std(val_accuracies),
                 np.mean(val_accuracies) + np.std(val_accuracies),
                 alpha=0.1)


plt.ylim(0.80, 1.00)
plt.ylabel('Accuracy')
plt.legend()

plt.show()



# Train the model with the best parameters on the whole training data
best_model = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                    max_depth=best_params['max_depth'],
                                    min_samples_leaf=best_params['min_samples_leaf'],
                                    min_samples_split=best_params['min_samples_split'],
                                    max_features = best_params['max_features'])

best_model.fit(X, y)


# Predict y for X_test from test.csv
X_test = test
pred_test = best_model.predict(X_test)


print(best_accuracy)
print(train_accuracy)