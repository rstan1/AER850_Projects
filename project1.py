import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error


# Reading  data from a csv file and convert that into a dataframe, which will allow for all further
# analysis and data manipulation.
df = pd.read_csv("Project_1_Data.csv")

print(df.info())
print(df.head())

# Load the CSV file
file_path = 'Project_1_Data.csv'
data = pd.read_csv(file_path)

# Extracting the relevant columns for plotting
x = data['X']
y = data['Y']
z = data['Z']
steps = data['Step']

# Creating a 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(projection='3d')  # No need to import Axes3D separately

# Plotting the coordinates
scatter = ax.scatter(x, y, z, c=steps, cmap='viridis', marker='o')

# Adding color bar to show steps mapping
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label('Step')

# Labels and title
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title('3D Plot of Coordinates (X, Y, Z) versus Steps')

# Show the plot for 2.2
plt.show()

# Create a categorical variable for stratification based on 'Step'
# For example, categorize the 'Step' into ranges (adjust bins as needed)
data["step_categories"] = pd.cut(data["Step"],
                                 bins=[0, 1, 2, 3, 4, np.inf],  # Adjust bins if necessary
                                 labels=[1, 2, 3, 4, 5])

# Initialize StratifiedShuffleSplit
my_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Split data using stratified sampling
for train_index, test_index in my_splitter.split(data, data["step_categories"]):
    strat_df_train = data.loc[train_index].reset_index(drop=True)
    strat_df_test = data.loc[test_index].reset_index(drop=True)

# Drop the step_categories after stratification
strat_df_train = strat_df_train.drop(columns=["step_categories"], axis=1)
strat_df_test = strat_df_test.drop(columns=["step_categories"], axis=1)

# Separate features (X_train) and target (y_train)
X_train = strat_df_train[['X', 'Y', 'Z']]
y_train = strat_df_train['Step']
X_test = strat_df_test[['X', 'Y', 'Z']]
y_test = strat_df_test['Step']

# Scaling the data using StandardScaler
my_scaler = StandardScaler()
my_scaler.fit(X_train)

# Scale the training data
scaled_data_train = my_scaler.transform(X_train)
scaled_data_train_df = pd.DataFrame(scaled_data_train, columns=X_train.columns)

# Scale the testing data
scaled_data_test = my_scaler.transform(X_test)
scaled_data_test_df = pd.DataFrame(scaled_data_test, columns=X_test.columns)

# Optional: If you need to join the scaled data with the target 'Step' for later use
X_train_scaled = scaled_data_train_df.join(y_train)
X_test_scaled = scaled_data_test_df.join(y_test)

# Correlation matrix on the scaled training data (without 'Step')
correlation_matrix = scaled_data_train_df.corr()

# Plotting 2.3 the correlation matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(np.abs(correlation_matrix), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Scaled Features (X, Y, Z) in Training Data')
plt.show()

#Check the range and distribution of the Step values.
data['Step'].hist(bins=20)
plt.title('Distribution of Step Values')
plt.xlabel('Step')
plt.ylabel('Frequency')
plt.show()

# Logistic Regression
log_reg = LogisticRegression(solver='lbfgs', random_state=42)  # Ensure the solver is set
param_grid_lr = {
    'penalty': ['l2'],  # Only include 'l2' since 'lbfgs' does not support 'l1' or 'elasticnet'
    'C': [0.1, 1, 10, 100],
    'max_iter': [1000, 1500, 2000, 2500]  # Increased max_iter values
}
grid_search_lr = GridSearchCV(log_reg, param_grid_lr, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_lr.fit(X_train, y_train)
best_model_lr = grid_search_lr.best_estimator_
print("Best Logistic Regression Model:", best_model_lr)

# Support Vector Machine (SVM)
svm = SVC(random_state=42)
param_grid_svm = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto']
}
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_svm.fit(X_train, y_train)
best_model_svm = grid_search_svm.best_estimator_
print("Best SVM Model:", best_model_svm)

# Support Vector Machine (SVM)
svm = SVC(random_state=42)
param_grid_svm = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto']
}
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_svm.fit(X_train, y_train)
best_model_svm = grid_search_svm.best_estimator_
print("Best SVM Model:", best_model_svm)

# Random Forest
random_forest = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [10, 30, 50],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_rf = GridSearchCV(random_forest, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)
best_model_rf = grid_search_rf.best_estimator_
print("Best Random Forest Model:", best_model_rf)

# Decision Tree Classifier
decision_tree = DecisionTreeClassifier(random_state=42)
param_dist_dt = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
random_search_dt = RandomizedSearchCV(decision_tree, param_dist_dt, cv=5, scoring='accuracy', n_iter=10, n_jobs=-1, random_state=42)
random_search_dt.fit(X_train, y_train)
best_model_dt = random_search_dt.best_estimator_
print("Best Decision Tree Model (RandomizedSearchCV):", best_model_dt)

# Training and testing error for Linear Regression
y_train_pred_lr = best_model_lr.predict(X_train)
y_test_pred_lr = best_model_lr.predict(X_test)
mae_train_lr = mean_absolute_error(y_train, y_train_pred_lr)
mae_test_lr = mean_absolute_error(y_test, y_test_pred_lr)
print(f"Linear Regression - MAE (Train): {mae_train_lr}, MAE (Test): {mae_test_lr}")

# Training and testing error for SVM
y_train_pred_svr = best_model_svm.predict(X_train)
y_test_pred_svr = best_model_svm.predict(X_test)
mae_train_svr = mean_absolute_error(y_train, y_train_pred_svr)
mae_test_svr = mean_absolute_error(y_test, y_test_pred_svr)
print(f"SVM - MAE (Train): {mae_train_svr}, MAE (Test): {mae_test_svr}")

# Training and testing error for Decision Tree
y_train_pred_dt = best_model_dt.predict(X_train)
y_test_pred_dt = best_model_dt.predict(X_test)
mae_train_dt = mean_absolute_error(y_train, y_train_pred_dt)
mae_test_dt = mean_absolute_error(y_test, y_test_pred_dt)
print(f"Decision Tree - MAE (Train): {mae_train_dt}, MAE (Test): {mae_test_dt}")

# Training and testing error for Random Forest
y_train_pred_rf = best_model_rf.predict(X_train)
y_test_pred_rf = best_model_rf.predict(X_test)
mae_train_rf = mean_absolute_error(y_train, y_train_pred_rf)
mae_test_rf = mean_absolute_error(y_test, y_test_pred_rf)
print(f"Random Forest - MAE (Train): {mae_train_rf}, MAE (Test): {mae_test_rf}")

