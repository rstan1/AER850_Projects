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
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, accuracy_score
from sklearn.ensemble import StackingClassifier
import joblib

# Reading  data from a csv file and convert that into a dataframe
df = pd.read_csv("Project_1_Data.csv")

print(df.info())
print("\n")
print(df.head())
print("\n")

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

#Statisitcs of Data
df = pd.DataFrame(data)
print("Statistics")
print(df.describe())
print("\n")

data["step_categories"] = pd.cut(data["Step"],
                                 bins=[0, 1, 2, np.inf], 
                                 labels=[1, 2, 3])

# StratifiedShuffleSplit
my_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

# Split data
for train_index, test_index in my_splitter.split(data, data["step_categories"]):
    strat_df_train = data.loc[train_index].reset_index(drop=True)
    strat_df_test = data.loc[test_index].reset_index(drop=True)

strat_df_train = strat_df_train.drop(columns=["step_categories"], axis=1)
strat_df_test = strat_df_test.drop(columns=["step_categories"], axis=1)

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

scaled_data_train_df_with_step = scaled_data_train_df.copy()
scaled_data_train_df_with_step['Step'] = y_train.values

# Correlation matrix 
correlation_matrix_with_step = scaled_data_train_df_with_step.corr()

# Plotting step 2.3 correlation matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(np.abs(correlation_matrix_with_step), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Scaled Features (X, Y, Z) and Step in Training Data')
plt.show()

data['Step'].hist(bins=20)
plt.title('Distribution of Step Values')
plt.xlabel('Step')
plt.ylabel('Frequency')
plt.show()

# Logistic Regression
log_reg = LogisticRegression(solver='lbfgs', random_state=42) 
param_grid_lr = {
    'penalty': ['l2'], 
    'C': [0.1, 1, 10, 100],
    'max_iter': [3000] 
}
grid_search_lr = GridSearchCV(log_reg, param_grid_lr, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_lr.fit(X_train, y_train)
best_model_lr = grid_search_lr.best_estimator_
print("Best Logistic Regression Model:", best_model_lr)
print("\n")

# Predictions for Logistic Regression
y_pred_logreg = best_model_lr.predict(X_test)

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
print("\n")

# Predictions for SVM
y_pred_svc = best_model_svm.predict(X_test)

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
print("\n")

# Predictions for Random Forest
y_pred_rf = best_model_rf.predict(X_test)

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
print("\n")

# Predictions for Decision Tree
y_pred_dt = best_model_dt.predict(X_test)

# Training and testing error for Logistic Regression
y_train_pred_lr = best_model_lr.predict(X_train)
y_test_pred_lr = best_model_lr.predict(X_test)


# Training and testing error for SVM
y_train_pred_svr = best_model_svm.predict(X_train)
y_test_pred_svr = best_model_svm.predict(X_test)


# Training and testing error for Decision Tree
y_train_pred_dt = best_model_dt.predict(X_train)
y_test_pred_dt = best_model_dt.predict(X_test)


# Training and testing error for Random Forest
y_train_pred_rf = best_model_rf.predict(X_train)
y_test_pred_rf = best_model_rf.predict(X_test)

results = pd.DataFrame({
    'Actual': y_test,
    'Logistic Regression Predicted': y_pred_logreg,
    'SVM Predicted': y_pred_svc,
    'Random Forest Predicted': y_pred_rf,
    'Decision Tree Predicted': y_pred_dt
})

results = pd.DataFrame({
    'Actual': y_test,
    'Logistic Regression Predicted': y_pred_logreg,
    'SVM Predicted': y_pred_svc,
    'Random Forest Predicted': y_pred_rf,
    'Decision Tree Predicted': y_pred_dt
})

print(results.head(20))

models = ['Logistic Regression', 'SVM', 'Random Forest', 'Decision Tree']
y_preds = [y_pred_logreg, y_pred_svc, y_pred_rf, y_pred_dt]

for model, y_pred in zip(models, y_preds):
    print(f"{model} Metrics:")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("\n")

# Confusion matrix for the best model (dt)
cm_dt = confusion_matrix(y_test, y_pred_dt)

# Plotting confusion matrix for Decision Tree
plt.figure(figsize=(8, 6))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(y_test), 
            yticklabels=np.unique(y_test))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix for Decision Tree')
plt.show()

# Base models to stack
estimators = [
    ('rf', best_model_rf),  # assuming best_model_rf is your trained RandomForest model
    ('dt', best_model_dt)   # assuming best_model_dt is your trained DecisionTree model
]

stacked_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=RandomForestClassifier(random_state=42)
)

stacked_clf.fit(X_train, y_train)

y_pred_stacked = stacked_clf.predict(X_test)

f1_stacked = f1_score(y_test, y_pred_stacked, average='weighted')
precision_stacked = precision_score(y_test, y_pred_stacked, average='weighted')
accuracy_stacked = accuracy_score(y_test, y_pred_stacked)

print("Stacked Model Performance:")
print(f"F1 Score: {f1_stacked}")
print(f"Precision: {precision_stacked}")
print(f"Accuracy: {accuracy_stacked}")
print("\n")

# Confusion Matrix for the Stacked Model
cm_stacked = confusion_matrix(y_test, y_pred_stacked)

# Plotting confusion matrix for the stacked model
plt.figure(figsize=(8, 6))
sns.heatmap(cm_stacked, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(y_test), 
            yticklabels=np.unique(y_test))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix for Stacked Model')
plt.show()

# Saving the stacked model to joblib file
joblib_filename = 'stacked_model.joblib'
joblib.dump(stacked_clf, joblib_filename)
print(f"Stacked model saved as {joblib_filename}")

loaded_model = joblib.load(joblib_filename)

# Random set of coordinates provided
coordinates = [
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0, 3.0625, 1.93],
    [9.4, 3, 1.8],
    [9.4, 3, 1.3]
]

column_names = ['X', 'Y', 'Z']
coordinates_df = pd.DataFrame(coordinates, columns=column_names)

# Making predictions using the loaded model
predictions = loaded_model.predict(coordinates_df)

# Printing predictions
for i, coord in enumerate(coordinates):
    print(f"Coordinates: {coord} -> Predicted Class: {predictions[i]}")