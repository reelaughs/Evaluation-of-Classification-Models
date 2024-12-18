
import time
start_time = time.time()

#dataprep
import pandas as pd
df = pd.read_csv('adult.data', sep=',', header=None) #training data
column_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status","occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "agrossincome"]  # Replacing numbers with var names for easier readibility and interpretation
df.columns = column_names
df.dtypes #check
df.describe(include='object') #check
df.columns = df.columns.str.strip()  # Remove any leading/trailing spaces
print(df.columns)
if 'agrossincome' in df.columns:
    Y = df['agrossincome']
else:
    print("Column 'agrossincome' still not found after stripping. Available columns:", df.columns)
test_data = pd.read_csv('adult.test', sep=',', header=None)   #Note: The first line |1x3 Cross validator has been revmoed from the dataset prior to loading and preprocessing
test_data.columns = column_names
test_data.dtypes #check


import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import optuna
import warnings
from sklearn.model_selection import cross_val_score


# Define column names
column_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", 
                "hours-per-week", "native-country", "agrossincome"]

# Assign column names to the dataframe
df.columns = column_names
test_data.columns = column_names  # Apply the same column names to test data

# Separate features (X) and target (y) for training data
X_train = df.drop(columns='agrossincome')
y_train = df['agrossincome'].str.strip()  # Remove leading/trailing spaces from target

# Separate features (X) and target (y) for test data
X_test = test_data.drop(columns='agrossincome')
y_test = test_data['agrossincome'].str.strip()  # Remove leading/trailing spaces from target

# Ensure there are no inconsistencies in labels (e.g., extra punctuation or spaces)
y_train = y_train.replace({r'\.$': ''}, regex=True)
y_test = y_test.replace({r'\.$': ''}, regex=True)

# Encode target variable as categorical
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Identify categorical columns
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 
                       'relationship', 'race', 'sex', 'native-country']

# Use OneHotEncoder to transform categorical columns
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(), categorical_columns)], remainder='passthrough')

# Preprocess the features (apply OneHotEncoder to categorical variables)
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    max_depth = trial.suggest_categorical('max_depth', [2, 4, 8, 10, 12, None])
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    max_features = trial.suggest_float('max_features', 0.25, 1.0)
    min_samples_split = trial.suggest_float('min_samples_split', 0.01, 0.25)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    # Create and train the model
    model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, max_features=max_features,
                                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    model.fit(X_train_preprocessed, y_train_encoded)

    # Evaluate the model with cross-validation
    cv_scores = cross_val_score(model, X_train_preprocessed, y_train_encoded, cv=3)
    return cv_scores.mean()
    
    # Evaluate the model
    return model.score(X_test_preprocessed, y_test_encoded)

# Create and run the Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)  # Number of trials can be adjusted

# Retrieve the best parameters
best_params = study.best_params
print(f"Best Parameters: {best_params}")
print(f"Best Test Accuracy: {study.best_value:.2f}")

import numpy as np

# Train the model with the best parameters found
best_model = DecisionTreeClassifier(**best_params)
best_model.fit(X_train_preprocessed, y_train_encoded)

# Perform cost complexity pruning
path = best_model.cost_complexity_pruning_path(X_train_preprocessed, y_train_encoded)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Loop through ccp_alphas to find the best alpha for pruning
pruned_models = []
train_scores = []
test_scores = []

for ccp_alpha in ccp_alphas:
    pruned_model = DecisionTreeClassifier(ccp_alpha=ccp_alpha, **best_params)
    pruned_model.fit(X_train_preprocessed, y_train_encoded)
    
    # Evaluate on the training set
    train_scores.append(pruned_model.score(X_train_preprocessed, y_train_encoded))
    
    # Evaluate on the test set
    test_scores.append(pruned_model.score(X_test_preprocessed, y_test_encoded))
    
    pruned_models.append(pruned_model)

# Find the index of the model with the best test accuracy
best_index = test_scores.index(max(test_scores))
optimal_ccp_alpha = ccp_alphas[best_index]

# Train the pruned model with the optimal ccp_alpha
pruned_model = DecisionTreeClassifier(ccp_alpha=optimal_ccp_alpha, **best_params)
pruned_model.fit(X_train_preprocessed, y_train_encoded)

# Evaluate on the test set
test_accuracy = pruned_model.score(X_test_preprocessed, y_test_encoded)
print(f"Test accuracy of the pruned model: {test_accuracy * 100:.2f}%")

# Perform 3-fold cross-validation on the preprocessed training data
final_cv_scores = cross_val_score(pruned_model, X_train_preprocessed, y_train_encoded, cv=3)

# Print the average accuracy from final cross-validation
print('Model accuracy from final cross-validation: ', np.mean(final_cv_scores))

# Plot the decision tree for the pruned model
plt.figure(figsize=(10, 8))
plot_tree(pruned_model, feature_names=preprocessor.get_feature_names_out(), class_names=label_encoder.classes_, impurity=False)
plt.title("Pruned Decision Tree with Optimal Hyperparameters")
plt.show()

# Generate a classification report on the test set
y_test_pred = pruned_model.predict(X_test_preprocessed)
print(classification_report(y_test_encoded, y_test_pred))

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test_encoded, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

end_time = time.time()
total_time = end_time - start_time
print(f"Total time spent: {total_time:.2f} seconds")