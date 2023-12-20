# Import packages
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from math import sqrt

import pickle


# Load the data
print()
print("Loading dataset...")
df = pd.read_excel("./data/Folds5x2_pp.xlsx")

print("Loading completed")
print()

# Remove duplicates to prevent data leakage
print("Cleaning data...")
df.drop_duplicates(inplace=True)

print("Data cleaned")
print()

# Define target variable
target_variable = "PE"

# Separate features and target
X = df.drop(target_variable, axis=1)
y = df[target_variable]

# Define feature list
features = X.columns.values.tolist()

# Load the best model trained in the notebook
print("Loading model hyperparameters and algorithms from experimentation ...")
print("The processing steps, hyperparameters, algorithm and of the best")
print("performing model will be used to train the final model.")
top_result_file = "./experimentation_result/top_result.pkl"
with open(top_result_file, "rb") as f:
    top_result = pickle.load(f)

# Get the algorithm name and hyperparameters of the model
algorithm_name = top_result["Regressor"]
params = top_result["Model"].get_params()

print(f"Algorithm and hyperparameters loaded from '{top_result_file}'.")
print()

# Set the random state using the corresponding hyperparameter from the loaded
# top result file
random_state = [
    v for k, v in params["regressor"].get_params().items() if k == "random_state"
][0]

# Split data in the ratio 0.75 for training and 0.25 for validation and testing
print("Splitting the dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=random_state
)

# Split the 25% evenly to create a validation dataset and a test dataset
X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size=0.5, random_state=random_state
)

# Concatenate the training and validation datasets to form the final dataset
# used to train the final model
X_full_train = pd.concat([X_train, X_val])
y_full_train = pd.concat([y_train, y_val])

print("Splitting completed")
print()

# Check the shape of the full train and test datasets
print("Shape of full train and test datasets")
print("X_full_train    y_full_train")
print(f"{X_full_train.shape}       {y_full_train.shape}")
print()
print("X_test          y_test")
print(f"{X_test.shape}       {y_test.shape}")
print()

# Define the pipeline based on the selected model
print("Creating pipeline...")
steps = [("scaler", params["scaler"])]
if algorithm_name == "Support Vector":  # Add PCA for non-tree algorithm
    steps.append(("pca", params["pca"]))
steps.append(("regressor", params["regressor"]))

final_pipeline = Pipeline(steps)

print(final_pipeline)
print("Pipeline created")
print()

print("Training final model...")
final_model = final_pipeline.fit(X_full_train, y_full_train)

print("Training complete")
print()

# Evaluate the current model using test data
y_pred = final_model.predict(X_test)

# Display model details and performance metrics from test data
print("Model performance on test data")
print("―" * 30)
print(f"Algorithm: {algorithm_name}")
for k, v in final_model.get_params()["regressor"].get_params().items():
    print(f"{k}: {v}")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"RMSE: {sqrt(mean_squared_error(y_test, y_pred))}")
print(f"R2: {r2_score(y_test, y_pred)}")
print()

# Save the final model for use in production
final_model_file = "./model/ccpp_model.pkl"
with open(final_model_file, "wb") as f:
    pickle.dump(final_model, f)

print(f"Final model saved to '{final_model_file}'.")
print()
