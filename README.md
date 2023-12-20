# Combined Cycle Power Plant Full Load Electrical Power Output Prediction

## Overview
This repository contains the analysis and predictive modelling of the full load electrical power output a combined cycle power plant (CCPP) which is composed of gas turbines (GT), steam turbines (ST) and heat recovery steam generators. In a CCPP, the electricity is generated by gas and steam turbines, which are combined in one cycle, and is transferred from one turbine to another.

Computing the output of steam and gas power generation systems via thermodynamic analysis involves making many assumptions or solving thousands of nonlinear equations which requires an impractically large amount of computational effort and time. Machine learning can be an alternative approach to solving these kinds of problems. The machine learning approach approximates the performance of specific systems and configurations rather than solve the underlying thermodynamic equations to produce 'good-enough' answers for operational needs in a timely manner.

## Dataset Source
The dataset used in this analysis was obtained from UC Irvine Machine Learning Repository, at the URL:
[https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant](https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip). A copy of the Excel file included in the Zip archive downloaded from the link is included in `data` directory in this repository.

## Data Description
The dataset contains 9568 data points collected from a Combined Cycle Power Plant in Türkiye, over 6 years, from 2006 to 2011, when the power plant was set to work with full load. Features consist of hourly-averaged variables, AT (Ambient Temperature, °C), AP (Atmospheric Pressure, mbar), RH (Relative Humidity, %) and V (Exhaust Vacuum, cm Hg) to predict the net hourly electrical energy output PE(Full Load Electrical Power Output, MW) of the plant. Vacuum (V) measurements were made at the steam turbines. The other three variables, Ambient Temperature (AT), Atmospheric Pressure (AP) and Relative Humidity (RH), were collected at the gas turbines.

# Project Structure and Files Description

This project is organized into a set of folders and files to facilitate the full machine learning workflow, from data analysis to model deployment. Below is the folder structure and description of each element in this repository:
```
.
├── Dockerfile
├── Pipfile
├── Pipfile.lock
├── README.md
├── data
│   ├── Folds5x2_pp.xlsx
│   └── Readme.txt
├── deployment.yaml
├── experimentation_result
│   └── top_result.pkl
├── model
│   └── ccpp_model.pkl
├── notebook.ipynb
├── predict.py
├── predict_test_direct.py
├── predict_test_kube.py
├── service.yaml
├── test_data
│   └── test_data.csv
└── train.py
```

### Folders and Files

- `data`: Contains the dataset in an Excel file and a readme file

- `experimentation_result`: Stores the best performing machine learning model from experimentation
  - `top_result.pkl`: A serialized copy of a dataFrame containing the hyperparameters, performance metrics and a copy of the  best-performing model trained during the experimentation phase.

- `model`: Stores the final machine learning model
  - `ccpp_model.pkl`: A serialized copy of the final model, based on the best-performing model from experimentation, and trained on training and validation data.

- `test_data`: Stores the final machine learning model
  - `test_data.csv`: A CSV copy of the test dataset resulting from train-test-split during the experimentation phase. It is used in `predict_test.py` to provide test cases.

### Individual Files

- `deployment.yaml`: Instructions for creating a deployment in Kubernetes for hosting the Docker container of the prediction service.

- `Dockerfile`: Instructions for Docker to build the image used for creating containers that run the prediction service.

- `notebook.ipynb`: A Jupyter notebook for data preprocessing, basic exploratory data analysis (EDA), model experimentation, and hyperparameter tuning.

- `Pipfile` and `Pipfile.lock`: Files for replicating the development and production environments including within Docker containers.

- `predict_test_direct.py`: A Python script to test the prediction service directly in a local environment.

- `predict_test_kube.py`: A Python script to test the prediction service deployed in a Kubernetes cluster.

- `predict.py`: Contains the prediction service code. This script will be run within a Docker container hosted in a Kubernetes cluster.

- `README.md`: This file, which provides an overview and documentation for the project.

- `service.yaml`: Instructions for creating the service that routes and serves the prediction service.

- `train.py`: The training script used to fit the final machine learning model using details (selected features, hyperparameters, etc) of a model selected from candidates created during the experimentation phase. This script can be rerun for model updates or retraining purposes.

# Workflow Sequence and Code Walkthrough

The project workflow encompasses several stages, from data preparation to feature selection, to model training, and evaluation. Below is an outline of the sequence of activities.

### Data Ingestion

1. **Data Loading and Cleaning**: 
   - The raw dataset is loaded and preliminary checks for missing values and duplicates are conducted. Duplicates are removed to avoid data leakage.
   - Sanity check on the value ranges is done to verify data quality.

### Exploratory Data Analysis (EDA)

### Data Preparation

2. **Data Splitting**:
   - Data is split into training, validation, and test sets to ensure proper evaluation without data leakage.
   - The validation and test datasets are sized to contain over 1,000 rows each for robust testing.

3. **Preprocessing**:
   - A `StandardScalar` is used to scale the features prio. Only scaling is done as the data was already encoded.

4. **Feature Selection**:
   - Recursive Feature Elimination with Cross-Validation (RFECV) is employed to identify the most predictive features for each model.
   
5. **Model Training and Hyperparameter Tuning**:
   - Several machine learning algorithms are explored, including Logistic Regression, Random Forest, and XGBoost.
   - Class balancing techniques are used to handle imbalanced data, with strategies like class weighting, undersampling, and SMOTE.
   - Grid search with cross-validation is conducted to find the optimal set of hyperparameters for each model.

### Model Evaluation and Selection

6. **Evaluation**:
   - Models are evaluated on the validation set using metrics like accuracy, precision, recall, F1 score, and AUC-ROC.
   - Confusion matrices are generated to assess model performance in more detail.

7. **Compilation of Results**:
   - Evaluation results are compiled into a DataFrame, which is then sorted based on F1 score and other metrics to aid in model selection.

8. **Model Selection**:
   - The 'best' model is selected based on the evaluation criteria, which in this instance, uses F1 score and a leaning towards better positive class recall than that of the negative class, while maintaining some semblance of balance between the two.

### Finalization and Deployment

9. **Final Model Testing**:
   - The selected model is fit on the combined training and validation set and evaluated on the test set to verify generalization and performance.

10. **Model Serialization**:
   - The best model is serialized and saved in the `model` directory, ready for deployment in a production environment.

# How to Replicate This Project

To replicate this project and run the code on your own system, please follow the instructions below. These steps will guide you through the process of setting up the environment, cloning the repository, and running the model training and prediction service. _A [video](https://youtu.be/RqFJBDcS-v0) of the steps below is available on my YouTube channel._

### Prerequisites

- Ensure that you have Python, pip, and Git installed on your system.

### Environment Setup

1. **Install pipenv**:
   - Pipenv is a packaging tool for Python that simplifies dependency management. Install it using the command:
     ```
     pip install pipenv --user
     ```
   - For more detailed instructions on pipenv installation, refer to the [official documentation](https://pipenv.pypa.io/en/latest/installation/#make-sure-you-have-python-and-pip).

2. **Clone the Repository**:
   - Create a directory for the project and navigate to it in your terminal.
   - Clone the repository with the following command:
     ```
     git clone git@github.com:wongcheefah/mlzoomcamp_capstone_1.git
     ```

3. **Replicate the Environment**:
   - Inside the project directory, run the following command to replicate the development environment:
     ```
     pipenv install --dev
     ```
   - This will create a virtual environment and install all the necessary dependencies as specified in the `Pipfile`.

4. **Activate the Virtual Environment**:
   - To activate the virtual environment, use the command:
     ```
     pipenv shell
     ```

### Running the Model Training Script

- To train the model using the provided script, execute:
    ```
    python3 train.py
    ```

### Building and Running the Docker Image

1. **Build the Docker Image**:
 - With Docker installed on your system, build the image using:
   ```
   docker build -t ccpp:v001 .
   ```

2. **Start the Prediction Service Container**:
 - To start a Docker container that will run the prediction service, use:
   ```
   docker run -it --rm -p 9696:9696 ccpp
   ```
 - The container will listen for prediction requests on port `9696`.

### Testing the Prediction Service

- Open another terminal and run the following command to test the prediction service:
    ```
    python3 predict_test.py
    ```

- The test script will send a prediction request to the service running in the Docker container. The output will display the details of the request, the prediction response, and a comparison with the actual value.

By following these steps, you should be able to replicate the project environment and run the model training and prediction service as detailed in this project.

# Conclusion

This Combined Cycle Power Plant Full Load Electrical Power Output Prediction machine learning project aims to provide a simple and convenient predictive model to predict electrical power output without having to solve complex thermodynamic equations. The goal has been reached by training and serving a performant model.

The project underscores the usefulness of machine learning in power generation scenarios and the potential for predictive models to simplify and lower the computational cost of repetitive production planning and management in complex but specific and unchanging power generation scenarios. This is a good showcase of an alternative solution for computationally expensive or intractable problems that have to be solved repeatedly on a regular basis. However, depending on how they are set up, machine learning solutions may be limited to be sepcific scenarios or configurations and may not be generally applicable to variations of the same problem, as is the case here.

This project encapsulates the end-to-end process of a machine learning workflow, including data loading, data cleaning, exploratory data analysis, feature engineering, model training and tuning, and deployment of a prediction service. By following the replication instructions, users can set up their environment, run the model training script, and deploy the resulting model as a prediction service to make power output predictions based on four measured values. The provided Dockerfile ensures that the prediction service is containerized, making the deployment consistent and scalable across different platforms.

This repository is structured to not only serve as a platform for CCPP electrical power output prediction but also as an educational resource for those looking to learn and apply machine learning operations (MLOps) practices. It demonstrates the use of various tools and technologies such as Jupyter notebooks for experimentation, Pipenv for environment management, Docker for containerization, and Kubernetes for service serving.

Thank you for your interest in this project. For questions, suggestions, or contributions, please reach out through the project's GitHub repository.