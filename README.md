# Combined Cycle Power Plant Full Load Electrical Power Output Prediction

## Overview
This repository contains the analysis and predictive modelling of the full load electrical power output a combined cycle power plant (CCPP) which is composed of gas turbines (GT), steam turbines (ST) and heat recovery steam generators. In a CCPP, the electricity is generated by gas and steam turbines, which are combined in one cycle, and is transferred from one turbine to another.

Computing the output of steam and gas power generation systems via thermodynamic analysis involves making many assumptions or solving thousands of nonlinear equations which requires an impractically large amount of computational effort and time. Machine learning can be an alternative approach to solving these kinds of problems. The machine learning approach approximates the performance of specific systems and configurations rather than solve the underlying thermodynamic equations to produce 'good-enough' answers for operational needs in a timely manner. This project aims to provide a simple and convenient predictive model to predict CCPP full load electrical power output without having to solve complex thermodynamic equations.

## Dataset Source
The dataset used in this analysis was obtained from UC Irvine Machine Learning Repository, at the URL:
[https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant](https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip). A copy of the Excel file included in the Zip archive downloaded from the link is included in `data` directory in this repository.

## Data Description
The dataset contains 9568 data points collected from a Combined Cycle Power Plant in Türkiye, over 6 years, from 2006 to 2011, when the power plant was set to work with full load. Features consist of hourly-averaged variables, AT (Ambient Temperature, °C), AP (Atmospheric Pressure, mbar), RH (Relative Humidity, %) and V (Exhaust Vacuum, cm Hg) to predict the net hourly electrical energy output PE(Full Load Electrical Power Output, MW) of the plant. Vacuum (V) measurements were made at the steam turbines. The other three variables, Ambient Temperature (AT), Atmospheric Pressure (AP) and Relative Humidity (RH), were collected at the gas turbines.

# Project Structure and Files Description

The folder and file structure used for this project facilitates the full machine learning workflow, from data analysis to model deployment, while keeping the files organised. Below is the folder structure and description of each element in this repository:
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

- `deployment.yaml`: Instructions for creating a deployment in kind Kubernetes for hosting the Docker container of the prediction service.

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

### Exploratory Data Analysis (EDA)

2. **Understanding Dataset Characteristics**:
   - Basic characteristics of the dataset, such as number of observations, number of variables, presence of missing values, etc is observed.
   - Sanity check on the value ranges is done to verify data quality.

3. **Investigating Correlations and Data Distribution**:
   - The correlation between feature-target pairs and feature-feature pair is looked into via a pair plot and a correlation matrix.
   - Histograms and boxplot are created to study the distribution od the data and check for presence of outliers.
   - Further investigation into outliers is done via scatterplot to see where they occur and how far they are from the rest of the data points and whether they are attached to or detached from them.

### Data Preparation

4. **Data Splitting**:
   - Data is split into training, validation, and test sets to ensure proper evaluation without data leakage.
   - The validation and test datasets are sized to contain over 1,000 rows each for robust testing.

5. **Preprocessing**:
   - A `StandardScalar` is used to scale all the features.
   - To address multicolinearity in the features, `PCA` is applied for the `SVR` pipeline but not required in `RandomForestRegressor` and `GradientBoostingRegressor`.

### Base Model Training and Tuning

7. **Model Training and Hyperparameter Tuning**:
   - Several machine learning algorithms are explored, including Support Vector, Random Forest, and Gradient Boosting.
   - Grid search with cross-validation is conducted to find the optimal set of hyperparameters for each model.

### Base Model Evaluation and Selection

8. **Evaluation**:
   - Models are tested on validation data using mean squared error as the performance metric.

9. **Compilation of Results**:
   - Model details such as tuned hyperparameters and performance metrics and the models themselves are stored into a dataframe, which is then sorted based on mean squared error to facilitate model selection.

10. **Model Selection**:
   - The dataframe row with the lowest mean squared error is selected and saved as a Pickle file for use later in training the final model.

### Model Finalization

11. **Final Model Training**:
   - The final model pipeline is created using the saved model as a template. Training is done using the combined training and validation datasets and evaluated on the test set to verify generalization and performance.

12. **Final Model Evaluation**:
   - The final model is then evaluated using the test dataset to verify performance and generalization.

13. **Model Serialization**:
   - It is serialized and saved in the `model` directory, ready for deployment in a production environment.

# Replicating This Project

The instructions given here are specific to Ubuntu on WSL, the environment the project is done in. To replicate this project and the environment, follow the instructions below. These steps will guide you through the process of setting up the environment, cloning the repository, and running the model training and prediction service. If you are running a different OS or platform, you may need to refer to online resources for instructions specific to your setup. _A [video](https://youtu.be/RqFJBDcS-v0) of the steps below is available on my YouTube channel._

### System Prerequisites

Ensure that you have Python, pip, Git and Docker installed on your system and that Docker Desktop (if you are on Windows or MacOS) or the Docker daemon (on Linux) is running.

### Project Environment Setup

1. **Install pipenv**:
   - Pipenv is a packaging tool for Python that simplifies dependency management. Install it using the command:
     ```
     pip install pipenv --user
     ```
   - The installation completion message will include instruction to update your path environment variable to include to path to pipenv. You can do so by adding `export PATH="<pipenv path>:$PATH"` to the `.bashrc` file in your home directory. Be sure to use the correct pipenv path.
   - For more detailed instructions on pipenv installation, refer to the [official documentation](https://pipenv.pypa.io/en/latest/installation/#make-sure-you-have-python-and-pip).

2. **Install kind**:
   - In the your home directory, download the file:
     ```
     [ $(uname -m) = x86_64 ] && curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
     ```
   - Make the file executable:
     ```
     chmod +x ./kind
     ```
   - Move the file::
     ```
     sudo mv ./kind /usr/local/bin/kind
     ```
   - Test the installation by checking the version:
     ```
     kind --version
     ```
   - If you need to install it on other OS or platform, navigate to [https://kind.sigs.k8s.io/docs/user/quick-start/](https://kind.sigs.k8s.io/docs/user/quick-start/) for specific instructions.

3. **Install kubectl**:
   - In the your home directory, download version 1.29.0 of kubectl:
     ```
     curl -LO "https://dl.k8s.io/release/v1.29.0/bin/linux/amd64/kubectl"
     ```
   - Install kubectl:
     ```
     sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
     ```
   - Test the installation by checking the version:
     ```
     kubectl version --client
     ```
   - Delete the installation file:
     ```
     rm -f kubectl
     ```
   - If you need to install it on other OS or platform, navigate to [https://kubernetes.io/docs/tasks/tools/](https://kubernetes.io/docs/tasks/tools/) for specific instructions.

4. **Clone the Repository**:
   - Create a directory for the project and navigate to it in your terminal:
     ```
     mkdir projects
     cd projects
     ```
   - Clone the repository with the following command:
     ```
     git clone https://github.com/wongcheefah/mlzoomcamp_capstone_1.git
     ```

5. **Replicate the Environment**:
   - Navigate to the capstone project directory:
     ```
     cd mlzoomcamp_capstone_1
     ```
   - Inside the capstone project directory, run the following command to replicate the Python virtual environment:
     ```
     pipenv install --dev
     ```
   - This will create a virtual environment and install all the necessary dependencies as specified in the `Pipfile`.

### Running the Model Training Script

6. **Final Model Training**:
   - In the capstone project directory, activate the virtual environment:
     ```
     pipenv shell
     ```
   - Run the train script:
     ```
     python3 train.py
     ```
   - The script will train the final model using the model, saved in the `experimentation_result` directory, as a template, on training and validation data combined.

### Direct Prediction Testing

7. **Direct Testing**:
   - In the capstone project directory, activate the virtual environment:
     ```
     pipenv shell
     ```
   - Run the prediction script:
     ```
     python3 predict.py
     ```
   - Open another terminal and navigate to the capstone project directory:
     ```
     cd projects/mlzoomcamp_capstone_1
     ```
   - Activate the virtual environment:
     ```
     pipenv shell
     ```
   - Run the direct prediction script:
     ```
     python3 predict_test_direct.py
     ```

### Building the Docker Image

8. **Build the Docker Image**:
   - In the capstone project directory, build the prediction service container image from the Dockerfile:
     ```
     docker build -t ccpp:v001 .
     ```
   - Verify the image has been built:
     ```
     docker image list ccpp
     ```

### Kubernetes Prediction Testing

9. **Create a Kubernetes Cluster**:
   - In the capstone project directory, create a Kubernetes cluster:
     ```
     kind create cluster
     ```
   - Check that the cluster is up by getting cluster info:
     ```
     kubectl cluster-info --context kind-kind
     ```
   - Load the image into node:
     ```
     kind load docker-image ccpp:v001
     ```

10. **Create a Deployment**:
    - In the capstone project directory, create a deployment:
      ```
      kubectl apply -f deployment.yaml
      ```
    - Verify creation of ccpp deployment:
      ```
      kubectl get deployment
      kubectl get pod
      ```

11. **Create a Service**:
    - In the capstone project directory, create a service:
      ```
      kubectl apply -f service.yaml
      ```
    - Verify creation of ccpp service:
      ```
      kubectl get service ccpp
      ```
    - Configure port-forwarding for the service:
      ```
      kubectl port-forward service/ccpp 8080:80
      ```

12. **Kubernetes Testing**:
    - Open another terminal and navigate to the capstone project directory:
      ```
      cd projects/mlzoomcamp_capstone_1
      ```
    - Activate the virtual environment:
      ```
      pipenv shell
      ```
    - Run the direct prediction script:
      ```
      python3 predict_test_kube.py
      ```

The test script will send a prediction request to the service running in the Docker container in the deployment in the Kubernetes cluster. The output will display the details of the request, the prediction response, and a comparison with the actual value.

By following these steps, you should be able to replicate the project environment and run the model training and prediction service.

# Conclusion

This Combined Cycle Power Plant Full Load Electrical Power Output Prediction machine learning project aims to provide a simple and convenient predictive model to predict electrical power output without having to solve complex thermodynamic equations. The goal has been reached by training and serving a performant model.

The project underscores the usefulness of machine learning in power generation scenarios and the potential for predictive models to simplify and lower the computational cost of repetitive production planning and management in complex but specific and unchanging power generation scenarios. This is a good showcase of an alternative solution for computationally expensive or intractable problems that have to be solved repeatedly on a regular basis. However, depending on how they are set up, machine learning solutions may be limited to be sepcific scenarios or configurations and may not be generally applicable to variations of the same problem, as is the case here.

This project encapsulates the end-to-end process of a machine learning workflow, including data loading, data cleaning, exploratory data analysis, feature engineering, model training and tuning, and deployment of a prediction service. By following the replication instructions, users can set up their environment, run the model training script, and deploy the resulting model as a prediction service to make power output predictions based on four measured values. The provided Dockerfile ensures that the prediction service is containerized, making the deployment consistent and scalable across different platforms.

This repository is structured to not only serve as a platform for CCPP electrical power output prediction but also as an educational resource for those looking to learn and apply machine learning operations (MLOps) practices. It demonstrates the use of various tools and technologies such as Jupyter notebooks for experimentation, Pipenv for environment management, Docker for containerization, and Kubernetes for service serving.

Thank you for your interest in this project. For questions, suggestions, or contributions, please reach out through the project's GitHub repository.