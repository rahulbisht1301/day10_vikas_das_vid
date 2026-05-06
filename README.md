# Vehicle Insurance Response Prediction - MLOps Pipeline

A comprehensive **end-to-end Machine Learning Operations (MLOps)** project that predicts vehicle insurance customer response using a production-ready machine learning pipeline. This project demonstrates best practices in ML model development, deployment, and inference with cloud integration.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Pipeline Components](#pipeline-components)
- [API Endpoints](#api-endpoints)
- [Model Details](#model-details)
- [Cloud Integration](#cloud-integration)
- [Deployment](#deployment)
- [Logging & Monitoring](#logging--monitoring)
- [Contributing](#contributing)

---

## 🎯 Project Overview

This project implements a **complete MLOps workflow** for predicting whether vehicle insurance customers will respond to a new product (vehicle damage insurance). The system includes:

- **Automated data pipeline** that fetches data from MongoDB
- **Data validation and transformation** modules
- **Model training with hyperparameter optimization**
- **Model evaluation and comparison**
- **Production-ready inference pipeline**
- **Web interface** for making predictions
- **Cloud deployment** with AWS S3 integration

### Business Problem
Predict if customers will be interested in buying vehicle damage insurance based on their profile and historical behavior.

### Target Variable
- `Response`: Binary classification (0: Not interested, 1: Interested)

---

## ✨ Features

- ✅ **End-to-End Pipeline**: Data ingestion → Validation → Transformation → Training → Evaluation → Deployment
- ✅ **Cloud-Native**: AWS S3 integration for model registry
- ✅ **MongoDB Integration**: Seamless data ingestion from MongoDB
- ✅ **FastAPI Web Interface**: Modern, async REST API for predictions
- ✅ **Model Versioning**: Automatic model versioning with timestamps
- ✅ **Comprehensive Logging**: Detailed logging at every pipeline stage
- ✅ **Exception Handling**: Custom exception management across the project
- ✅ **Data Validation**: Schema-based data validation with YAML configuration
- ✅ **Model Evaluation**: Multi-metric model performance evaluation
- ✅ **Reproducibility**: Fixed random seeds and consistent preprocessing

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA SOURCE (MongoDB)                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            DATA INGESTION COMPONENT                         │
│  - Fetch data from MongoDB                                  │
│  - Create feature store (CSV)                               │
│  - Split train/test (75/25)                                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            DATA VALIDATION COMPONENT                        │
│  - Validate schema against YAML config                      │
│  - Generate validation reports                              │
│  - Handle data quality checks                               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         DATA TRANSFORMATION COMPONENT                       │
│  - Encode categorical variables (Gender, Vehicle_Age, etc)  │
│  - Scale numerical features (MinMaxScaler)                  │
│  - Handle imbalanced data (SMOTE)                           │
│  - Feature engineering                                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│          MODEL TRAINING COMPONENT                           │
│  - Random Forest Classifier                                 │
│  - Hyperparameters: n_estimators=20, max_depth=10          │
│  - Train/evaluate on transformed data                       │
│  - Save model and preprocessing objects                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│          MODEL EVALUATION COMPONENT                         │
│  - Evaluate trained model on test set                       │
│  - Compare with production model (if exists)                │
│  - Calculate performance metrics                            │
│  - Determine if model is production-ready                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│          MODEL PUSHER COMPONENT                             │
│  - Upload model to AWS S3 (model registry)                  │
│  - Version management                                       │
│  - Production deployment                                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│        PREDICTION PIPELINE & WEB API                        │
│  - Load production model from S3                            │
│  - FastAPI endpoints for predictions                        │
│  - Web UI for interactive predictions                       │
│  - Real-time inference                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
day10_vikas_das_vid/
├── app.py                           # FastAPI application entry point
├── demo.py                          # Demo/testing script
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Project metadata and build config
├── setup.py                         # Package setup configuration
├── Dockerfile                       # Docker containerization
├── template.py                      # Project template generator
│
├── config/                          # Configuration files
│   ├── model.yaml                   # Model hyperparameters
│   └── schema.yaml                  # Data schema validation
│
├── src/                             # Main source code
│   ├── components/                  # Pipeline components
│   │   ├── data_ingestion.py        # Data ingestion from MongoDB
│   │   ├── data_validation.py       # Schema validation
│   │   ├── data_transformation.py   # Feature engineering & preprocessing
│   │   ├── model_trainer.py         # Model training
│   │   ├── model_evaluation.py      # Model evaluation
│   │   └── model_pusher.py          # Push model to S3
│   │
│   ├── pipline/                     # Pipeline orchestration
│   │   ├── training_pipeline.py     # Training pipeline orchestrator
│   │   └── prediction_pipeline.py   # Inference pipeline
│   │
│   ├── entity/                      # Data entities
│   │   ├── config_entity.py         # Configuration classes
│   │   ├── artifact_entity.py       # Artifact classes
│   │   ├── estimator.py             # Model estimator
│   │   └── s3_estimator.py          # S3-based model estimator
│   │
│   ├── configuration/               # External service connections
│   │   ├── mongo_db_connection.py   # MongoDB connection
│   │   └── aws_connection.py        # AWS S3 connection
│   │
│   ├── cloud_storage/               # Cloud storage integration
│   │   └── aws_storage.py           # AWS S3 operations
│   │
│   ├── data_access/                 # Data access layer
│   │   └── proj1_data.py            # MongoDB data access
│   │
│   ├── logger/                      # Logging configuration
│   │   └── __init__.py              # Logger setup
│   │
│   ├── exception/                   # Exception handling
│   │   └── __init__.py              # Custom exceptions
│   │
│   ├── constants/                   # Project constants
│   │   └── __init__.py              # Constant definitions
│   │
│   └── utils/                       # Utility functions
│       └── ...                      # Helper utilities
│
├── templates/                       # HTML templates
│   └── vehicledata.html             # Web form for predictions
│
├── static/                          # Static assets
│   └── css/                         # CSS stylesheets
│
├── notebook/                        # Jupyter notebooks
│   ├── data.csv                     # Sample data
│   ├── exp-notebook.ipynb           # Experimentation notebook
│   └── mongodb_file.ipynb           # MongoDB integration notebook
│
├── artifact/                        # Pipeline artifacts (timestamped)
│   └── [TIMESTAMP]/
│       ├── data_ingestion/          # Ingested data
│       ├── data_validation/         # Validation reports
│       ├── data_transformation/     # Transformed data & preprocessing objects
│       └── model_trainer/           # Trained models
│
├── logs/                            # Application logs
│
└── src.egg-info/                    # Package information
```

---

## 🛠️ Technology Stack

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.x |
| **Web Framework** | FastAPI, Uvicorn |
| **ML/Data** | scikit-learn, pandas, numpy |
| **Data Processing** | imbalanced-learn (SMOTE), dill |
| **Database** | MongoDB |
| **Cloud** | AWS S3, boto3 |
| **Visualization** | matplotlib, plotly, seaborn |
| **Configuration** | PyYAML |
| **Templating** | Jinja2 |
| **Development** | mypy-boto3-s3, certifi |
| **Jupyter** | ipykernel, jupyter |

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- MongoDB instance (local or cloud)
- AWS S3 bucket (for model registry)
- pip package manager

### Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd day10_vikas_das_vid
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package in development mode**
   ```bash
   pip install -e .
   ```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# MongoDB
MONGODB_URL=<your-mongodb-connection-string>

# AWS S3
AWS_ACCESS_KEY_ID=<your-aws-access-key>
AWS_SECRET_ACCESS_KEY=<your-aws-secret-key>
```

### Configuration Files

#### `config/schema.yaml`
Defines the data schema for validation:
```yaml
columns:
  - id: int
  - Gender: category
  - Age: int
  - Driving_License: int
  - Region_Code: float
  - Previously_Insured: int
  - Vehicle_Age: category
  - Vehicle_Damage: category
  - Annual_Premium: float
  - Policy_Sales_Channel: float
  - Vintage: int
  - Response: int

numerical_columns:
  - Age, Driving_License, Region_Code, Previously_Insured, 
    Annual_Premium, Policy_Sales_Channel, Vintage, Response

categorical_columns:
  - Gender, Vehicle_Age, Vehicle_Damage
```

#### `config/model.yaml`
Model hyperparameters and configuration (can be extended)

### Constants
Key constants are defined in [src/constants/__init__.py](src/constants/__init__.py):

```python
DATABASE_NAME = "Proj1"
COLLECTION_NAME = "Proj1-Data"
TARGET_COLUMN = "Response"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO = 0.25
MODEL_TRAINER_EXPECTED_SCORE = 0.6
MODEL_TRAINER_N_ESTIMATORS = 20
MODEL_TRAINER_MAX_DEPTH = 10
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE = 0.02
```

---

## 🚀 Usage

### Running the Training Pipeline

```python
from src.pipline.training_pipeline import TrainPipeline

# Initialize and run the training pipeline
train_pipeline = TrainPipeline()
train_pipeline.run_pipeline()
```

Or via the API:
```bash
curl http://localhost:5000/train
```

### Running the Web Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

#### Web Interface Features:
- Input form for vehicle customer data
- Real-time predictions
- Model training trigger
- Visual results display

### API Endpoints

#### 1. **Home Page**
```
GET /
```
Returns the main HTML form page for making predictions.

**Response**: HTML page with vehicle data input form

---

#### 2. **Predict**
```
POST /
Content-Type: multipart/form-data

Parameters:
- Gender: int (0/1)
- Age: int
- Driving_License: int (0/1)
- Region_Code: float
- Previously_Insured: int (0/1)
- Annual_Premium: float
- Policy_Sales_Channel: float
- Vintage: int
- Vehicle_Age_lt_1_Year: int (0/1, one-hot encoded)
- Vehicle_Age_gt_2_Years: int (0/1, one-hot encoded)
- Vehicle_Damage_Yes: int (0/1, one-hot encoded)
```

**Response**: 
```json
{
    "prediction": 1,
    "probability": 0.85
}
```

---

#### 3. **Train Model**
```
GET /train
```
Triggers the full training pipeline.

**Response**:
```
Training successful!!!
```

Or error message if training fails.

---

### Making Predictions Programmatically

```python
from src.pipline.prediction_pipeline import VehicleData, VehicleDataClassifier

# Create input data
vehicle_data = VehicleData(
    Gender=1,
    Age=25,
    Driving_License=1,
    Region_Code=28.0,
    Previously_Insured=1,
    Annual_Premium=40000.0,
    Policy_Sales_Channel=26.0,
    Vintage=217,
    Vehicle_Age_lt_1_Year=1,
    Vehicle_Age_gt_2_Years=0,
    Vehicle_Damage_Yes=1
)

# Make prediction
classifier = VehicleDataClassifier()
prediction = classifier.predict(vehicle_data)
print(f"Prediction: {prediction}")
```

---

## 🔄 Pipeline Components

### 1. Data Ingestion
**File**: [src/components/data_ingestion.py](src/components/data_ingestion.py)

**Responsibilities**:
- Connects to MongoDB and fetches data
- Saves data to feature store (CSV)
- Splits data into train (75%) and test (25%) sets
- Handles data persistence

**Artifacts Generated**:
- `feature_store/data.csv` - Full dataset
- `ingested/train.csv` - Training data
- `ingested/test.csv` - Testing data

---

### 2. Data Validation
**File**: [src/components/data_validation.py](src/components/data_validation.py)

**Responsibilities**:
- Validates data against schema defined in `config/schema.yaml`
- Checks data types, missing values, and column presence
- Generates validation reports
- Validates numerical and categorical columns separately

**Artifacts Generated**:
- `report.yaml` - Validation report with status

---

### 3. Data Transformation
**File**: [src/components/data_transformation.py](src/components/data_transformation.py)

**Responsibilities**:
- Encodes categorical variables (Gender, Vehicle_Age, Vehicle_Damage)
- Scales numerical features using MinMaxScaler
- Handles class imbalance using SMOTE
- Saves preprocessing objects for inference
- Generates transformed train and test datasets

**Preprocessing Steps**:
1. Label Encoding for categorical features
2. OneHotEncoding for Vehicle_Age and Vehicle_Damage
3. MinMaxScaling for Annual_Premium
4. SMOTE for handling class imbalance

**Artifacts Generated**:
- `transformed/train.npy` - Transformed training data
- `transformed/test.npy` - Transformed testing data
- `transformed_object/preprocessing.pkl` - Preprocessing pipeline

---

### 4. Model Training
**File**: [src/components/model_trainer.py](src/components/model_trainer.py)

**Responsibilities**:
- Trains Random Forest Classifier on transformed data
- Applies hyperparameter optimization
- Evaluates model on training data
- Saves trained model

**Model Configuration**:
```python
RandomForestClassifier(
    n_estimators=20,
    max_depth=10,
    criterion='entropy',
    min_samples_split=7,
    min_samples_leaf=6,
    random_state=101
)
```

**Artifacts Generated**:
- `trained_model/model.pkl` - Trained model object

---

### 5. Model Evaluation
**File**: [src/components/model_evaluation.py](src/components/model_evaluation.py)

**Responsibilities**:
- Evaluates trained model on test set
- Compares with existing production model (if available)
- Calculates performance metrics (accuracy, precision, recall, F1)
- Determines if model meets quality threshold
- Generates evaluation reports

**Evaluation Metrics**:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

**Quality Threshold**: Model must achieve score ≥ 0.6

---

### 6. Model Pusher
**File**: [src/components/model_pusher.py](src/components/model_pusher.py)

**Responsibilities**:
- Uploads approved model to AWS S3
- Creates model registry for versioning
- Makes model available for production inference
- Handles S3 bucket operations

**Cloud Operations**:
- Upload to S3 bucket: `my-model-mlopsproj-rahul-project-mlops`
- Model registry key: `mokdel-registry`

---

## 🤖 Model Details

### Algorithm
**Random Forest Classifier** - An ensemble learning method that constructs multiple decision trees and aggregates their predictions.

### Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| n_estimators | 20 | Number of trees in the forest |
| max_depth | 10 | Maximum depth of each tree |
| criterion | entropy | Function to measure split quality |
| min_samples_split | 7 | Minimum samples required to split a node |
| min_samples_leaf | 6 | Minimum samples required at a leaf node |
| random_state | 101 | Seed for reproducibility |

### Input Features (11 features after preprocessing)
1. **Gender** (Encoded: 0/1)
2. **Age** (Raw numerical)
3. **Driving_License** (Binary: 0/1)
4. **Region_Code** (Numerical)
5. **Previously_Insured** (Binary: 0/1)
6. **Annual_Premium** (Scaled numerical)
7. **Policy_Sales_Channel** (Numerical)
8. **Vintage** (Raw numerical)
9. **Vehicle_Age_lt_1_Year** (Binary: 0/1, one-hot)
10. **Vehicle_Age_gt_2_Years** (Binary: 0/1, one-hot)
11. **Vehicle_Damage_Yes** (Binary: 0/1, one-hot)

### Target Variable
- **Response**: Binary (0: Not Interested, 1: Interested)

### Expected Performance
- **Minimum Expected Score**: 0.6 (60% accuracy)
- **Model Comparison Threshold**: 2% improvement required for deployment

---

## ☁️ Cloud Integration

### AWS S3 Integration

**Purpose**: Store trained models in a centralized registry for production deployment

**Configuration**:
- **Bucket**: `my-model-mlopsproj-rahul-project-mlops`
- **Region**: `us-east-1`
- **Model Registry Key**: `mokdel-registry`

**Files**: 
- [src/configuration/aws_connection.py](src/configuration/aws_connection.py) - AWS connection setup
- [src/cloud_storage/aws_storage.py](src/cloud_storage/aws_storage.py) - S3 operations

**Operations**:
- Upload models to S3
- Download models from S3 for inference
- List available models in registry
- Version management

### MongoDB Integration

**Purpose**: Store and retrieve training data

**Configuration**:
- **Database**: `Proj1`
- **Collection**: `Proj1-Data`

**Files**:
- [src/configuration/mongo_db_connection.py](src/configuration/mongo_db_connection.py) - MongoDB connection
- [src/data_access/proj1_data.py](src/data_access/proj1_data.py) - Data access operations

---

## 🐳 Deployment

### Docker Deployment

Build Docker image:
```bash
docker build -t vehicle-prediction:latest .
```

Run container:
```bash
docker run -p 5000:5000 \
  -e MONGODB_URL=<your-mongodb-url> \
  -e AWS_ACCESS_KEY_ID=<your-aws-key> \
  -e AWS_SECRET_ACCESS_KEY=<your-aws-secret> \
  vehicle-prediction:latest
```

### Local Deployment

1. Install dependencies (see Installation section)
2. Set environment variables
3. Run the application:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 5000
   ```

### Production Considerations
- Use environment-specific configuration
- Implement API authentication and rate limiting
- Set up monitoring and alerting
- Use load balancers for high availability
- Implement CI/CD pipeline
- Set up model monitoring for data drift detection

---

## 📊 Logging & Monitoring

### Logging System

The project uses a comprehensive logging system configured in [src/logger/__init__.py](src/logger/__init__.py)

**Log Levels**:
- `INFO` - General information about pipeline execution
- `WARNING` - Warning messages for potential issues
- `ERROR` - Error messages when operations fail

**Log Locations**:
- Console output during execution
- Timestamped log files in `logs/` directory

### Example Logs
```
INFO - Entered the start_data_ingestion method of TrainPipeline class
INFO - Getting the data from mongodb
INFO - Shape of dataframe: (382111, 12)
INFO - Got the train_set and test_set from mongodb
INFO - Data validation successful
INFO - Data transformation completed
INFO - Model training completed with accuracy: 0.85
INFO - Model evaluated and ready for deployment
```

### Exception Handling

Custom exception handling in [src/exception/__init__.py](src/exception/__init__.py)

- Comprehensive error messages with traceback
- Structured exception logging
- Pipeline halting on critical errors
- Automatic retry logic (where applicable)

---

## 📈 Artifact Management

Artifacts are automatically organized with timestamps:

```
artifact/
├── 03_01_2026_20_15_35/
│   ├── data_ingestion/
│   │   ├── feature_store/
│   │   │   └── data.csv
│   │   └── ingested/
│   │       ├── train.csv
│   │       └── test.csv
│   ├── data_validation/
│   │   └── report.yaml
│   ├── data_transformation/
│   │   ├── transformed/
│   │   │   ├── train.npy
│   │   │   └── test.npy
│   │   └── transformed_object/
│   │       └── preprocessing.pkl
│   └── model_trainer/
│       └── trained_model/
│           └── model.pkl
```

**Benefits**:
- Easy experiment tracking
- Model version history
- Reproducibility of results
- Easy rollback to previous versions

---

## 🧪 Testing

### Experimental Notebooks

- [notebook/exp-notebook.ipynb](notebook/exp-notebook.ipynb) - Model experimentation
- [notebook/mongodb_file.ipynb](notebook/mongodb_file.ipynb) - MongoDB integration testing

### Sample Data

- [notebook/data.csv](notebook/data.csv) - Sample dataset for testing

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Create a new branch for your feature: `git checkout -b feature/your-feature-name`
2. Make your changes and commit: `git commit -m "Add your feature"`
3. Push to the branch: `git push origin feature/your-feature-name`
4. Open a pull request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to functions and classes
- Include error handling and logging
- Write unit tests for new features

---

## 📝 License

This project is open source and available under the MIT License.

---

## 👤 Author

**Rahul Bisht**
- Email: rahulbisht1301@gmail.com
- Project: MLOps Vehicle Insurance Response Prediction

---

## 🆘 Troubleshooting

### Common Issues

1. **MongoDB Connection Error**
   - Verify `MONGODB_URL` environment variable
   - Ensure MongoDB instance is running
   - Check network connectivity

2. **AWS S3 Access Error**
   - Verify AWS credentials
   - Check S3 bucket permissions
   - Ensure bucket name is correct

3. **Model Not Found in Prediction**
   - Run training pipeline first: `GET /train`
   - Check S3 model registry
   - Verify model evaluation passed threshold

4. **Memory Issues with Large Datasets**
   - Reduce batch size in data transformation
   - Use data sampling for initial testing
   - Consider distributed processing with Spark

---

## 📚 References

- [scikit-learn Documentation](https://scikit-learn.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MongoDB Documentation](https://docs.mongodb.com/)
- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)
- [MLOps Best Practices](https://ml-ops.systems/)

---

**Last Updated**: May 7, 2026  
**Project Status**: Active Development
