# talent-job-matching

This project contains the solution for a tech case assignment.

## Getting Started

### Prerequisites

Make sure you have the following installed on your system:

- I am using Python 3.11 for this project but in general Python 3.7+ can be fine
- Virtual environment manager (optional, but recommended, e.g., virtualenvwrapper)

### Clone the Repository

```bash
git clone https://github.com/dsflu/talent-job-matching.git
cd talent-job-matching
```

### Set Up Virtual Environment

```bash
mkvirtualenv talent-job-matching
workon talent-job-matching
```

or use your own way like `python3 -m venv env`

### Install Dependencies

under root path:

```bash
pip install -r requirements.txt
```

### Change Raw Data Path
Under `src/config/data_config.yaml`, change `raw_data_path` to your actual local raw data path to `data.json`.

## Project Structure

```plaintext
TALENT-JOB-MATCHING/
├── artifacts/                                # Directory for storing various artifacts
│   ├── api_request_examples/                 # Example requests for the API, you can copy paste those to swagger UI directly
│   ├── model_evaluations/                    # Evaluation restuls for different models
│   ├── trained_models/                       # Trained models
├── notebooks/                                # Jupyter notebooks for data exploration and analysis
│   ├── data_insights.ipynb
├── processed_data/                           # Processed datasets for training and testing
├── src/                                      # Source code directory
│   ├── api/                                  # FastAPI related modules
│   │   ├── api.py                            # FastAPI main app
│   │   ├── examples.py                       # Example data for API requests
│   │   ├── schemas.py                        # Pydantic models for API requests
│   │   ├── search.py                         # Search class implementation for different endpoints
│   ├── config/                               # Configuration files and settings
│   │   ├── data_config.yaml                  # Data configuration for raw data path and processed data save path
│   │   ├── feature_settings.py               # Feature settings for transformations
│   │   ├── model_logistic_regression.yaml    # Logistic regression model configuration
│   │   ├── model_random_forest.yaml          # Random forest model configuration
│   │   ├── model_rule_based.yaml             # Rule-based model configuration
│   ├── data_processing/                      # Data loading and featire transformation modules
│   │   ├── data_loader.py                    # Data loading classes
│   │   ├── feature_transformer/              # Feature transformation classes
│   │       ├── base_transformer.py
│   │       ├── degree_transformer.py
│   │       ├── job_roles_transformer.py
│   │       ├── language_transformer.py
│   │       ├── main_transformer.py
│   │       ├── salary_transformer.py
│   │       ├── seniority_transformer.py
│   ├── evaluation/                           # Model evaluation modules
│   │   ├── evaluator.py
│   ├── models/                               # Machine learning models
│   │   ├── base_model.py                     
│   │   ├── logistic_regression_model.py      
│   │   ├── random_forest_model.py            
│   │   ├── rule_based_model.py               # A special Rule-based model
│   ├── pipelines/                            # Data preparation, training, and evaluation pipelines
│   │   ├── data_preparation.py               # Data preparation pipeline
│   │   ├── eval_model.py                     # Model evaluation pipeline
│   │   ├── train_model.py                    # Model training pipeline
│   ├── utils/                                # Utility modules
│   │   ├── config_utils.py                   # Configuration file loading utils
│   │   ├── log.py                            # Logging utils
├── README.md                                 
├── requirements.txt                          # Python dependencies
├── run_api.py                                # Script to run the API server
├── run_data_preparation.py                   # Script to run data preparation
├── run_train_eval_model.py                   # Script to run training and evaluation
├── setup.py                                  
```

## Looking into Tasks Step by Step
### Part 0: Data Exploration
This step is not requested by the task, but I always find it useful to first look into the data to:
1. do some basic feature/label cleaning or analysis, or sanity checkings in general
2. do some simple feature transformations and check their correlations with the label (this will give us the rough impressions how good those features can be and some of the feature transformation methods can be reused later on)

You can find the data exploration notebook at `notebooks/data_insights.ipynb`.

To summarize what I did inside the notebook:
1. Sanity checking on original features (min, max, missing values) and label distributions.
2. Simple feature transformations for salary, degree, seniorities, roles, language based on the matching between talent and job requirements.
3. For all those transformed features, check their correlations (linear) with the labels.

Main findings:
1. The dataset itself is generally clean and balanced with only a few missing values on some features like seniority and degree.
2. The transformed matching features have high linear corrlations with the label, which means that a linear model like logistic regression can be a good start to try.
3. But at the same time, we also see the high linear correlations from the transformed features themselves, which needs to be careful because of the potential multicollinearity problems.

### Part 1: Feature Transformation and Model Training
to be added

### Part 2: Search & Ranking API
to be added

## Minor Improvement Ideas (if restrictions relaxed)
1. If we can use extermal ML libs: to better match roles from talent and job side, fine tuned embeddings can be used to better catch the semantic meanings instead of simply doing look-up.
2. If we can link external datasets like ESCO or ISCO taxonomy: still to better match roles from talent and job side (after normalization), but then we can have job hierarchical relationships and the corresponding skill information as well (which can help us do better matching based on skills).