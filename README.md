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

Main findings:
1. The dataset itself is generally clean and balanced with only a few missing values on some features like seniority and degree.
2. The transformed matching features have high linear corrlations with the label, **which means that a linear model like logistic regression can be a good start to try.**
3. But at the same time, we also see the high linear correlations from the transformed features themselves, which needs to be careful because of the potential **multicollinearity** problems.

### Part 1: Feature Transformation and Model Training
For this part, I actually split it into two parts:
1. Training and testing data/feature preparation
2. Model training and evaluation

**which makes more sense to me since in production these two steps are normally separated.**

You can run
```bash
python run_data_preparation.py
```
to create the training and testing data with transformed features.

The feature transformation classes are all under `src/data_processing/feature_transformer`. **I intentionally made them extendable in case we need to add more features transformations in the furture.** All those feature transformation methods are no big deals but the same as what we tried in the data exploration notebooks.

After we create the training and testing datasets under `processed_data`, we can move on with the model training and evaluation by running:
```bash
python run_train_eval_model.py
```
Make sure you have the correct `model_config_path` in the pipeline. The pipeline will read the model yaml config file you provided under `src/config/model_xxx.yaml`.

I created 3 models here for demo:
1. logstic regression
2. random forest
3. rule based model

I don't think I need to explain too much on the first two simple models. I want to explain a few more on the 3rd rule based model:
1. What does this model do?
- Simply checking features: `salary_match_binary`, `degree_match_binary`, `seniority_match_binary`, `job_roles_match_binary`, `language_must_have_match_binary`, it will output 1 if all specified interaction features are 1, otherwise 0.

2. Why I want to include this simple model?
- **Because of the nature of this simple dataset, i think simple filtering can help us do most of the matching correctly. We don't need training for this model and this model can serve as the fall back strategy in production.**
- But we also know that, even if all those requirements are met for talent and job, the job can still be a non-match. That is also why we will see the ML models can perform better than this rule based model later on in evaluations.

For evaluation, I checked the model's performance on both training and testing data, and you can find the evaluation results under `artifacts/model_evaluations/*.json`。

**We can simply see that in general, `random forest` performs slightly better than `logstic regression`, and those two perform better than the `rule based model`.**

### Part 2: Search & Ranking API
For this part, in general I want to create an API to use the trained model to perfrom match, ranking, and filtering requests, so that we can serve this model to other services.

You can run
```bash
python run_api.py
```
to start the server. And check `http://0.0.0.0:8000/docs` for Swagger UI for testing.

I provided 3 endpoints here:
1. `/match`:  This method takes a talent and job as input and uses the trained model to predict the label and score. **This can be useful when a service want to predict the talent-job matching on the fly.**
2. `/match_bulk`: This method takes multiple talents and jobs as input and uses the trained model to predict the label for each combination. **This can be useful when we want to do batch recommendations for all talents and jobs. I also added an additional filter `filter_false_predictions` here to filter out all non-matched jobs from the model (so that the users won't see non-matched jobs).**
3. `/rank_and_filter`: This method ranks and filters job opportunities for a given talent based on model predictions and specified filtering criteria. **This can be used for job recommendation for a given talent on the fly. The `criteria` fitlering can also help to filter out the jobs that are not in the considerations from talent's request.**

To test the endpoiints, you can find the request examples under `artifacts/api_request_examples/*.json`.

## Minor Improvement Ideas (if restrictions relaxed)
1. If we can use extermal ML libs: to better match roles from talent and job side, fine tuned embeddings can be used to better catch the semantic meanings instead of simply doing look-up.
2. If we can link external datasets like ESCO or ISCO taxonomy: still to better match roles from talent and job side (after normalization), but then we can have job hierarchical relationships and the corresponding skill information as well (which can help us do better matching based on skills).