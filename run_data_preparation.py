"""
This script prepares the training and testing dataset from the raw json file.
"""
from src.pipelines.data_preparation import prepare_data

# path to the data config file (raw data path and processed data save location)
config_path = 'src/config/data_config.yaml'

if __name__ == "__main__":
    prepare_data(config_path)