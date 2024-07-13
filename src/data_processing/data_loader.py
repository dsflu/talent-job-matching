import json
import pandas as pd
from typing import Optional
from src.utils.log import log

class BaseDataLoader:
    """
    Base class for data loaders.
    """
    def __init__(self, file_path: str):
        """
        Init BaseDataLoader
        
        Args:
            file_path (str): The path to the data file.
        """
        self.file_path = file_path
        self.data = None
        log.info(f"BaseDataLoader initialized with file path: {file_path}")
    
    def load_data(self) -> None:
        """
        Method to load data.
        
        Raises:
            NotImplementedError
        """
        raise NotImplementedError("Method not implemented")
    
    def get_data(self):
        """
        Gets the loaded data.
        """
        if self.data is None:
            log.warning("Data not loaded. Please call load_data() first.")
        return self.data


class JSONDataLoader(BaseDataLoader):
    """
    Data loader for JSON files (and convert the data to pandas dataframe).
    """
    
    def load_data(self) -> None:
        """
        Loads data from a JSON file.
        """
        try:
            with open(self.file_path, 'r') as file:
                self.data = json.load(file)
            log.info("JSON data loaded.")
        except Exception as e:
            log.error(f"Error loading JSON data: {e}")
    
    def to_pandas(self) -> Optional[pd.DataFrame]:
        """
        Converts the loaded JSON data to a Pandas DataFrame.
        
        Returns:
            Optional[pd.DataFrame]: The converted Pandas DataFrame, or None.
        """
        if self.data is None:
            log.warning("Data not loaded. Please call load_data() first.")
            return None
        try:
            df = pd.json_normalize(self.data)
            log.info("Data successfully converted to Pandas DataFrame.")
            return df
        except Exception as e:
            log.error(f"Error converting data to DataFrame: {e}")
            return None