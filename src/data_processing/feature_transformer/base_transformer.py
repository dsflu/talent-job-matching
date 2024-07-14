import pandas as pd

class BaseTransformer:
    """
    Base class for feature transformers.
    """

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.
        """
        return df

    def apply_transformation(self, df: pd.DataFrame, transformation_fn) -> pd.DataFrame:
        """
        Apply a transformation function to the DataFrame.
        
        Args:
            df (pd.DataFrame): The DataFrame to transform.
            transformation_fn (callable): The transformation function to apply.
        
        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        return transformation_fn(df)