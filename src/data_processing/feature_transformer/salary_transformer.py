import pandas as pd
import numpy as np
from src.config.feature_settings import salary_bins, salary_labels
from src.data_processing.feature_transformer.base_transformer import BaseTransformer

class SalaryTransformer(BaseTransformer):
    """
    Salary-related transformations.
    """

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in salary by filling them with 0
        """
        df['job.max_salary'] = df['job.max_salary'].fillna(0)
        df['talent.salary_expectation'] = df['talent.salary_expectation'].fillna(0)
        return df

    def transform_salary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Two salary feature transformations are made here:
        1. salary_match_binary: see whether talent's expected salary is below the max salary from the job side
        2. salary_diff: the +- difference percentage of them

        Args:
            df: DataFrame with the original features to be transformed

        Returns:
            DataFrame with added salary-related features.
        """
        df = df.assign(
            salary_match_binary=lambda df: (df['talent.salary_expectation'] <= df['job.max_salary']).astype(int),
            salary_diff=lambda df: np.where(
                df['job.max_salary'] == 0,
                0,
                round((df['job.max_salary'] - df['talent.salary_expectation']) / df['job.max_salary'], 2)
            )
        )
        return df

    def bin_salary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bin salary features into predefined categories for less noise.

        Args:
            df: DataFrame with the original features to be transformed

        Returns:
            DataFrame with binned salary features.
        """
        df = df.assign(
            talent_salary_bin=lambda df: pd.cut(df['talent.salary_expectation'], bins=salary_bins, labels=salary_labels, right=False),
            job_max_salary_bin=lambda df: pd.cut(df['job.max_salary'], bins=salary_bins, labels=salary_labels, right=False)
        )
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all transformations to the DataFrame.
        """
        df = self.apply_transformation(df, self.handle_missing_values)
        df = self.apply_transformation(df, self.transform_salary_features)
        df = self.apply_transformation(df, self.bin_salary_features)
        return df