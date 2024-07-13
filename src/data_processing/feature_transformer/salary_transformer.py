import pandas as pd
from src.config.feature_settings import salary_bins, salary_labels
from src.data_processing.feature_transformer.base_transformer import BaseTransformer

class SalaryTransformer(BaseTransformer):
    """
    Salary-related transformations.
    """

    def transform_salary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(
            salary_match_binary=lambda df: (df['talent.salary_expectation'] <= df['job.max_salary']).astype(int),
            salary_diff=lambda df: df['job.max_salary'] - df['talent.salary_expectation']
        )
        return df

    def bin_salary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(
            talent_salary_bin=lambda df: pd.cut(df['talent.salary_expectation'], bins=salary_bins, labels=salary_labels, right=False),
            job_max_salary_bin=lambda df: pd.cut(df['job.max_salary'], bins=salary_bins, labels=salary_labels, right=False)
        )
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.apply_transformation(df, self.transform_salary_features)
        df = self.apply_transformation(df, self.bin_salary_features)
        return df