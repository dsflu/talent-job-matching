import pandas as pd
from src.data_processing.feature_transformer.base_transformer import BaseTransformer

class JobRolesTransformer(BaseTransformer):
    """
    Job roles-related transformations.
    """

    @staticmethod
    def job_roles_match(talent_roles, job_roles):
        """
        Check if there are intersections between talent and job roles.
        """
        return int(bool(set(talent_roles) & set(job_roles)))
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in roles by filling them with ['none']
        """
        df['talent.job_roles'] = df['talent.job_roles'].apply(lambda x: ['none'] if x is None else x)
        df['job.job_roles'] = df['job.job_roles'].apply(lambda x: ['none'] if x is None else x)
        return df

    def transform_job_roles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        One role feature transformation is made here:
        1. job_roles_match_binary: see if any of the job roles in the job is in talent's interested job roles.

        Args:
            df: DataFrame with the original features to be transformed

        Returns:
            DataFrame with added role-related features.
        """
        df = df.assign(
            job_roles_match_binary=lambda df: df.apply(
                lambda row: self.job_roles_match(row['talent.job_roles'], row['job.job_roles']), axis=1
            )
        )
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all transformations to the DataFrame.
        """
        df = self.apply_transformation(df, self.handle_missing_values)
        df = self.apply_transformation(df, self.transform_job_roles)
        return df