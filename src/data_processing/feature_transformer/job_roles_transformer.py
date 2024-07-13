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

    def transform_job_roles(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(
            job_roles_match_binary=lambda df: df.apply(
                lambda row: self.job_roles_match(row['talent.job_roles'], row['job.job_roles']), axis=1
            )
        )
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.apply_transformation(df, self.transform_job_roles)
        return df