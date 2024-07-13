# Salary bins and labels for categorization
salary_bins = [0, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, float('inf')]
salary_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Degree mapping
degree_mapping = {
    'none': 0,
    'apprenticeship': 1,
    'bachelor': 2,
    'master': 3,
    'doctorate': 4
}

# Seniority mapping
seniority_mapping = {
    'none': 0,
    'junior': 1,
    'midlevel': 2,
    'senior': 3
}

# Rating mapping for languages
rating_mapping = {
    'A1': 1,
    'A2': 2,
    'B1': 3,
    'B2': 4,
    'C1': 5,
    'C2': 6
}

# Selected features for later model training
selected_features = [
    'salary_match_binary', 'salary_diff', 'talent_salary_bin', 'job_max_salary_bin', 
    'degree_match_binary', 'seniority_match_binary', 'seniority_exceed_binary', 
    'job_roles_match_binary', 'language_must_have_match_binary', 'language_good2have_count'
]

# Label column name
label_column = 'label'