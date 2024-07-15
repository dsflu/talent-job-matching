example_talent = {
    "talent_id": "talent_1",
    "languages": [
        {"rating": "C2", "title": "German"},
        {"rating": "B2", "title": "English"},
        {"rating": "B1", "title": "French"}
    ],
    "job_roles": [
        "business-analyst",
        "marketing-team-lead",
        "customer-success-manager",
        "business-development-manager",
        "copywriter"
    ],
    "seniority": "senior",
    "salary_expectation": 45000,
    "degree": "apprenticeship"
}

example_talent_2 = {
    "talent_id": "talent_2",
    "languages": [
        {"rating": "B1", "title": "German"},
        {"rating": "C1", "title": "English"}
    ],
    "job_roles": [
        "database-administrator",
        "social-media-marketing-manager",
        "java-developer",
        "tech-lead",
        "product-manager"
    ],
    "seniority": "none",
    "salary_expectation": 60000,
    "degree": "bachelor"
}

example_job = {
    "job_id": "job_1",
    "languages": [
        {"title": "German", "rating": "C1", "must_have": True}
    ],
    "job_roles": ["customer-success-manager"],
    "seniorities": ["none", "junior", "midlevel", "senior"],
    "max_salary": 50000,
    "min_degree": "apprenticeship"
}

example_job_2 = {
    "job_id": "job_2",
    "languages": [
        {"title": "German", "rating": "B1", "must_have": True}
    ],
    "job_roles": ["full-stack-developer", "java-developer"],
    "seniorities": ["midlevel", "junior", "senior"],
    "max_salary": 82000,
    "min_degree": "bachelor"
}

example_criteria = {
    "salary_expectation": 45000,
    "seniority": "junior"
}