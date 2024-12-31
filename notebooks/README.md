### Task 1- EDA & Stats
This repository contains the code and analysis for insurance datasets.The project focuses on Perform Exploratory Data Analysis (EDA) and stats to provide insights to guide business decisions.

## Project Overview

# 1. Repository Setup
- Create a GitHub repository with a dedicated branch: task-1.
- Committ progress at least three times a day with descriptive messages.
# 2. EDA & Stats
The objective of this phase is to provide a comprehensive analysis of Insurance Data. Key tasks include: 
   - Data Summarization:
       - Calculate the variability for numerical features such as   TotalPremium, TotalClaim
   - Data Quality Assessment: check missing Value
   - Univariate Analysis: Visualize the distribution of variables
   - Bivariate or Multivariate Analysis:Explore relationships between two variables.
   - Data Comparison:  Examine the variation in Geographical Trends
   - Outlier Detection: identify and Detect outlier 

# How to Use
- Clone the Repository:
git clone https://github.com/hanaDemma/Insurance-data-Analysis-week-3.

- Switch to Task-1 Branch:

git checkout task-1

- Run the Notebook:

    - Install dependencies:

        pip install -r requirements.txt

    - Open and execute the Jupyter notebook.

# Key Files
- insurance_analysis.ipynb: Contains the analysis and visualizations
- requirements.txt: List of required Python libraries.
- README.md: Project documentation.

# Technologies Used
- Libraries:
    - pandas, matplotlib, seaborn: For data manipulation and visualization.

# Task-2 Data Version Control (DVC)
The objective of this phase is to install dvc and push to localstorage

# Development Instructions
- Create a task-2 Branch for development.
- Commit progress regularly with clear and detailed commit messages.
- Merge updates into the main branch via a Pull Request (PR).

# Task-3 A/B Hypothesis Testing
- To accept or reject the null hypothesis by evaluating differences between two groups (A and B) using statistical methods.
- Key Tasks:
 - Select Metrics:

  - Identify relevant metrics that will be measured (e.g., conversion rate, user engagement, revenue).
 - Data Segmentation:

  - Divide the data into two groups (control and experimental) to compare the impact of different variables.
 - Statistical Testing:
  - For Categorical Data: Use chi-squared tests to assess the relationship between categories (e.g., user preferences).
  - For Numerical Data: Use t-tests or z-tests to compare means and evaluate the differences between groups (e.g., average purchase amounts).
- Analyze the p-value:

  - Use the p-value from the statistical test to determine if the observed differences are statistically significant.
  - If the p-value is below the significance level (e.g., 0.05), reject the null hypothesis.
- Analyze and Report:

  - Summarize findings, indicating whether there is sufficient evidence to reject or accept the null hypothesis.
  - Present key insights and the impact of the experiment on the selected metrics.

# Development Instructions
- Create a task-3 Branch for development.
- Commit progress regularly with clear and detailed commit messages.
- Merge updates into the main branch via a Pull Request (PR).


# Task-4 Statistical Modeling
- To identify patterns, relationships, and trends within datasets to enhance decision-making through effective statistical modeling.
 - Key tasks
      - Data Preparation
          - Handling Missing Data
          - Feature Engineering
          - Encoding Categorical Data
          - Train-Test Split

      - Modeling Techniques
            - Linear Regression
            - Decision Trees
            - Random Forests
            - Gradient Boosting Machines (GBMs): XGBoost
      - Model Building
      - Model Evaluation
      - Feature Importance Analysis


# Development Instructions
- Create a task-4 Branch for development.
- Commit progress regularly with clear and detailed commit messages.
- Merge updates into the main branch via a Pull Request (PR).