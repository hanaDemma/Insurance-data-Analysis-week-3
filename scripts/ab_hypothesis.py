import pandas as  pd
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind
import matplotlib.pyplot as plt

def ABhypothesisTesting(insurance_data,feature,metric1,metric2,kpi):
    group_a = insurance_data[insurance_data[feature] == metric1][kpi]
    group_b = insurance_data[insurance_data[feature] == metric2][kpi]

    # Perform an independent t-test
    t_stat, p_value = stats.ttest_ind(group_a.dropna(), group_b.dropna(),equal_var=False, nan_policy='omit')

    # Print the results
    print(f"T-statistic of {feature} values {metric1} and {metric2}: {t_stat}")
    print(f"P-value of {feature} values {metric1} and {metric2}: {p_value}")

    # Interpret the results
    alpha = 0.05  # significance level
    if p_value < alpha:
        print("Reject the null hypothesis: There is a significant difference between the groups.")
    else:
        print("Fail to reject the null hypothesis: No significant difference between the groups.")


def hypothesis_test_difference_between_columns(df, kpi_column, group_column):
  
    group_codes = df[group_column].unique()
   
    column_groups = [df[df[group_column] == group_code][kpi_column].dropna() for group_code in group_codes]

    t_stat, p_value = stats.f_oneway(*column_groups)
    print(f"T-statistic of {group_column}: {t_stat}")
    print(f"P-value of {group_column}: {p_value}")
     # Interpret the results
    alpha = 0.05  
    if p_value < alpha:
        print("Reject the null hypothesis: There is a significant difference between the groups.")
    else:
        print("Fail to reject the null hypothesis: No significant difference between the groups.")

def chi_squared_test(df, categorical_column1, categorical_column2):
    """
    Performs a chi-squared test to determine if there's a significant association between two categorical variables.

    Args:
        df: The pandas DataFrame containing the data.
        categorical_column1: The first categorical column.
        categorical_column2: The second categorical column.

    Returns:
        chi2: The chi-squared test statistic.
        p_value: The p-value associated with the chi-squared test.
    """

    # Create a contingency table
    contingency_table = pd.crosstab(df[categorical_column1], df[categorical_column2])

    # Perform the chi-squared test
    chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
    print(f"Chi-squared statistic of {categorical_column1} and {categorical_column2}:", chi2)
    print("P-value:", p_value)
    
    alpha = 0.05  # significance level
    if p_value < alpha:
        print("Reject the null hypothesis: There is a significant difference between the groups.")
    else:
        print("Fail to reject the null hypothesis: No significant difference between the groups.")