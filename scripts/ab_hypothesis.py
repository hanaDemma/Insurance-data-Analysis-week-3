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

