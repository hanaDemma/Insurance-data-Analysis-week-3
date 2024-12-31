import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
import shap
import lime
import lime.lime_tabular




def feature_engineering(df):
    """
    Performs feature engineering to create new features relevant to TotalPremium and TotalClaims.

    """
    
    # Step 1: Calculate vehicle age
    current_year = pd.Timestamp.now().year
    df['VehicleAge'] = current_year - df['RegistrationYear']
    
    # Step 2: Create Claims-to-Premium ratio
    df['ClaimsToPremiumRatio'] = df['TotalClaims'] / (df['TotalPremium'] + 1e-5)  # Add small value to avoid division by zero
    
    # Step 3: Create Vehicle Power Index (cubic capacity * kilowatts / cylinders)
    df['VehiclePowerIndex'] = (df['cubiccapacity'] * df['kilowatts']) / (df['Cylinders'] + 1e-5)
    
    # Step 4: Calculate insurance tenure in months (from TransactionMonth to now)
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
    df['InsuranceTenureMonths'] = (pd.Timestamp.now() - df['TransactionMonth']).dt.days / 30
    
    # Step 5: Flag high-risk vehicle types (You can adjust based on domain knowledge)
    high_risk_vehicle_types = ['Taxi', 'Truck', 'Bus']  # Example vehicle types
    df['IsHighRiskVehicle'] = df['VehicleType'].apply(lambda x: 1 if x in high_risk_vehicle_types else 0)
    
    # Step 6: Flag high-risk regions based on historical data (e.g., higher claims)
    high_risk_regions = ['RegionA', 'RegionB']  # Replace with actual high-risk regions
    df['IsHighRiskRegion'] = df['Province'].apply(lambda x: 1 if x in high_risk_regions else 0)

    return df


def encode_categorical_data(df):
    """
    Encodes categorical data using Label Encoding for binary categories
    and One-Hot Encoding for non-binary categories.
    """
    # Columns to apply Label Encoding (for binary and ordinal categories)
    label_encode_columns = ['NewVehicle', 'WrittenOff', 'Rebuilt', 'Converted', 'CrossBorder', 'AlarmImmobiliser', 
                            'TrackingDevice', 'MaritalStatus', 'Gender', 'VehicleType']  # Added 'VehicleType'
    
    # Apply Label Encoding to binary/ordinal columns
    for col in label_encode_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str).fillna('Unknown'))  # Handle missing values as 'Unknown'
    
    # Columns to apply One-Hot Encoding (for nominal categories with multiple values)
    one_hot_encode_columns = ['Citizenship', 'LegalType', 'Title', 'Language', 'Bank', 'AccountType', 'Country', 
                              'MainCrestaZone', 'SubCrestaZone', 'ItemType', 'make', 'Model', 'bodytype', 'TermFrequency', 
                              'ExcessSelected', 'CoverCategory', 'CoverType', 'CoverGroup', 'Section', 'Product', 
                              'StatutoryClass', 'StatutoryRiskType', 'Province']  # Added 'Province' earlier
    
    # Apply One-Hot Encoding to nominal columns
    df = pd.get_dummies(df, columns=one_hot_encode_columns, drop_first=True)
    
    return df

def clean_data_for_modeling(df):
    """
    Clean the dataset by replacing or removing non-numeric values that may prevent the model from training.
    """
    # Replace 'Not specified' and other non-numeric entries with NaN
    df.replace('Not specified', np.nan, inplace=True)
    df.replace('Unknown', np.nan, inplace=True)

    
    # Fill missing values with the median for numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # For categorical columns, you can use mode or drop the rows
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    

    for col in df.columns:
        if df[col].dtype == 'object':  # Check if the column is object type (usually indicates strings)
            df[col] = df[col].str.replace(',', '', regex=False)  # Remove commas from numbers
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, setting invalid parsing as NaN
    
    return df



def convert_datetime_to_numeric(df):
    """
    Converts datetime columns to numeric values.
    Extract useful features from datetime columns.
    """
    # Identify datetime columns
    datetime_columns = df.select_dtypes(include=['datetime', 'datetime64']).columns
    
    for col in datetime_columns:
        # Convert datetime to numeric by extracting useful parts or converting to timestamp
        df[col + '_year'] = df[col].dt.year
        df[col + '_month'] = df[col].dt.month
        df[col + '_day'] = df[col].dt.day
        df[col + '_weekday'] = df[col].dt.weekday
        
        # Optionally, you can remove the original datetime column if not needed
        df.drop(columns=[col], inplace=True)
    
    return df


def train_test_split_data(df, target_column, test_size=0.3, random_state=42):
    """
    Splits the data into training and testing sets for modeling.
    
    """
    
    # Step 1: Separate features (X) and target (y)
    X = df.drop(columns=[target_column])  # Features (all columns except target)
    y = df[target_column]  # Target variable
    
    # Step 2: Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test



def train_linear_regression(X_train, y_train):
    """
    Trains a Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    """
    Trains a Decision Tree model.
    """
    model = DecisionTreeRegressor(max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None):
    """
    Trains a Random Forest model.
    """
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    """
    Trains an XGBoost model.
    """
    model = XGBRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model using Mean Squared Error and R^2 Score.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2
