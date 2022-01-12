import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# SKLEARN
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import Bunch
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    accuracy_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    plot_roc_curve)
from sklearn import set_config

set_config(display="diagram")

# FAIRLEARN 

from fairlearn.metrics import (
    MetricFrame,
    true_positive_rate,
    false_positive_rate,
    false_negative_rate,
    selection_rate,
    count,
    false_negative_rate_difference
)

from fairlearn.postprocessing import ThresholdOptimizer, plot_threshold_optimizer
from fairlearn.postprocessing._interpolated_thresholder import InterpolatedThresholder
from fairlearn.postprocessing._threshold_operation import ThresholdOperation
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds, TruePositiveRateParity

# Model Card Toolkit works in Google Colab, but it does not work on all local environments
# that we tested. If the import fails, define a dummy function in place of the function
# for saving figures into images in a model card..

try:
    from model_card_toolkit import ModelCardToolkit
    from model_card_toolkit.utils.graphics import figure_to_base64str
    model_card_imported = True
except Exception:
    model_card_imported = False
    def figure_to_base64str(*args):
        return None

categorical_features = [
    "race",
    "gender",
    "age",
    "discharge_disposition_id",
    "admission_source_id",
    "medical_specialty",
    "primary_diagnosis",
    "max_glu_serum",
    "A1Cresult",
    "insulin",
    "change",
    "diabetesMed",
    "readmitted"
]


special_cols = {'time_in_hospital', 'num_lab_procedures','num_procedures', 'num_medications', 'number_diagnoses'}


def plot_pointplot(df, metric):
    """
        This function generates three pointplots
        showing distribution of people who used emergency services
        
        Parameters
        ----------
            df (dataframe)
        
        Returns
        -------
            None
    """
    # Intialize figure
    fig, axes = plt.subplots(1, 4, figsize=(18, 8), sharey=True)

    # Generate plot
    sns.pointplot(ax=axes[0],y="had_inpatient_days", x="readmit_30_days",
                      data=df, dodge=True,join=False);
    axes[0].grid(True)
    axes[0].set_title("Rate of non emergency visits")
    
    sns.pointplot(ax=axes[1],y="had_inpatient_days", x="readmit_30_days", hue=metric, data=df,
            kind="point", ci=95, dodge=True, join=False);
    axes[1].set_title(f'Rate of non emergency visits (by {metric})');
    axes[1].grid(True)

    sns.pointplot(ax=axes[2],y="had_emergency", x="readmit_30_days",
                      data=df, ci=95,dodge=True, join=False);
    axes[2].set_title("Rate of emegergency visits")
    axes[2].grid(True)

    sns.pointplot(ax=axes[3],y="had_emergency", x="readmit_30_days", hue=metric, data=df,
                kind="point", ci=95, dodge=True, join=False);
    axes[3].grid(True)
    axes[3].set_title(f'Rate of emegergency visits (by {metric})');
    
   
    plt.show()
    
def resample_dataset(X_train, Y_train, A_train):
    """
    Resample dataset to account for balance in model
    
    Parameters:
    -----------
        X_train (unbalanced)
        Y_train (unbalanced)
        A_train (unbalanced)
        
    Returns:
    --------
        X_train (balanced)
        Y_train (balanced)
        A_train (balanced)
    """
    negative_ids = Y_train[Y_train == 0].index
    positive_ids = Y_train[Y_train == 1].index
    balanced_ids = positive_ids.union(np.random.choice(a=negative_ids, size=len(positive_ids)))

    X_train = X_train.loc[balanced_ids, :]
    Y_train = Y_train.loc[balanced_ids]
    A_train = A_train.loc[balanced_ids, :]
    
    return X_train, Y_train, A_train
    
    
def train_model(df):
    
    df_c = df.copy()
    
    # Set random seed
    
    random_seed = 445
    np.random.seed(random_seed)
    
    # Set target variable, demographic and data sensitivity
    target_variable = "readmit_30_days"
    demographic = ["race", "gender"]
    sensitive = ["race"]
    
    Y, A = df_c.loc[:, target_variable], df.loc[:, sensitive]
    
    # We next drop the features that we don't want to use in 
        # our model and expand the categorical features into 0/1 indicators ("dummies").
    X = pd.get_dummies(df_c.drop(columns=[
            "race",
            "discharge_disposition_id",
            "readmitted",
            "readmit_binary",
            "readmit_30_days"
        ]))

    display(X)
    
    ## Split data intro training and testing data
    
    X_train, X_test, Y_train, Y_test, A_train, A_test, df_train, df_test = train_test_split(
                                                                            X,
                                                                            Y,
                                                                            A,
                                                                            df,
                                                                            test_size=0.50,
                                                                            stratify=Y,
                                                                            random_state=random_seed)
    
    
    return [X_train, X_test, Y_train, Y_test, A_train, A_test, df_train, df_test, X]


def plot_descriptive_stats(A_train_bal, Y_train_bal, A_test, Y_test):
    # Intialize figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

    # Generate plot for sensitive attributes
    sns.countplot(ax=axes[0],x="race", data=A_train_bal)
    axes[0].grid(True)
    axes[0].set_title("Sensitive Attributes for Training Dataset")
    axes[0].tick_params(labelrotation=45)
    sensitive_train = figure_to_base64str(axes[0])

    # Generte plot target label
    sns.countplot(ax=axes[1],x=Y_train_bal)
    axes[1].grid(True)
    axes[1].set_title("Target Label Histogram for Training Dataset")
    axes[1].tick_params(labelrotation=45)
    outcome_train = figure_to_base64str(axes[1])

    # Senstive attributes
    sns.countplot(ax=axes[2],x="race", data=A_test)
    axes[2].grid(True)
    axes[2].set_title("Sensitive Attributes for Testing Dataset")
    axes[2].tick_params(labelrotation=45)
    sensitive_test = figure_to_base64str(axes[2])

    # Target histogram test dataset
    sns.countplot(ax=axes[3],x=Y_test)
    axes[3].grid(True)
    axes[3].set_title("Target Label Histogram for Test Dataset")
    axes[3].tick_params(labelrotation=45)
    outcome_test = figure_to_base64str(axes[3])
    
    return [sensitive_train, outcome_train, sensitive_test, outcome_test]

if __name__ == '__main__':
    
    print("Loading data")
    df = pd.read_csv("https://raw.githubusercontent.com/fairlearn/talks/main/2021_scipy_tutorial/data/diabetic_preprocessed.csv")
    display(df.head())
    display(df.info())
    
    print("Assigning category-based columns")
    # Show the values of all binary and categorical features
    categorical_values = {}
    for col in df:
        if col not in special_cols:
            categorical_values[col] = pd.Series(df[col].value_counts().index.values)
   
    categorical_values_df = pd.DataFrame(categorical_values).fillna('')
    categorical_values_df.T
    
    # Generate categorical-based columns
    for col_name in categorical_features:
        df[col_name] = df[col_name].astype("category")