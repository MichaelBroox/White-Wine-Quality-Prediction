
#####################################################################################
#####################################################################################
# Loading the iconic trio 🔥
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
plt.style.use('fivethirtyeight')

# Importing model_selection to get access to some dope functions like GridSearchCV()
from sklearn import model_selection

from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SVMSMOTE

# Loading models
from sklearn import tree
from sklearn import ensemble
import xgboost
from sklearn import linear_model

# custom
from custom import helper

# Loading evaluation metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import pickle
#####################################################################################
#####################################################################################



####################
### Loading Data ###
####################

preprocessed_data = pd.read_csv("data/preprocessed_data.csv")
# preprocessed_data.head()


##########################################
### Creating a Copy of the Loaded Data ###
##########################################

data_with_targets = preprocessed_data.copy()
# data_with_targets.head()


#################################
### Dropping `quality` Column ###
#################################

data_with_targets = data_with_targets.drop(['quality'], axis=1)
# data_with_targets.head()


###############################################################
### Splitting the Data into Feature Matrix and Target Label ###
###############################################################

target_variable = 'quality_rate'

# Unscaled Features
X = data_with_targets.drop([target_variable], axis=1)

# Target Variable
y = data_with_targets[target_variable]


#####################################################
### SMOTE Sampling to deal with imbalance classes ###
#####################################################

# Setting Seed Value
seed = 81

smote = SVMSMOTE(random_state=seed)

resampled_X, resampled_y = smote.fit_sample(X, y)


##########################################
### Splitting into Train and Test sets ###
##########################################

X_train, X_test, y_train, y_test = model_selection.train_test_split(resampled_X, resampled_y, test_size=0.3, stratify=resampled_y, random_state=seed)


####################################
### Scalling Train and Test sets ###
####################################

column_names = list(X.columns.values)

scaler = StandardScaler()

normalized_X_train = pd.DataFrame(
    scaler.fit_transform(X_train), 
    columns=column_names, 
    index=X_train.index
)


normalized_X_test = pd.DataFrame(
    scaler.transform(X_test), 
    columns=column_names, 
    index=X_test.index
)


#################
### Modelling ###
#################

# Instantiating baseline models
models = [
    ("Decision Tree", tree.DecisionTreeClassifier(random_state=seed)),
    ("Random Forest", ensemble.RandomForestClassifier(random_state=seed)),
    ("AdaBoost", ensemble.AdaBoostClassifier(random_state=seed)),
    ("ExtraTree", ensemble.ExtraTreesClassifier(random_state=seed)),
    ("GradientBoosting", ensemble.GradientBoostingClassifier(random_state=seed)),
    ("XGBOOST", xgboost.XGBClassifier(random_state=seed)),
]

feature_importance_of_models, df_model_features_with_importance, model_summary = helper.baseline_performance(
    models=models, 
    X_train=normalized_X_train, 
    y_train=y_train, 
    X_test=normalized_X_test, 
    y_test=y_test, 
    column_names=list(X.columns.values), 
    csv_path='csv_tables', 
    save_model_summary=True, 
    save_feature_importance=True, 
    save_feature_imp_of_each_model=True,
)


##########################################
### Plotting Train and Test Accuracies ###
##########################################

helper.plot_model_summary(
    model_summary=model_summary, 
    figsize=(20, 14), 
    dpi=600, 
    transparent=True,
    save_visualization=True, 
    figure_name='Train and Test Accuracies', 
    figure_path='figures',
)


###################################
### Plotting Feature Importance ###
###################################

helper.plot_feature_importance(
    df_model_features_with_importance, 
    figsize=(20, 15), 
    dpi=600, 
    transparent=True,
    annotate_fontsize='xx-large',
    save_plot=True,
    path='figures',
)


##############################
### Getting Top 6 Features ###
##############################

top_6_features = list(feature_importance_of_models['GradientBoosting'].head(6))
# top_6_features

normalized_X_train_new = normalized_X_train[top_6_features]
# normalized_X_train_new.head()

normalized_X_test_new = normalized_X_test[top_6_features]
# normalized_X_test_new.head()

# X_train_new, X_test_new, y_train_new, y_test_new = model_selection.train_test_split(X_new, y, test_size=0.2, stratify=y, random_state=80)


############################
### Creating New Folders ###
############################

helper.create_folder('./new_csv_tables/')
helper.create_folder('./new_figures/')


###################################
### Modelling on Top 6 Features ###
###################################

feature_importance_of_models_new, model_features_with_importance_new, model_summary_new = helper.baseline_performance(
    models=models, 
    X_train=normalized_X_train_new, 
    y_train=y_train, 
    X_test=normalized_X_test_new, 
    y_test=y_test, 
    column_names=list(top_6_features), 
    csv_path='new_csv_tables', 
    save_model_summary=True, 
    save_feature_importance=True, 
    save_feature_imp_of_each_model=True
)


##########################################
### Plotting Train and Test Accuracies ###
##########################################

helper.plot_model_summary(
    model_summary=model_summary_new, 
    figsize=(20, 14), 
    dpi=300, 
    transparent=True,
    save_visualization=True, 
    figure_name='Train and Test Accuracies_new', 
    figure_path='new_figures',
)


###################################
### Plotting Feature Importance ###
###################################

helper.plot_feature_importance(
    feature_importance=model_features_with_importance_new, 
    figsize=(20, 14), 
    dpi=600, 
    transparent=True,
    annotate_fontsize='xx-large',
    save_plot=True,
    path='new_figures',
)


###############################
### Cross Validating Models ###
###############################

# Splitting data into 10 folds
cv_kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=150)

scorer = "f1"

# Instantiating model_names as an empty list to keep the names of the models
model_names = []

# Instantiating cv_mean_scores as an empty list to keep the cross validation mean score of each model
cv_mean_scores = []

# Instantiating cv_std_scores as an empty list to keep the cross validation standard deviation score of each model
cv_std_scores = []

# Looping through the baseline models and cross validating each model
for model_name, model in models:
    model_scores = model_selection.cross_val_score(
        model, X, y, cv=cv_kfold, scoring=scorer, n_jobs=-1, verbose=1,
    )
    
    print(
        f"{model_name} Score: %0.2f (+/- %0.2f)"
        % (model_scores.mean(), model_scores.std() * 2)
    )

    # Appending model names to model_name
    model_names.append(model_name)
    
    # Appending cross validation mean score of each model to cv_mean_score
    cv_mean_scores.append(model_scores.mean())
    
    # Appending cross validation standard deviation score of each model to cv_std_score
    cv_std_scores.append(model_scores.std())


# Parsing model_names, cv_mean_scores and cv_std_scores and a pandas DataFrame object
cv_results = pd.DataFrame({"model_name": model_names, "mean_score": cv_mean_scores, "std_score": cv_std_scores})

# Sorting the Dataframe in descending order
cv_results.sort_values("mean_score", ascending=False, inplace=True,)

# Saving the DataFrame as a csv file
cv_results.to_csv("csv_tables/cross_validation_results.csv", index=True)

# Showing the final results
# cv_results


#######################################
### Choosing Classifier to Evaluate ###
#######################################

# classifier = models[2][1]
    
classifier = models[4][1]

classifier.fit(normalized_X_train_new, y_train)


############################################################
### Getting Train and Test Accuracy of the Choosen Model ###
############################################################

train_accuracy = classifier.score(normalized_X_train_new, y_train)

test_accuracy = classifier.score(normalized_X_test_new, y_test)


#############################
### Evaluating Classifier ###
#############################

helper.evaluate_classifier(
    estimator=classifier, 
    X_test=normalized_X_test_new, 
    y_test=y_test,
    save_figure=True,
    figure_path='figures',
    transparent=True, 
    dpi=600,
    cmap="Purples",
)
