
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
from sklearn import metrics

import pickle

#####################################################################################
#####################################################################################



####################
### Loading Data ###
####################

preprocessed_data = pd.read_csv("preprocessed_data.csv")
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
cv_results.to_csv("cross_validation_results.csv", index=True)

# Showing the final results
print(cv_results)


#######################################
### Choosing Classifier to Evaluate ###
#######################################

# classifier = models[2][1]
    
classifier = models[4][1]

classifier.fit(normalized_X_train, y_train)


############################################################
### Getting Train and Test Accuracy of the Choosen Model ###
############################################################

train_accuracy = classifier.score(normalized_X_train, y_train)

test_accuracy = classifier.score(normalized_X_test, y_test)


#############################
### Evaluating Classifier ###
#############################

y_pred = classifier.predict(normalized_X_test)
y_proba = classifier.predict_proba(normalized_X_test)[:, 1]

test_accuracy_score = metrics.accuracy_score(y_test, y_pred)

precision = metrics.precision_score(y_test, y_pred)

recall = metrics.recall_score(y_test, y_pred)

f1_score = metrics.f1_score(y_test, y_pred)

# Write scores to a text file
with open("metrics.txt", 'w') as output_text_file:
        
        output_text_file.write(f"Training Accuracy variance explained: {round(train_accuracy, 4)}\n")
        
        output_text_file.write(f"Test Accuracy variance explained: {round(test_accuracy_score, 4)}\n")
        
        output_text_file.write(f"Precision Score: {round(precision, 4)}\n")
        
        output_text_file.write(f"Recall Score: {round(recall, 4)}\n")
        
        output_text_file.write(f"F1 Score: {round(f1_score, 4)}\n")


#################################
### Plotting Confusion Matrix ###
#################################

cm_output = metrics.confusion_matrix(y_test, y_pred)

# Put it into a dataframe for seaborn plot function
cm_df = pd.DataFrame(cm_output)

fig, ax = plt.subplots(figsize=(7, 7), dpi=300)

sns.heatmap(
    cm_df, 
    annot=True, 
    square=True,
    annot_kws={"size": 18}, 
    cmap='Blues', 
    fmt='d', 
    linewidths=2, 
    linecolor="darkorange", 
    cbar=False, 
    xticklabels=[0, 1], 
    yticklabels=[0, 1]
)

plt.title("Confusion Matrix", fontsize=25, pad=20)

plt.xlabel("Predicted Label", fontsize=18, labelpad=3)
plt.xticks(fontsize=15)

plt.ylabel("Actual Label", fontsize=18)
plt.yticks(fontsize=15)

ax.text(2.25, -0.10,'Test Accuracy: '+str(round(test_accuracy_score, 3)), fontsize=14)

ax.text(2.25, 0.0,'Precision: '+str(round(precision, 4)), fontsize=14)

ax.text(2.25, 0.1,'Recall: '+str(round(recall, 4)), fontsize=14)

ax.text(2.25, 0.2,'F1 Score: '+str(round(f1_score, 4)), fontsize=14)

fig.tight_layout()

plt.savefig('confusion_matrix_plot.png', dpi=600, transparent=False)
# plt.show()


##########################
### Plotting ROC_Curve ###
##########################

false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_proba)

roc_score = metrics.roc_auc_score(y_test, y_proba)

fig = plt.figure(figsize=(12, 10), dpi=300)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(false_positive_rate, true_positive_rate, color='darkorange')

plt.fill_between(false_positive_rate, true_positive_rate, alpha=0.2, color='orange')

plt.title(f'ROC Curve - AUC : {round(roc_score, 3)}', fontsize=25, pad=20)

plt.xlabel('False Positive Rate', fontsize=20, labelpad=5)
plt.xticks(fontsize=20)

plt.ylabel('True Positive Rate', fontsize=20, labelpad=5)
plt.yticks(fontsize=20)

plt.grid(color='grey')

fig.tight_layout()

plt.savefig('ROC_Curve.png', dpi=600, transparent=False)
# plt.show()


##########################
### Plotting Residuals ###
##########################

fig = plt.figure(figsize=(7, 7), dpi=300)

y_pred_ = classifier.predict(normalized_X_test) + np.random.normal(0, 0.25, len(y_test))

y_jitter = y_test + np.random.normal(0, 0.25, len(y_test))

residuals_df = pd.DataFrame(list(zip(y_jitter, y_pred_)), columns=["Actual Label", "Predicted Label"])

ax = sns.scatterplot(
    x="Actual Label", 
    y="Predicted Label",
    data=residuals_df,
    facecolor='dodgerblue',
    linewidth=1.5,
)
    
ax.set_xlabel('Actual Label', fontsize=14) 

ax.set_ylabel('Predicted Label', fontsize=14)#ylabel

ax.set_title('Residuals', fontsize=20)

min = residuals_df["Predicted Label"].min()

max = residuals_df["Predicted Label"].max()

ax.plot([min, max], [min, max], color='black', linewidth=1)

plt.tight_layout()

plt.savefig('Residuals_plot.png', dpi=600, transparent=False)
# plt.show()
