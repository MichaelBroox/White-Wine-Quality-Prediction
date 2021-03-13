######################################################
######################################################
# iconic trio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import pandas_profiling as pp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Loading evaluation metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score

from sklearn import metrics

import os

from IPython.display import display

import matplotlib as mpl
mpl.rcParams.update({'figure.max_open_warning': 100})

######################################################
######################################################



# Set random seed
seed = 42


def create_folder(directory):
    '''
    Function to Create a directory
    
    Parameters
    ----------
    directory: str, name of the directory to be created.
    
    Returns
    -------
    None
        This function doesn't return anything.
    
    Examples
    --------
    # Create a folder in the current working directory
    >>> create_folder('./folder_name/')
    '''
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


def load_csv_data(filename_filepath):
    '''
    Load a csv data
    
    Parameters
    ----------
    filename_filepath : str, path object or file-like object
    Any valid string path is acceptable. The string could be a URL.
    filepath could be: file://localhost/path/to/data.csv.
    filename could be: data.csv
    
    Returns
    -------
    DataFrame
    A comma-separated values (csv) file is returned as two-dimensional
    data structure with labeled axes.

    Examples
    --------
    >>> data = load_csv_data('data.csv')
    
    '''
    dataset = filename_filepath

    try:
        df = pd.read_csv(dataset,  index_col=0)
        print ("Successfully loaded!")
        
        return df
    
    except:
        print ("Something went wrong.")
        
        return None



def missing_data(df):
    '''
    Function to summarise missing values in dataset
    
    Parameters
    ----------
    df : DataFrame where each column in the DataFrame is a variable and
    each row is an observation.
    
    
    Returns
    -------
    DataFrame
        This function prints missing data summary of a DataFrame.
    
    Examples
    --------
    >>> missing_data(df)
    '''
    total = df.isnull().sum()
    percent = (df.isnull().sum() / df.isnull().count() * 100)
    missing_values = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
    types = []
    for column in df.columns:
        dtype = str(df[column].dtype)
        types.append(dtype)
    
    missing_values['Types'] = types
    missing_values.sort_values('Total', ascending=False, inplace=True)
    
    return missing_values.T



def outlier_info(df):
    
    quantile_75 = []
    
    quantile_25 = []
    
    inter_quantile_range = []
    
    minimum_value = []
    
    maximum_value = []
    
    number_of_outliers = []
    
    percentage_of_outliers = []
    
    numeric_df = df.select_dtypes(np.number)
    
    for column in numeric_df.columns:
        
        q75, q25 = np.percentile(df[column], [75, 25])
        quantile_75.append(q75)
        
        quantile_25.append(q25)
        
        iqr = q75 - q25
        inter_quantile_range.append(iqr)
        
        min_value = q25 - (iqr * 1.5)
        minimum_value.append(min_value)
        
        max_value = q75 + (iqr * 1.5)
        maximum_value.append(max_value)
        
        outlier_count = len(np.where((df[column] > max_value) | (df[column] < min_value))[0])
        number_of_outliers.append(outlier_count)
        
        outlier_percent = round(outlier_count/len(df[column])*100, 2)
        percentage_of_outliers.append(outlier_percent)
        
    outliers_summary = pd.DataFrame(
        {
            'Number of Outliers' : number_of_outliers,
            'Outliers Percentage' : percentage_of_outliers,
            '75% Quantile' : quantile_75,
            '25% Quantile' : quantile_25,
            'Inter Quantile Range' : inter_quantile_range,
            'Maximum Value' : maximum_value,
            'Minimum Value' : minimum_value,
        }
    ).T
    
    outliers_summary.columns = numeric_df.columns.values
    
    return outliers_summary



def detect_outliers(
    df, 
    image_name=None, 
    path=None, 
    plot_size=(25,10), 
    xticklabels_fontsize=12, 
    yticklabels_fontsize=12, 
    title_fontsize=22, 
    boxplot_xlabel_fontsize=12, 
    save=False, 
    dpi=300, 
    transparent=False,
):
    '''
    Detect outliers in dataset
    
    Parameters
    ----------
    df : DataFrame where each column in the DataFrame is a variable and
    each row is an observation.
    
    image_name : str, optional (default=None)
    If str, image_name is given to the generated correlation heatmap as the image filename.

    path : str, 

    plot_size : tuple, optional (default=(25, 10))
    If tuple, plot_size is used to set the plot size of the correlation heatmap.
    
    xticklabels_fontsize : int, 
    
    yticklabels_fontsize : int, 
    
    title_fontsize : int, 
    
    boxplot_xlabel_fontsize : int, 
    
    save : bool, optional (default=False)
    If bool, save determines whether to save the generated correlation heatmap or not.
    
    dpi : int, optional (default=200)
    If bool, dpi is used to set the image resolution of the correlation heatmap.
    
    transparent : bool, optional (default=False)
    If bool, transparent is used to set the image background to transparent or not.
    
    
    Returns
    -------
    None
        This function plots a histogram and a boxplot and returns None.
    
    Examples
    --------
    >>> detect_outliers(df, image_name='outliers', plot_size=(25,10), save=True, dpi=300, transparent=True)
    
    '''
    for i, column in enumerate(df.columns):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), clear=False)
        
        
        df[column].hist(bins=30, ax=ax1)
        
        ax = sns.boxplot(x=column, data=df, ax=ax2)
        ax = sns.stripplot(x=column, data=df, color="maroon", jitter=0.2, size=2.5)
        
        
        ax1.set_title('Distribution of ' + column, fontsize=title_fontsize)
        ax2.set_title('Boxplot of ' + column, fontsize=title_fontsize)
        
        
        
        plt.setp(ax1.get_xticklabels(), fontsize=xticklabels_fontsize)
        plt.setp(ax1.get_yticklabels(), fontsize=yticklabels_fontsize)
        
        plt.setp(ax2.get_xticklabels(), fontsize=xticklabels_fontsize)
        ax2.set_xlabel(ax2.get_xlabel(), fontsize=boxplot_xlabel_fontsize)
        
        plt.grid(b=True, axis='both', color='black', linewidth=0.5)
        
        fig.tight_layout()
        
        if save == False:
            plt.show()
        else:
            plt.savefig(path+'/'+image_name+str(i), dpi=dpi, transparent=transparent)
            # plt.show()



def outlier_remover(df):
    '''
    Function to remove outliers from data set
    
    Parameters
    ----------
    df : DataFrame where each column in the DataFrame is a variable and
    each row is an observation.
    
    Returns
    -------
    DataFrame 
        This function returns a DataFrame where each column in the DataFrame is a variable and
        each row is an observation.
    
    Examples
    --------
    >>> df = outlier_remover(df)
    '''
    # 1st quantile
    Q1 = df.quantile(0.5)
    
    # 3rd quantile
    Q3 = df.quantile(0.75)
    
    # Inter-quantile range
    IQR = Q3 - Q1
    
    # Minimum
    lower_bound = Q1 - (IQR * 1.5)
    
    # Maximum
    upper_bound = Q3 + (IQR * 1.5) 
    
    # Printing out 1st and 3rd quantiles, Inter-quantile range, Maximum and Minimum
    print(pd.concat([Q1, Q3, IQR, lower_bound, upper_bound], axis=1, keys=['Q1', 'Q3', 'IQR', 'lower_bound', 'upper_bound']))
    
    # Querying outliers
    outlier_data_points = (df < lower_bound) | (df > upper_bound)
    
    # Removing outliers from data
    df = df[~(outlier_data_points).any(axis = 1)]
    
    return df



def correlation_viz(
    df, 
    image_name=None, 
    path=None, 
    plot_size=(25, 10), 
    cor_bar_orient='horizontal', 
    colormap='YlGnBu', 
    save=False, 
    dpi=300, 
    transparent=False,
):
    '''
    Function to generate correlation heatmap of data features.
    
    Parameters
    ----------
    df : DataFrame, where each column in the DataFrame is a variable and
    each row is an observation.
    
    image_name : str, optional (default=None)
    If str, image_name is given to the generated correlation heatmap as the filename.

    path: str, optional (default=None)

    plot_size : tuple, optional (default=(25, 10))
    If tuple, plot_size is used to set the plot size of the correlation heatmap.
    
    cor_bar_orient : str, optional (default=horizontal)
    
    colormap : str, optional (default=YlGnBu)
    
    save : bool, optional (default=False)
    If bool, save determines whether to save the generated correlation heatmap or not.
    
    dpi : int, optional (default=300)
    If bool, dpi is used to set the image resolution of the correlation heatmap.
    
    transparent : bool, optional (default=False)
    If bool, transparent is used to set the image background to transparent or not.

    Returns
    -------
    None
        This function plots a correlation heatmap of data features and returns None.
    
    Examples
    --------
    >>> correlation_viz(df, image_name="correlation_heatmap", plot_size=(20,10), save=True, dpi=300, transparent=True)
    
    '''
    sns.set(font_scale=1.8)
    
    if save == False:
        
        plt.figure(figsize=plot_size, dpi=dpi)
        
        correlation = sns.heatmap(
            df.corr(), 
            annot=True, 
            annot_kws={"size": 20}, 
            cbar_kws={
                'label':'correlation bar', 
                'orientation':cor_bar_orient, 
                'shrink':0.4,
            }, 
            square=True, 
            linewidths=2, 
            linecolor='maroon', 
            cmap=colormap,
        )
        
        plt.show()
        
    else:
        
        plt.figure(figsize=plot_size, dpi=dpi)
        correlation = sns.heatmap(
            df.corr(), 
            annot=True, 
            annot_kws={"size": 20}, 
            cbar_kws={
                'label':'correlation bar', 
                'orientation':cor_bar_orient, 
                'shrink':0.4,
            }, 
            square=True, 
            linewidths=2, 
            linecolor='maroon', 
            cmap=colormap,
        )
        
        plt.savefig(path+'/'+image_name, dpi=dpi, transparent=transparent)
        # plt.show()



def data_profile(df, path=None):
    '''
    Funtion to get data report on the dataset
    
    Parameters
    ----------
    df : DataFrame where each column in the DataFrame is a variable and
    each row is an observation.

    path : str, 
    
    Returns
    -------
    An html data report of the dataset
    
    '''
    report = pp.ProfileReport(df)
    
    return report.to_file(path+'/'+'data_profile.html')



def scale(X_train, X_test, X_val=None, scale_validation=False, scaler_type=MinMaxScaler()):
    '''
    Function to normalize dataset
    
    Parameters
    ----------
    X_train : array-like, sparse matrix of shape (n_rows, n_features)
    Training data
    
    X_test : array-like, sparse matrix of shape (n_rows, n_features)
    Testing data
    
    X_val : array-like, sparse matrix of shape (n_rows, n_features) optional (default=None)
    Validation data
    
    scale_validation : bool, optional (default=False)
    If bool, X_val will be scaled.
    
    scaler_type : estimator object, optional (default=MinMaxScaler()). This estimator scales and translates each feature individually
    implementing 'fit', 'transform' or 'fit_transform' to learn and scale the data.
    
    Returns
    -------
    X_train : array-like, sparse matrix of shape (n_rows, n_features)
    Training data
    
    X_test : array-like, sparse matrix of shape (n_rows, n_features)
    Testing data
    
    X_val : array-like, sparse matrix of shape (n_rows, n_features)
    Validation data
    
    Examples
    --------
    >>> X_train, X_test = scale(X_train, X_test, MinMaxScaler())
    
    >>> X_train, X_test = scale(X_train, X_test, StandardScaler())
    
    >>> X_train, X_test = scale(X_train, X_test)
    
    >>> X_train, X_test, X_val = scale(X_train, X_test, X_val, scale_validation=True)
    
    '''
    
    scaler = scaler_type
    
    if scale_validation == False:
        
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test
    
    else:
        
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_val = scaler.transform(X_val)

        return X_train, X_test, X_val
    


def split_data(df, target_variable, test_size=0.2, random_state=seed, validation_data=False):
    '''
    Function to split dataset into X_train, X_test, X_val, y_train, y_test and y_val
    
    Parameters
    ----------
    df : DataFrame where each column in the DataFrame is a variable and
    each row is an observation.
    
    target_variable : str, column name of the output variable.
    Target variable(s) or label
    
    test_size : float, int, or optional (default=0.2)
    If float, should be between 0.0 and 1.0 and represent the proportion
    of the dataset to use in spliting the dataset in test split. If int, represents the
    absolute number of test samples.  
    
    random_state : int, optional (default=42)
    If int, random_state is the seed used by the random number generator for reproducibility.
    
    validation_data : bool optional (default=False)
    If bool, validation_data is used to generate validation data set.
    
    Returns
    -------
    X :  array-like of shape (n_rows,) or (n_rows, n_features)
    Feature set.
    
    y :  array-like of shape (n_rows,) or (n_rows, n_features)
    Target variable(s).
    
    X_train :  array-like of shape (n_rows,) or (n_rows, n_features)
    Training data.
    
    X_test : array-like of shape (n_rows,) or (n_rows, n_features)
    Testing data.  
    
    X_val : array-like of shape (n_rows,) or (n_rows, n_features)
    Validation data.  
    
    y_train : array-like of shape (n_rows,) or (n_rows, n_features)
    Training data label. 
    
    y_test : array-like of shape (n_rows,) or (n_rows, n_features)
    Testing data label.
    
    y_val : array-like of shape (n_rows,) or (n_rows, n_features)
    Validation data label.
    
    Examples
    --------
    >>> X, y, X_train, X_test, y_train, y_test = split_data(df, target_variable)
    >>> X, y, X_train, X_test, y_train, y_test = split_data(df, target_variable, test_size=0.25, random_state=142)
    >>> X, y, X_train, X_test, X_val, y_train, y_test, y_val = split_data(df, target_variable, validation_data=True)
    '''
        
    X = df.drop([target_variable], axis=1)
    y = df[target_variable]

    if validation_data == False:
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        print('Data is splitted into X, y, X_train, X_test, y_train, y_test.')
        
        print('\nShape Info of Features Training Set:')
        print(f'Number of datapoints (rows): {X_train.shape[0]}')
        print(f'Number of features (columns): {X_train.shape[1]}')

        print(f'\nShape Info of Features Test Set:')
        print(f'Number of datapoints (rows): {X_test.shape[0]}')
        print(f'Number of features (columns): {X_test.shape[1]}')

        return X, y, X_train, X_test, y_train, y_test
        
    else:
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)

        print('Data is splitted into X, y, X_train, X_test, X_val, y_train, y_test, y_val.')

        print('\nShape Info of Features Training Set:')
        print(f'Number of datapoints (rows): {X_train.shape[0]}')
        print(f'Number of features (columns): {X_train.shape[1]}')

        print(f'\nShape Info of Features Test Set:')
        print(f'Number of datapoints (rows): {X_test.shape[0]}')
        print(f'Number of features (columns): {X_test.shape[1]}')

        print(f'\nShape Info of Features Validation Set:')
        print(f'Number of datapoints (rows): {X_val.shape[0]}')
        print(f'Number of features (columns): {X_val.shape[1]}')

        return X, y, X_train, X_test, X_val, y_train, y_test, y_val
    

    
def evaluate_regressor(
    X_test, 
    y_test, 
    model_name, 
    model, 
    path=None, 
    save=False, 
    title_fontsize=25, 
    xlabel_fontsize=20, 
    ylabel_fontsize=20, 
    xticks_fontsize=18, 
    yticks_fontsize=18, 
    transparent=False, 
    dpi=300, 
    figsize=(20, 10), 
    facecolor='navy', 
    edgecolor='orange',
):
    '''
    Function to evaluate regression models
    
    Parameters
    ----------
    X_test : [array-like, sparse matrix] of shape (n_rows, n_features)
    Testing data.
    
    y_test : array-like of shape (n_rows,) or (n_rows, n_features)
    Testing data.
    
    model_name : str, name of the estimator object being implemented.
    
    model : estimator object implementing 'predict' and 'score'
    The object use to 'fit' the data and 'score' on data
    
    path : str, 

    save : bool,
    
    title_fontsize : int, optional (default=25)
    
    xlabel_fontsize : int, optional (default=20)
    
    ylabel_fontsize : int, optional (default=20)
    
    xticks_fontsize : int, optional (default=18)
    
    yticks_fontsize : int, optional (default=18)
    
    transparent : bool, optional (default=False)
    
    dpi : int, optional (default=300)
    
    figsize : tuple, optional (default=(20, 10))
    
    facecolor : str, optional (default='navy')
    
    edgecolor : str, optional (default='orange')
    
    Returns
    -------
    None
        This method prints the values of regression evaluation metrics with a scatter and returns None.
    
    Examples
    --------
    >>> evaluate(X_test, y_test, 'Random Forest Regressor', RandomForestRegressor())
    
    
    >>> models = [
    ('Linear Regression', LinearRegression()),
    ('Random Forest', RandomForestRegressor())
    ]
    
    >>> for model_name, model in models:
            evaluate(X_test, y_test, model_name, model)
    '''
    regressor_model = model
    y_pred = regressor_model.predict(X_test)
    
    test_accuracy = regressor_model.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    explained_var_score = explained_variance_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    median_ae = median_absolute_error(y_test, y_pred)
    msle = mean_squared_log_error(y_test, y_pred) 
    
    evaluation_summary = pd.DataFrame(
        {
            'mean_squared_error':mse, 
            'root_mean_sqaured_error':rmse, 
            'r2_score':r2, 
            'explained_variance_score':explained_var_score, 
            'mean_absolute_error':mae, 
            'median_absolute_error':median_ae, 
            'mean_sqaured_log_error':msle,
        }, index=[0]
    )
    
    display(evaluation_summary)
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.scatter(
        y_test, 
        y_pred, 
        facecolor=facecolor, 
        edgecolors=edgecolor, 
        label=f'{model_name} Accuracy {round(test_accuracy, 3)}', 
        linewidth=1,
    )
    
    plt.title("Actual vs Predicted", fontsize=title_fontsize)
    plt.xlabel("Actual Target Variable: $Y_i$", fontsize=xlabel_fontsize)
    plt.ylabel("Predicted Target Variable: $\hat{Y}_i$", fontsize=ylabel_fontsize)
    plt.xticks(fontsize=xticks_fontsize)
    plt.yticks(fontsize=yticks_fontsize)
    
    plt.grid(b=True, axis='both', color='grey', linewidth=0.8)
    plt.legend(loc='best', fontsize='xx-large', numpoints=1, frameon=True, shadow=True, fancybox=True)
    
    fig.tight_layout()
    
    if save == False:
        plt.show()
    else:
        plt.savefig(path+'/'+model_name, dpi=dpi, transparent=transparent)
        # plt.show()
        
        
    return None



def baseline_performance(
    models, 
    X_train, 
    y_train, 
    X_test, 
    y_test, 
    column_names=None, 
    csv_path=None, 
    save_model_summary=True, 
    save_feature_importance=True, 
    save_feature_imp_of_each_model=False,
):
    '''
    Function to get baseline performance of chosen estimators.
    
    Parameters
    ----------
    models : estimator(s), 
    
    X_train : [array-like, sparse matrix] of shape (n_rows, n_features),
    Training data.
    
    y_train : array-like of shape (n_rows,) or (n_rows, n_features),
    Training data. 
    
    X_test : [array-like, sparse matrix] of shape (n_rows, n_features)
    Testing data., 
    
    y_test : array-like of shape (n_rows,) or (n_rows, n_features)
    Testing data., 
    
    column_names : list,  
    
    csv_path : str, 
    
    save_model_summary : bool, 
    
    save_feature_importance : bool, 
    
    save_feature_imp_of_each_model : bool, 
    
    
    Returns
    -------
    
    
    Examples
    --------
    
    
    '''
    
    # Instantiating model_names as an empty list to keep the names of the models
    model_names = []

    # Instantiating train_accuracies as an empty list to keep the train accuracy scores of the models
    train_accuracies = []

    # Instantiating test_accuracies as an empty list to keep the test accuracy scores of the models
    test_accuracies = []
    
    # Instantiating models_selected_features as an empty dictionary to keep model names and their selected feature.
    models_selected_features = {}
    
    # Instantiating model_features_with_importance as an empty dictionary to keep features, model names and importance.
    model_features_with_importance = {'Features' : column_names}
    
    # Fitting models to Training Dataset and Scoring them on Test set
    for model_name, model in models:
        model.fit(X_train, y_train)
        
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        model_names.append(model_name)

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        
        
        # Feature Importance
        try:
            if column_names == None:
                return None
            else:
                features = column_names
                importances = model.feature_importances_
                
                feature_importance = pd.DataFrame(
                    {
                        'Feature': features, 
                        'Importance': importances,
                    }
                )
                
                selected_importance = []
                for importance in feature_importance['Importance']:
                    selected_importance.append(importance)
                
                model_features_with_importance.update({model_name : selected_importance})
                
                    
                    
                # Saving Feature Importance of Each Model
                if save_feature_imp_of_each_model == True:
                    feature_importance.to_csv(csv_path+'/feature_importance_'+model_name+'.csv', index=False)
                    
                    if csv_path == None:
                        print("###########################################################################################")
                        print(f'Feature Importance csv of {model_name} Successsfully saved at current working directory.')
                        print("###########################################################################################")
                        print('\n')
                    else:
                        print("###########################################################################################")
                        print(f'Feature Importance csv of {model_name} Successsfully saved to path `{csv_path}`.')
                        print("###########################################################################################")
                        print('\n')
                else:
                    return None
                
                
                # Sorting feature importance in descending order (highest to lowest) inplace
                feature_importance.sort_values("Importance", inplace=True, ascending=False)
                
                # Selecting model's Features from it's feature importance
                important_features = feature_importance['Feature']
                
                selected_features = []
                
                # appending each selected feature by the model to selected_features
                for feature in important_features:
                        selected_features.append(feature)
                
                # Updating models_selected_features dictionary with each model and their corresponding selected features
                models_selected_features.update({model_name: selected_features})
                
        except:
            return f'Error Encounted while trying to accquire Feature Importance!'
    
    
    # Creating a pandas DataFrame for models and their selected features
    feature_importance_of_models = pd.DataFrame(models_selected_features)
    
    # Saving a summary of feature importance of the selected predictors to csv without their feature importance values
    if save_feature_importance == True:
        
        feature_importance_of_models.to_csv(csv_path+"/Summary_Feature_Importance_of_Selected_Predictors.csv", index=False)
        
        if csv_path == None:
            print("###############################################################################")
            print('Feature Importance csv Successsfully saved at current working directory.')
            print("###############################################################################")
            print('\n')
        else:
            print("###################################################################")
            print(f'Feature Importance csv Successsfully saved to path `{csv_path}`.')
            print("###################################################################")
            print('\n')
    
    else:
        return None
    
    
    
    # Creating a pandas DataFrame of models and their selected importance
    df_model_features_with_importance = pd.DataFrame(model_features_with_importance)
    
    # Saving a summary of feature importance of the selected predictors to csv without their feature importance values
    if save_feature_importance == True:
        
        df_model_features_with_importance.to_csv(csv_path+"/Summary_Feature_Importance_of_Selected_Predictor(s).csv", index=False)
        
        if csv_path == None:
            print("#####################################################################################")
            print('Summary of Feature Importance csv Successsfully saved at current working directory.')
            print("#####################################################################################")
            print('\n')
        else:
            print("###############################################################################")
            print(f'Summary of Feature Importance csv Successsfully saved to path `{csv_path}`.')
            print("###############################################################################")
            print('\n')
    
    else:
        return None
    
    
    
    # Model Summary
    model_summary = pd.DataFrame(
        {
            'Model Name' : model_names, 
            'Train Accuracy' : train_accuracies, 
            'Test Accuracy' : test_accuracies,
        }
    )
    
    # Saving Model Summary to csv
    if save_model_summary == True:
        
        model_summary.to_csv(csv_path+"/model(s)_summary.csv", index=True)
        
        if csv_path == None:
            print('Model(s) Summary csv Successsfully saved at current working directory.')
            print("###################################################################")
            print(model_summary)
            print("###################################################################")
            print('\n')
        else:
            print(f'Model(s) Summary csv Successsfully saved to path `{csv_path}`.')
            print("###################################################################")
            print(model_summary)
            print("###################################################################")
            print('\n')
    
    else:
        print("###################################################################")
        print(model_summary)
        print("###################################################################")
        print('\n')
    
    
    return feature_importance_of_models, df_model_features_with_importance, model_summary




def plot_model_summary(
    model_summary, 
    figsize=(20, 14), 
    dpi=300,
    transparent=True, 
    save_visualization=True, 
    figure_name=None, 
    figure_path=None,
):
    '''
    Function to plot model summary
    
    Parameters
    ----------
    model_summary : DataFrame, 
    
    figsize : tuple, 
    
    dpi : int,
    
    transparent : bool, 
    
    save_visualization : bool, 
    
    figure_name : str, 
    
    figure_path : str,
    
    Returns
    -------
    
    
    Examples
    --------
    
    
    '''
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    width = 0.4

    x_ticks_location = np.arange(len(model_summary['Model Name']))


    test_plot = ax.bar(
        x_ticks_location, 
        model_summary['Test Accuracy'], 
        width, 
        color="deepskyblue", 
        edgecolor="orangered", 
        align='center', 
        linewidth=3, 
        label='Test Accuracy', 
        hatch=("/"),
    )


    train_plot = ax.bar(
        x_ticks_location + width, 
        model_summary['Train Accuracy'], 
        width, color="darkorange", 
        edgecolor="royalblue", 
        align='center', 
        linewidth=3, 
        label='Train Accuracy',
    )


    ax.set_title("Models Accuracy", fontsize=28, pad=25)

    ax.set_xticks(x_ticks_location + width / 2)
    ax.set_xticklabels(model_summary['Model Name'], fontsize=25)

    ax.set_xlabel('Model', fontsize=28, labelpad=20)

    plt.yticks(fontsize=25)
    ax.set_ylabel('Accuracy', fontsize=28, labelpad=20)

    ax.grid(b=True, which="both", axis="both", color="black", linewidth=0.8)

    ax.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize='xx-large', frameon=True, shadow=True, fancybox=True)

    fig.tight_layout()
    
    
    def label(plot):
        for patch in plot.patches:
            width = patch.get_width()
            height = patch.get_height()
            x, y = patch.get_xy()
            plt.text(
                x=x + width / 2, 
                y=y + height * 1.001,
                s=round(height, 3),
                ha='center',
                va="bottom",
                fontsize='xx-large',
            )

    label(test_plot)
    label(train_plot)

    if save_visualization == True and figure_name != None:
        # Saving the plot as a png file
        plt.savefig(figure_path+'/'+figure_name, dpi=dpi, transparent=transparent)

        # Showing the plot
        # plt.show()

        if figure_path == None:
            print("###############################################################################")
            print('Figure Successsfully Saved at current working directory.')
            print("###############################################################################")
            print('\n')
        else:
            print("###################################################################")
            print(f'Figure Successsfully Saved to path `{figure_path}`.')
            print("###################################################################")
            print('\n')
    elif save_visualization == True and figure_name == None:
        plt.savefig(figure_path+'/Bar_Charts_of_Models_and_their_Accuracy.png', dpi=dpi, transparent=True)

        # Showing the plot
        # plt.show()

        if figure_path == None:
            print("###############################################################################")
            print('Figure Successsfully Saved at current working directory.')
            print("###############################################################################")
            print('\n')
        else:
            print("###################################################################")
            print(f'Figure Successsfully Saved to path `{figure_path}`.')
            print("###################################################################")
            print('\n')
    else:
        # Showing the plot
        plt.show()

        return None
                     
                
                    
def plot_feature_importance(
    feature_importance, 
    figsize=(15, 10), 
    dpi=300, 
    transparent=True, 
    annotate_fontsize='large',
    save_plot=False,
    path=None,
):
    '''
    df_model_features_with_importance, 
    figsize, 
    dpi, 
    transparent, 
    annotate_fontsize,
    save_plot,
    path,
    '''
    
    column_names = list(feature_importance.iloc[:, 1:].columns.values)
    
    for column_name in column_names:
        
        feature_importance_subset = feature_importance[['Features', column_name]].sort_values(by=column_name, ascending=False, inplace=False)
        
        fig = plt.figure(figsize=figsize, dpi=dpi)
        
        ax = feature_importance_subset[column_name].plot(
            kind='bar', 
            rot=90, 
            color="royalblue", 
            edgecolor="darkorange", 
            linewidth=4,
        )
        
        ax.set_title(f'{column_name} Feature Importance', fontsize=28, pad=25)
        
        for patch in ax.patches:
            width = patch.get_width()
            height = patch.get_height()
            x, y = patch.get_xy()
            ax.annotate(
                f'{round(height, 3)}', 
                (x + width/2, y + height*1.001), 
                ha='center', 
                fontsize=annotate_fontsize
            )
        
        ax.set_xticklabels(feature_importance_subset['Features'], fontsize=25)
        ax.set_xlabel('Features', fontsize=25, labelpad=20)
        
        plt.yticks(fontsize=25)
        ax.set_ylabel('Importance', fontsize=28, labelpad=20)
        
        ax.grid(b=True, which="both", axis="both", color="black", alpha=0.1, linewidth=0.8)
        
        fig.tight_layout()
        
        if save_plot == True:
                    
            # Saving the plot as a png file
            plt.savefig(path+'/'+column_name, dpi=dpi, transparent=transparent)
                
            # Showing the plot
            # plt.show()
            
            if path == None:
                print("###############################################################################")
                print('Figure Successsfully Saved at current working directory.')
                print("###############################################################################")
                print('\n')
            else:
                print("###################################################################")
                print(f'Figure Successsfully Saved to path `{path}`.')
                print("###################################################################")
                print('\n')
        elif save_plot == False:
            
            # Show the plot
            plt.show()
                    
        else:
            print("The argument `save_plot` only takes bool (True or False) values!")
            
            
            
def evaluate_classifier(
    estimator, 
    X_test, 
    y_test,
    save_figure=False,
    figure_path="",
    transparent=False, 
    dpi=300,
    cmap="Purples",
):
    
    y_pred = estimator.predict(X_test)
    y_proba = estimator.predict_proba(X_test)[:, 1]
    
    report = metrics.classification_report(
        y_test, 
        y_pred, 
        digits=3, 
        output_dict=True
    )
    
    report_df = pd.DataFrame(report)
    
    test_accuracy_score = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred)
    
    ######################################
    #### Write scores to a text file #####
    ######################################
    
    with open("metrics.txt", 'w') as output_text_file:
        
        output_text_file.write(f"Test Accuracy variance explained: {round(test_accuracy_score, 4)}\n")
        
        output_text_file.write(f"Precision Score: {round(precision, 4)}\n")
        
        output_text_file.write(f"Recall Score: {round(recall, 4)}\n")
        
        output_text_file.write(f"F1 Score: {round(f1_score, 4)}\n")

    
    print("#############################################################")
    print(f"Classifier's Accuracy : {round(test_accuracy_score, 4)}")
    print(f"\nClassifier's Precision : {round(precision, 4)}")
    print(f"\nClassifier's Recall : {round(recall, 4)}")
    print(f"\nClassifier's F1-score : {round(f1_score, 4)}")
    print("#############################################################")
    print('\n')
    
    
    ######################################
    ##### Plotting Confusion Matrix ###### 
    ######################################
    
    cm_output = metrics.confusion_matrix(y_test, y_pred)
    
    # Put it into a dataframe for seaborn plot function
    cm_df = pd.DataFrame(cm_output)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=dpi)

    sns.heatmap(
        cm_df, 
        annot=True, 
        square=True,
        annot_kws={"size": 20}, 
        cmap=cmap, 
        fmt='d', 
        linewidths=2, 
        linecolor="darkorange", 
        cbar=False, 
        xticklabels=[0, 1], 
        yticklabels=[0, 1]
    )

    plt.title("Confusion Matrix", fontsize=25, pad=20)

    plt.xlabel("Predicted Label", fontsize=25, labelpad=5)
    plt.xticks(fontsize=20)

    plt.ylabel("Actual Label", fontsize=25, labelpad=5)
    plt.yticks(fontsize=20)
    
        
    ax.text(2.25, -0.10,'Test Accuracy: '+ str(round(test_accuracy_score, 4)), fontsize=14)
    
    ax.text(2.25, 0.0,'Precision: '+ str(round(precision, 4)), fontsize=14)
    
    ax.text(2.25, 0.1,'Recall: '+ str(round(recall, 4)), fontsize=14)
    
    ax.text(2.25, 0.2,'F1 Score: '+ str(round(f1_score, 4)), fontsize=14)
    
    
    fig.tight_layout()
    
    if save_figure == False:
        plt.show()
    
    elif save_figure == True:
        plt.savefig(figure_path+'/confustion_matrix_plot.png', dpi=dpi, transparent=transparent)
        # plt.show()
    
    else:
        print("The `save_figure` argument accepts bool (True or False) values only!")
        # plt.show()
    
    print("\n")
    print("\n")
    
    
    ######################################
    ####### Plotting ROC Curve ########### 
    ######################################
    
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_proba)
    roc_score = metrics.roc_auc_score(y_test, y_proba)
    
    fig = plt.figure(figsize=(12, 10), dpi=dpi)
    
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

    if save_figure == False:
        plt.show()

    elif save_figure == True:
        plt.savefig(figure_path+'/ROC_Curve.png', dpi=dpi, transparent=transparent)
        # plt.show()

    else:
        print("The `save_figure` argument accepts bool (True or False) values only!")
        # plt.show()
    
    
    ######################################
    ####### Plotting Residuals ########### 
    ######################################
    
    fig = plt.figure(figsize=(7, 7), dpi=dpi)
    
    y_pred_ = estimator.predict(X_test) + np.random.normal(0, 0.25, len(y_test))
    
    y_jitter = y_test + np.random.normal(0, 0.25, len(y_test))
    
    residual_df = pd.DataFrame(list(zip(y_jitter, y_pred_)), columns=["Actual Label", "Predicted Label"])
    
    ax = sns.scatterplot(
        x="Actual Label", 
        y="Predicted Label",
        data=residual_df,
        facecolor='dodgerblue',
        linewidth=1.5,
    )
    
    ax.set_xlabel('Actual Label', fontsize=14) 
    
    ax.set_ylabel('Predicted Label', fontsize=14)#ylabel
    
    ax.set_title('Residuals', fontsize=20)
    
    min = residual_df['Predicted Label'].min()
    
    max = residual_df['Predicted Label'].max()
    
    ax.plot([min, max], [min, max], color='black', linewidth=1)
    
    plt.tight_layout()
    
    if save_figure == False:
        plt.show()

    elif save_figure == True:
        plt.savefig(figure_path+'/Residuals_plot.png', dpi=dpi, transparent=transparent)
        # plt.show()
    
    else:
        print("The `save_figure` argument accepts bool (True or False) values only!")
        # plt.show()
    
    display(report_df)

    plt.close('all')