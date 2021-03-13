
#######################################
#######################################
# Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing seaborn for style and beauty
import seaborn as sns

plt.style.use('fivethirtyeight')

# custom
from custom import helper
#######################################
#######################################



#######################
### Loading Dataset ###
#######################

raw_data = pd.read_csv("data/winequality-white.csv", sep=';', index_col=None, encoding='ISO-8859-1', engine='python')
# raw_data.head()

# Creating a Copy of the Dataset
df = raw_data.copy()
# df.head()


###################################################
### Creating Folders to keep figures and tables ###
###################################################

helper.create_folder('./csv_tables/')
helper.create_folder('./figures/')

##################################################
### Getting Statistical Summary of the Dataset ###
##################################################

data_description = df.describe(include='all', datetime_is_numeric=True)
data_description.to_csv("csv_tables/data_description.csv", index=False)
# data_description


#################################################
### Getting Column Names and their Data Types ###
#################################################

dataset_columns = pd.DataFrame({'column_names':list(df.columns)})

data_types = []
for column in df.columns:
    dtype = str(df[column].dtype)
    data_types.append(dtype)

dataset_columns['data_type'] = data_types
dataset_columns.to_csv("csv_tables/column_heads_of_dataset.csv", index=True)
# dataset_columns


###############################
### Checking Missing Values ###
###############################

missing_values = helper.missing_data(df)
missing_values.to_csv("csv_tables/missing_values.csv", index=True)
# missing_values


##################################
### Removing Duplicated Values ###
#################################

df.drop_duplicates(keep='first', inplace=True)


#############################
### Checking Outlier Info ###
#############################

outliers = helper.outlier_info(df)
outliers.to_csv("csv_tables/outlier_info.csv", index=True)
# outliers


##########################
### Detecting Outliers ###
##########################

helper.detect_outliers(
    df.select_dtypes(np.number),
    image_name='Outlier',
    path='figures',
    plot_size=(25, 10), 
    xticklabels_fontsize=15, 
    yticklabels_fontsize=15, 
    title_fontsize=22, 
    boxplot_xlabel_fontsize=18, 
    save=True, 
    dpi=600, 
    transparent=True
)


#######################################
### Correlation of Dataset Features ###
#######################################

helper.correlation_viz(
    df.select_dtypes(np.number),
    image_name='Data_Features_Correlation',
    path='figures',
    plot_size=(25, 40),
    cor_bar_orient='vertical',
    colormap="Blues",
    save=True,
    dpi=600,
    transparent=True,
)


############################################
### Distribution Plot of Target Variable ###
############################################

plt.figure(figsize=(15, 10), dpi=300)

sns.distplot(df['quality'], color='darkorange', bins=30)
plt.title('Distribution of quality', fontsize=22)

plt.grid(color='grey')

plt.savefig('Distribution_Plot_of_quality.png', dpi=600, transparent=True)
# plt.show()


#####################################
### Count Plot of Target Variable ###
#####################################

def count_plot(column_name):

    fig = plt.figure(figsize=(10, 8), dpi=300)

    ax = df[column_name].value_counts().plot(
        kind='bar', 
        rot=0, 
        color=['deepskyblue', 'dodgerblue', 'royalblue'], 
        edgecolor='darkorange', 
        linewidth=3,
    )
    
    plt.title(column_name, fontsize=20, pad=16.5)
    
    for patch in ax.patches:
        width = patch.get_width()
        height = patch.get_height()
        x, y = patch.get_xy()
        ax.annotate(f'{height}', (x + width/2, y + height*1.02), ha='center')
    
    plt.xticks(fontsize=20)
    
    plt.grid(alpha=0.3, color='grey')
    
    
    plt.savefig(f'figures/{column_name}_count_plot', dpi=600, transparent=True)
#     plt.show()

count_plot('quality')


#############################################
### Creating Target Label for the Dataset ###
#############################################

mean_value_of_quality = df['quality'].mean()
targets = np.where(df['quality'] > mean_value_of_quality, 1, 0)
df['quality_rate'] = targets
# df.head()


#################################################
### Count Plot of the Created Target Variable ###
#################################################

count_plot('quality_rate')


##############################################
### Feature Statistics per target variable ###
##############################################

features_mean_stats = df.groupby('quality_rate').mean()
features_mean_stats.to_csv("features_mean_stats_per_target_variable.csv", index=True)
# features_mean_stats

for col in list(features_mean_stats.columns.values):
    subset = features_mean_stats[col]
    
    fig = plt.figure(figsize=(12, 8), dpi=300)
    
    ax = subset.plot(
        kind='bar',
        color=['#EF5350', '#FFE0B2'], 
        edgecolor='#F5F5F5', 
        linewidth=7,
        rot=0,
    )
    
    plt.title(f'Mean Value of {col.title()} per Target Label', fontsize=25, pad=20.5)
    
    plt.xticks(ticks=np.arange(len(subset)), labels=['Low Quality', 'High Quality'], fontsize=20)
    
    plt.grid(alpha=0.2, color='grey')
    
    for patch in ax.patches:
        width = patch.get_width()
        height = patch.get_height()
        x, y = patch.get_xy()
        ax.annotate(f'{round(height, 3)}', (x + width/2, y + height*1.02), ha='center')
    
    plt.savefig(f'figures/Mean Value of {col} per Target Label.png', dpi=600, transparent=True)
#     plt.show()

features_std_stats = df.groupby('quality_rate').std()

features_std_stats.to_csv("features_std_stats_per_target_variable.csv", index=True)
features_std_stats


######################################################
### Plot of Feature Statistics per Target Variable ###
######################################################

column_names = list(df.iloc[:, :11].columns.values)

labels = ['high_rate', 'low_rate']

colors = ['orange', 'cyan']

for column_name in column_names:
    
    fig = plt.figure(figsize=(10, 6))


    ax = sns.kdeplot(
        df.loc[(df['quality_rate'] == 1), column_name], 
        color=colors[0], 
        shade=True, 
        label=labels[0],
    )

    ax = sns.kdeplot(
        df.loc[(df['quality_rate'] == 0), column_name], 
        color=colors[1], 
        shade=True, 
        label=labels[1],
    )

    plt.title(f'{column_name} Distribution - high_rate vs. low_rate', fontsize=22)

    plt.tick_params(top=False, bottom=True, left=True, right=False)

    plt.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize='x-small', frameon=True, shadow=True, fancybox=True)

    plt.tight_layout()
    
    plt.savefig('figures/'+column_name+'Distribution of high_rate_low_rate_quality.png', dpi=600, transparent=True)
#     plt.show()


###############################################
### Creating a Copy of the modified Dataset ###
###############################################

df_modified = df.copy()
# df_modified.head(15)


##########################################
### Saving the Modified Dataset as csv ###
##########################################

df_modified.to_csv("preprocessed_data.csv", index=False)
