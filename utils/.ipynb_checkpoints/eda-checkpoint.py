import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt, abs, round
import pandas as pd
from pandas.api.types import is_numeric_dtype
import seaborn as sns
from scipy.stats import chi2_contingency, norm, t as t_dist

def get_basic_dataset_summary(X_train: pd.DataFrame,
                              y_train: list) -> None:
    
    print('X_train has {} rows and {} columns'
          .format(X_train.shape[0], X_train.shape[1]))
    print('y_train has {} rows'
          .format(len(y_train)))
    print('\nColumns in train dataset with any null values: {}'
          .format(X_train.columns[X_train.isna().any()].tolist()))
    
def plot_target(y: list) -> None:
    dist = dict((x, y.count(x)) for x in set(y))
    
    labels = list(dist.keys())
    values = list(dist.values())
    
    fig, ax = plt.subplots()
    
    ax.bar(range(len(dist)), values, tick_label=labels)
    ax.set_ylabel('quantity')
    ax.set_title('Distribution of target')
    plt.figure(figsize=(32, 24))
    plt.show()

def print_categorical_values(df: pd.DataFrame) -> None:
    unhashable = []
    for col in df.columns:
        if not is_numeric_dtype(df[col]):
            try:
                values = df[col].unique()
                if len(values) <= 20:
                    print('Column: {}\nPossible values: {}\n'.format(col, values))
                else:
                    print('Column: {}\nFeature with high cardinality (more than 20 categories)\n'
                          .format(col, values))
            except TypeError:
                unhashable.append(col)
    print('Columns with unhashable data type: {}'.format(unhashable))

def plot_nas(df: pd.DataFrame,
             title='Missing ratio of dataframe columns') -> None:
    if df.isnull().sum().sum() != 0:
        na_df = (df.isnull().sum() / len(df)) * 100      
        na_df = na_df.drop(na_df[na_df == 0].index).sort_values(ascending=False)
        missing_data = pd.DataFrame({'Missing Ratio %' :na_df})
        missing_data.plot(kind = 'barh', title=title)
        plt.show()
    else:
        print('No NAs found')
        
def one_hot_encode_list(df: pd.DataFrame,
                        col_name: str) -> pd.DataFrame:
    
    v = df[col_name].values
    l = [len(x) for x in v.tolist()]
    f, u = pd.factorize(np.concatenate(v))
    n, m = len(v), u.size
    i = np.arange(n).repeat(l)

    dummies = pd.DataFrame(
        np.bincount(i * m + f, minlength=n * m).reshape(n, m),
        df.index, u
    )

    return df.drop(col_name, axis=1).join(dummies)

def plot_over_time(df: pd.DataFrame,
                   time_key: str,
                   freq: str,
                   title: str) -> None:
    
    df.groupby(
        pd.Grouper(
            key='date_created', freq='10D')
    )['price'].count().plot(title='Data over date_created')

def plot_time_stability(df: pd.DataFrame,
                        time_col: str,
                        cat_col: str,
                        aux_col: str,
                        ax,
                        freq='10D') -> None:
    gp = (
        df.groupby(
            [pd.Grouper(key=time_col, freq=freq),
             pd.Grouper(cat_col)])[aux_col]
        .count()
        .reset_index()
        .rename(columns={aux_col: 'count'}))
        
    for key, data in gp.groupby(cat_col):
        data.plot(x=time_col, y='count', ax=ax, label=key)
    
    ax.set_title(cat_col)

def bivariate_analysis_categorical(df: pd.DataFrame,
                                   target: str,
                                   category: str) -> None:
    df = df[[category,target]][:]
    
    table = pd.crosstab(df[target], df[category],)
    f_obs = np.array([table.iloc[0][:].values,
                    table.iloc[1][:].values])
    
    chi, p, dof, expected = chi2_contingency(f_obs)
    
    if p < 0.05:
        sig = True
    else:
        sig = False

    ax = df.groupby(category)[target].value_counts(normalize=True).unstack()
    ax.plot(kind='bar', stacked='True',title=str(ax))
    int_level = df[category].value_counts()

def correlation_heatmap(df: pd.DataFrame,
                       corr_types=['pearson']) -> None:
    
    plt.figure(figsize=(36,16), dpi=140)
    for j,i in enumerate(corr_types):
        plt.subplot()
        correlation = df.dropna().corr(method=i)
        sns.heatmap(correlation, linewidth = 2)
        plt.title(i, fontsize=18)

def bivariate_numerical_categorical(df: pd.DataFrame,
                                    num_col: str,
                                    cat_col: str,
                                    pos_category: str,
                                    neg_category: str) -> None:
    
    x1 = df[num_col][df[cat_col]==pos_category][:]
    x2 = df[num_col][~(df[cat_col]==pos_category)][:]
    
    m1, m2 = x1.mean(), x2.mean()
    
    plt.figure(figsize = (20,4), dpi=140)
    
    plt.subplot(1,3,1)
    data = pd.DataFrame({'mean': [m1, m2], cat_col: [pos_category, neg_category]})
    sns.barplot(data=data, x=data[cat_col], y=data['mean'])
    plt.ylabel('mean {}'.format(num_col))
    plt.xlabel(cat_col)
    plt.title('Mean values for {}'.format(num_col))
    
    plt.subplot(1,3,2)
    sns.kdeplot(x1, fill= True, color='blue', label = pos_category, warn_singular=False)
    sns.kdeplot(x2, fill= False, color='green', label = neg_category,
                linewidth = 1, warn_singular=False)
    plt.title('{} over {}'.format(num_col, cat_col))
    
    plt.subplot(1,3,3)
    sns.boxplot(x=cat_col, y=num_col, data=df)
    plt.title('Boxplot per value of {}'.format(cat_col))
    plt.show()