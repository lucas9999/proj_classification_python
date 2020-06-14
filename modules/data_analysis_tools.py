
  
  ########################### ML tools
  
  
  #########
# NOTES #
#########

# In the beginning of the script most (but not all) necessary packages are provided. Packages dedicated for very particular use are sometimes placed before function definition. 
# If some part of code is no finished jet, a proper notice is added before.
# 
# Functions are divided into few subgroups:
# 
# 1. Basic statistics
# 2. Correlation
# 3. Basic plots
# 4. Variables transformation and encoding
# 5. Features rankings
# 6. Missing values imputation
# 7. IML
# 8. 
# Now most functions are very basic and their role is just to give support in early stages of model development.




# Required packages

import pandas      as pd
import numpy       as np
import sklearn     as skl
import statsmodels as stm
# import keras       as ker
import plotnine
import seaborn     as sb
import matplotlib.pyplot as plt
import seaborn as sns

import os
import itertools   as it
import sklearn.metrics as skm
import datetime
import math
import copy # deepcopies
import scipy
# import imblearn   # problem of imbalanced samples
from scipy            import stats
from scipy.special    import boxcox, inv_boxcox
from rfpimp           import permutation_importances
from sklearn.metrics  import r2_score # scores for assessment model performance
from sklearn.ensemble import IsolationForest # for outliered detecting

# from imblearn.over_sampling    import RandomOverSampler
# from imblearn.over_sampling    import SMOTE
# from imblearn.under_sampling   import RandomUnderSampler
from plotnine.themes.themeable import axis_line
from sklearn.model_selection   import train_test_split
from sklearn.ensemble          import AdaBoostClassifier
from sklearn.ensemble          import RandomForestClassifier
from sklearn.ensemble          import GradientBoostingClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.naive_bayes       import MultinomialNB
from sklearn.svm               import SVC
from sklearn.preprocessing     import OneHotEncoder
from sklearn.calibration       import CalibratedClassifierCV
# from sklearn.preprocessing     import Imputer
from sklearn.impute            import SimpleImputer as Imputer
import catboost
from catboost                  import *
from plotnine                  import * # ggplot for python
import shap
# https://www.kaggle.com/discdiver/category-encoders-examples
import category_encoders as ce

# pd.set_option('precision', 2)
# pd.options.display.float_format = '{:,}'.format
# pd.options.display.max_columns=100
# pd.options.display.max_rows=500
# pd.options.display.max_colwidth=30


####################
# Basic statistics #
####################



# >>>  glimpse of data  <<<

def glimpse_data(data, var_cat = []):
  """ 
  Purpose: 
    Fast glimpse at stucture of Data.Frame.
  
  Arguments:
    data: DataFrame
    var_cat: list of names for categorical data. If empty list, then all non-numerical variables are regarded as categorical. 
  
  Output:
    Data.Frame
    
  Example of use:
    glimpse(data = data_, var_cat = ['czy_wynagrodzenie','max_dpd_60'])
  """
  
  # deviding variables into two categories : numerical and categorical
  cat = list(data.select_dtypes(['object', pd.Categorical]).columns) + var_cat
  num = set(data.columns).difference(set(cat))
  cat = cat
  num = num
  
  # calculations for numerical variables
  df1 = data.loc[:,num].apply(lambda x : [sum(pd.isnull(x)), sum(pd.isnull(x))/len(x) , min(x), np.quantile(x, q=0.25), np.quantile(x, q=0.5), np.quantile(x, q=0.75),  max(x)], result_type = 'expand')
  df2 = pd.DataFrame()
  
  # calculations for categorical variables
  for i in var_cat:
    df2[[i]]=data.loc[:,[i]].apply(lambda x : [sum(pd.isnull(x))/len(x)]  + [sum(pd.isnull(x))] + list(map(lambda x, y : str(x)+' ('+str(y)+')', list(pd.value_counts(x).iloc[0:5].index), [str(x) for x in list(pd.value_counts(x).iloc[0:5].values)])), result_type = 'expand') 
  
  # binding results
  df3 = pd.concat([df1, df2], axis=1 )
  df3.index = ['NA', 'NA_proc', 'min', 'q_25', 'q_50', 'q_75', 'max']
  
  return(df3)



# >>> cross tables <<<<

def cross_tab(data, var_1 = None, var_2 = None, round = 1, normalize = 'index'):
  """
  Purpose:
    Table with count for each categories in one or two categorical variables. Results are also displayed as percentage. 
  
  Arguments:
    data: DataFrame
    var_1: name of first categorical variable
    var_2: name of second categorical variable (optional)
    round: rounding for percentages
    normalize: how to normalize percentage values in case of analyzis of two variables. For details see the documentation for pandas.crosstab.
    
  Output:
    DataFrame
  
  Example of use
    # for 1 variable
    cross_tab(data=data_, var_1='zrodlo_dochodu')
    # for 2 variables
    cross_tab(data=data_, var_1='zrodlo_dochodu', var_2 = 'FLAGA_RESPONSE')
  """
  
  t1 = data[var_1].value_counts()
  t2 = np.round(100*data[var_1].value_counts(normalize=True), round)
  res = pd.concat([t1,t2], axis=1)
  # res = res.reset_index(drop = False)
  res.columns = ['count', 'percent']
  
  if var_2 is None:
    return(res)
  else:
    t3 = pd.crosstab(data[var_1], data[var_2])
    t3.columns = ['c(' + str(x)+')' for x in t3.columns]
    t4 = np.round(100*pd.crosstab(data[var_1], data[var_2], normalize = normalize), round) # columns
    t4.columns = ['p(' + str(x)+')' for x in t4.columns]
    t5 = pd.concat([t3, t4], axis=1)
    res = pd.merge(res, t5, left_index=True, right_index=True)
    return(res)



def variable_diagnostic(data, var, print_res = True, levels_n_limit = 7, levels_n_max = 30):
  """
  Purpse:
    Basic diagnostic for any type of variables. 
    
  Arguments:
    data: DataFrame
    var: name or index of variable to analyse
    print_res: If 'True' then result will beprinted in more readable way. If 'False' function just returns list of results.
    levels_n_limit: Information of how many top count leves of categorical variable should be included in results.
    levels_n_max: The maximum number of levels of variable when it is treated as categorical variable (important for numerical variables which have low number of levels so they should be considered to be treaten as categorical type).  
 
    
  """
  if type(var) == int:
    data = data.iloc[:,var]
  else:
    data = data[var]
  d_type = data.dtype.name
  var_len = len(data)
  proc_na = np.round(100*sum(data.isna())/var_len, 3)
  
  n_levels = len(data.dropna().drop_duplicates())
  if n_levels < levels_n_max:
    cross = np.round(100*data.value_counts()/len(data),1)
    cross = pd.DataFrame({'cat':list(cross.index), 'count':cross.values})
    cross = cross.apply(lambda x: ''.join([str(x[0]), ' (', str(x[1]), ')'],), axis=1)
    cross = cross[0:levels_n_limit]
    lst = []
    for i in range(levels_n_limit):
      lst = lst + [cross.get(i, default=np.nan)]
    cross = lst
    if d_type not in ['category', 'object']:
      if n_levels == 2:
        rec_type = 'bn'
      else:
        rec_type = 'cn'
      stats = pd.DataFrame({ 'min': [np.nanmin(data.values)]
                ,'q_10':[np.nanquantile(data.values, 0.1)]
                ,'q_25':[np.nanquantile(data.values, 0.25)]
                ,'q_50':[np.nanquantile(data.values, 0.50)]
                ,'q_75':[np.nanquantile(data.values, 0.75)]
                ,'q_90':[np.nanquantile(data.values, 0.9)]
                ,'max':[np.nanmax(data.values)]})
    else:
      if n_levels == 2:
        rec_type = 'b'
      else:
        rec_type = 'c'
      stats = pd.DataFrame({ 'min': [np.nan]
                ,'q_10':[np.nan]
                ,'q_25':[np.nan]
                ,'q_50':[np.nan]
                ,'q_75':[np.nan]
                ,'q_90':[np.nan]
                ,'max':[np.nan]})
  elif n_levels > levels_n_max and d_type not in ['category', 'object']:
    rec_type = 'n'
    cross = [np.nan]*levels_n_limit
    stats = pd.DataFrame({ 'min': [np.nanmin(data.values)]
                ,'q_10':[np.nanquantile(data.values, 0.1)]
                ,'q_25':[np.nanquantile(data.values, 0.25)]
                ,'q_50':[np.nanquantile(data.values, 0.50)]
                ,'q_75':[np.nanquantile(data.values, 0.75)]
                ,'q_90':[np.nanquantile(data.values, 0.9)]
                ,'max':[np.nanmax(data.values)]})
      
  else:
    rec_type = 'c'
    cross = [np.nan]*levels_n_limit
    stats = pd.DataFrame({ 'min': [np.nan]
              ,'q_10':[np.nan]
              ,'q_25':[np.nan]
              ,'q_50':[np.nan]
              ,'q_75':[np.nan]
              ,'q_90':[np.nan]
              ,'max':[np.nan]})
  if print_res:
    print(' ', proc_na, '% \n', rec_type, '\n', d_type, '\n', n_levels, '\n', cross, '\n\n', stats)
  else:
    return([proc_na, rec_type, d_type, n_levels, cross, stats])




def glimpse_data_extended(data = None, levels_n_limit = 7, levels_n_max=35, copy_to_clipboard = True):
  
  """
  Purpose:
    Extended version of glimse_data. Results can be automatically copied to clipboard so you can easly transef them to excel for excample. This function is also a wrapper for 'variable_diagnostic' function.
  
  Arguments:
    data: DataFrame
    levels_n_limit: Information of how many top count leves of categorical variable should be included in results.
    levels_n_max: The maximum number of levels of variable when it is treated as categorical variable (important for numerical variables which have low number of levels so they should be considered to be treaten as categorical type).  
    copy_to_clipboard: If copy results to clipboard
    
  Output:
    DataFrame and copy of it to clipboard
    
  """
  
  df = pd.DataFrame()
  for i in range(data.shape[1]):
    c_t = variable_diagnostic(data=data, var = i, print_res = False, levels_n_limit = 7, levels_n_max=30)
    c_t_2 = pd.DataFrame({'nan_proc':[c_t[0]], 'rec_type':[c_t[1]], 'type':[c_t[2]],'n_levels':[c_t[3]]})
    c_t = pd.concat([pd.DataFrame({'var':[data.columns[i]]}), c_t_2, pd.DataFrame({ i : [c_t[4][i]] for i in range(0, len(c_t[4]) ) } ),  c_t[5]], axis = 1)
    df = pd.concat([df, c_t], axis = 0)
  df = df.reset_index(drop=True)
  df = df.reset_index(drop=False)
  df['remove'] = 0
  df['importance_cat'] = 1
  df['importance_rand'] = 0
  df['importatnce_permut'] = 0
  df['pearson'] = 1
  df['v_cramer'] = 1
  df['ratio'] = 1
  df['kendall'] = 1
  if copy_to_clipboard:
    df.to_clipboard(index=False)
  return(df)



# t1 = glimpse_data_extended(data=data)
# from IPython.terminal.debugger import set_trace
def A_F_correlation_selection(data = None, y = None, analytical_table_file = None, sheet = 'Arkusz1', round = 2):
#   import pdb; pdb.set_trace()
  # y = ''
  # sheet = 'Arkusz1'
  # analytical_table_file = '.xlsx'
  if sheet is None:
    analytical_table_full = analytical_table_file
  else:
    analytical_table_full = pd.read_excel(analytical_table_file, sheet_name = sheet)
  
  # remove variables marked to remove
  analytical_table = analytical_table_full.loc[analytical_table_full['remove']==0,]
  analytical_table = analytical_table.reset_index(drop = True)
  # recomended type for target variable
  y_rec_type = analytical_table.loc[analytical_table['var'] == y, 'rec_type'].values[0]
  y_rec_type
  df_corr = pd.DataFrame()
  
  
  
  for i in range(len(analytical_table)):
#     set_trace()
    # i = 10
    n_levels = analytical_table.loc[i, 'n_levels']
    x = analytical_table.loc[i, 'var']
    x_rec_type = analytical_table.loc[i, 'rec_type']
    
    v_cramer = np.nan
    pearson = np.nan
    ratio = np.nan
    kendall = np.nan
    
    # if y_rec_type in ['b', 'bn'] and x_rec_type in ['b', 'bn']: # correlation for pair of binary variables
    #   if analytical_table.loc[analytical_table['var'] == y, 'v_cramer'].values[0] == 1:
    #     v_cramer = correlation_cramers_v(data[x], data[y], round = round)
    if y_rec_type in ['n','bn','cn'] and x_rec_type in ['bn', 'n', 'cn']: # correlation for pair of numeric variables
      if analytical_table.loc[analytical_table['var'] == y, 'pearson'].values[0] == 1:
        pearson = np.round(data[[x,y]].corr().iloc[0,1], round)
      if analytical_table.loc[analytical_table['var'] == y, 'kendall'].values[0] == 1:
        kendall = np.round(data[[x,y]].corr(method = 'kendall').iloc[0,1], round)
    elif (y_rec_type in ['n', 'bn'] and x_rec_type in ['c']) or  (y_rec_type in ['c'] and x_rec_type in ['n', 'bn']): # correlation for pair of numeric and categorical variables
      if analytical_table.loc[analytical_table['var'] == y, 'ratio'].values[0] == 1:
        if y_rec_type in ['c'] and x_rec_type in ['n', 'bn']:
          ratio = correlation_ratio(data[y], data[x], round=round)
        else:
            try: # po 'try' jest blok kodu ktory chcemy obsluzyc pod katem ewentualnego bledu
                ratio = correlation_ratio(data_1[x], data_1[y])
            except: # po 'except' okreslamy co ma sie zadziac po wystapieniu bledu
                print(f'ratio_not_calculated for {x} and {y}.')  # jezeli wystapi jakis wlad 
    if (y_rec_type in ['c', 'cn', 'bn', 'b']) and (x_rec_type in ['c', 'cn', 'bn','b']) and (n_levels > 1): # correlation for pair of categorical variables
      if analytical_table.loc[analytical_table['var'] == y, 'v_cramer'].values[0] == 1:
        # mutual = np.round(skl.metrics.normalized_mutual_info_score(labels_true = data[x], labels_pred = data[y]), round) # normialized mutual value [0,1]
        v_cramer = correlation_cramers_v(data[x], data[y], round = round)
    
    df_corr = pd.concat([df_corr, pd.DataFrame({'var':[x], 'v_cramer_':[v_cramer], 'pearson_':[pearson], 'ratio_':[ratio], 'kendall_':[kendall]})], axis = 0)
  
  result = pd.merge(left = analytical_table_full.loc[:,'index':'kendall'], right=df_corr, left_on=['var'], right_on=['var'])
  result.to_clipboard(index = False)
  return(result)


# data_str = A_F_correlation_selection(data = data, y = 'FLAGA_RESPONSE', analytical_table_file='selekcja_zmiennnnych_auto.xlsx', sheet='Arkusz1')
# correlation_selection(data=data, y='FLAGA_RESPONSE')


def A_F_var_selection(analytical_table_file = 'selekcja_zmiennnnych_auto.xlsx', sheet = 'Arkusz1'):
  
  analytical_table_full = pd.read_excel(analytical_table_file, sheet_name = sheet)
  analytical_table = analytical_table_full.loc[analytical_table_full['remove']==0,]
  
  v_cramer = analytical_table.loc[(analytical_table['v_cramer'] == 1) & (analytical_table['rec_type'].isin(['b','bn']))         , 'var']  
  pearson  = analytical_table.loc[(analytical_table['pearson'] == 1)  & (analytical_table['rec_type'].isin(['n','bn','cn']))    , 'var'] 
  ratio_cat = analytical_table.loc[(analytical_table['ratio'] == 1)   & (analytical_table['rec_type'].isin(['c','cn','b','bn'])), 'var'] 
  ratio_num = analytical_table.loc[(analytical_table['ratio'] == 1)   & (analytical_table['rec_type'].isin(['n','bn','cn']))    , 'var']
  mutual   = analytical_table.loc[(analytical_table['mutual'] == 1)   & (analytical_table['rec_type'].isin(['c','cn','b','bn'])), 'var'] 
  
  return({'v_cramer' : v_cramer, 'pearson':pearson, 'ratio_cat':ratio_cat, 'ratio_num':ratio_num,  'mutual':mutual })

# A_F_var_selection()




def statistics_numeric(data, var_x, var_group = None , x_lim= None, round = 2):
  
  """
  Purpose:
    quantiles of numeric variables that can be grouped by categorical variabls
  
  Input:
    data: DataFrame
    var_x: name of numeric variables
    var_group: name of categorical variable (optional)
    x_lim: limit fo numeric variable values (optional)
  
  Output:
    DataFrame
  """
  
  if var_group is not None:
    data = data[[var_x, var_group]]
    if x_lim is not None:
      data = data.loc[data[var_x].between(x_lim[0], x_lim[1]),]      
    # stats = data.groupby(var_group).agg({var_x:{'min':lambda x:np.nanmin(x), 'q_10': lambda x:np.nanquantile(x, 0.1),'q_25': lambda x:np.nanquantile(x, 0.25), 'q_50': lambda x:np.nanquantile(x, 0.50), 'q_75': lambda x:np.nanquantile(x, 0.75), 'q_90': lambda x:np.nanquantile(x, 0.9),   'max':lambda x:np.nanmax(x) }})
    
    stats = data.groupby(var_group).agg( min_ = pd.NamedAgg(column= var_x, aggfunc=lambda x: np.nanmin(x))
                                        ,q_10 = pd.NamedAgg(column= var_x, aggfunc=lambda x:np.nanquantile(x, 0.10))
                                        ,q_25 = pd.NamedAgg(column= var_x, aggfunc=lambda x:np.nanquantile(x, 0.25))
                                        ,q_50 = pd.NamedAgg(column= var_x, aggfunc=lambda x:np.nanquantile(x, 0.50))
                                        ,q_75 = pd.NamedAgg(column= var_x, aggfunc=lambda x:np.nanquantile(x, 0.75))
                                        ,q_90 = pd.NamedAgg(column= var_x, aggfunc=lambda x:np.nanquantile(x, 0.90))
                                        ,max_ = pd.NamedAgg(column= var_x, aggfunc=lambda x: np.nanmax(x))
                                        )
    
  
  else:
    data = data[[var_x]]
    if x_lim is not None:
      data = data.loc[data[var_x].between(x_lim[0], x_lim[1]),]   
    stats = pd.DataFrame({ 'min': [np.nanmin(data.values)]
            ,'q_05':[np.nanquantile(data.values, 0.05)]
            ,'q_10':[np.nanquantile(data.values, 0.1)]
            ,'q_25':[np.nanquantile(data.values, 0.25)]
            ,'q_50':[np.nanquantile(data.values, 0.50)]
            ,'q_75':[np.nanquantile(data.values, 0.75)]
            ,'q_90':[np.nanquantile(data.values, 0.9)]
            ,'q_95':[np.nanquantile(data.values, 0.95)]
            ,'max':[np.nanmax(data.values)]})
  return(np.round(stats, round) )











###############
# Correlation #
###############







# >>>  V-cramer <<<<
  
def correlation_cramers_v(var_1, var_2, round = 2):
  """
  Purpose: 
    V_cramer coefficent to measure correlation between two binary variables. Code based on: https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
  
  Arguments:
    var_1: DataFrame column with binary variable
    var_2: DataFrame column with variable
  
  Output:
    float
  
  Example of use:
    correlation_cramers_v(var_1 = data['var_1'], var_2 = data['var_2'])
  """
  confusion_matrix = pd.crosstab(var_1, var_2)
  chi2 = scipy.stats.chi2_contingency(confusion_matrix)[0]
  n = confusion_matrix.sum().sum()
  phi2 = chi2/n
  r,k = confusion_matrix.shape
  phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
  rcorr = r-((r-1)**2)/(n-1)
  kcorr = k-((k-1)**2)/(n-1)
  return(np.round( np.sqrt(phi2corr/min((kcorr-1),(rcorr-1))), round) )



def correlation_cramer_v_matrix(data, vars, heatmap = True, width = 10, height = 10, color_threshold = 0.75, round = 2):
  """
  Purpore:
    A wrapper for 'correlation_cramers_v' function. Returns matrix of values for any given set of binary variables.
  
  Arguments:
    data: DataFrame
    vars_bin: names of binary variables i data
    heatmap: if print a heatmap. If not array is returned
    width: witdh of heatmap
    height: height of heatmap
    color_threshold: 
  
  Output:
    DataFrame or matlibplot
  
  """
  n = len(vars)
  ar = np.empty(shape=[n, n])
  for i in range(n):
    for j in range(n):
      if i > j:
        ar[i,j] = correlation_cramers_v(var_1 = data[vars[i]], var_2 = data[vars[j]], round = round)
      else:
        ar[i,j] = np.nan
  
  ar = pd.DataFrame(ar, columns = vars)
  ar.index = vars
  
  if heatmap:
      plot_heatmap(data=np.round(ar, 2) , width = width, height = height, color_threshold = color_threshold)
  else:
      return( ar)


# correlation_cramer_v_matrix(data_, vars=['czy_wynagrodzenie', 'FLAGA_RESPONSE'], heatmap=True)





# >>> correlation ratio <<<

def correlation_ratio(categories, measurements, round=2):
  """
  Purpose:
    Ratio correlation coefficient to measurse dependency between numerical and categorical variables. Code base on https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
  
  Arguments:
    categories: DataFrame column with categorical variable
    measurements: DataFrame column with numerical variable
  
  Output:
    Float
  
  Example of use:
    
  
  """
  fcat, _ = pd.factorize(categories)
  cat_num = np.max(fcat) + 1
  y_avg_array = np.zeros(cat_num)
  n_array = np.zeros(cat_num)
  for i in range(0,cat_num):
      cat_measures = measurements[np.argwhere(fcat == i).flatten()]
      n_array[i] = len(cat_measures)
      y_avg_array[i] = np.average(cat_measures)
  y_total_avg = np.nansum(np.multiply(y_avg_array,n_array))/np.nansum(n_array)
  numerator = np.nansum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
  denominator = np.nansum(np.power(np.subtract(measurements,y_total_avg),2))
  if numerator == 0:
      eta = 0.0
  else:
      eta = np.sqrt(numerator/denominator)
  return(np.round(eta, round))


def correlation_ratio_matrix(data, vars_cat, vars_num, heatmap = True, width = 10, height = 10, round = 2):
  """ 
  Purpose:
    A wrapper for 'correlation_ratio' function. Returns matrix of values for given set of variables.
  
  Arguments:
    data: DataFrame
    vars_cat: list of names of categorical variables
    vars_num: list of names of numerical variables
    heatmap: if print a heatmap. If not array is returned
    width: witdh of heatmap
    height: height of heatmap
   
  Output:
    DataFrame or matlibplot
  
  """
  
  ar = np.empty(shape=[len(vars_cat), len(vars_num)])
  
  for i in range(len(vars_cat)):
    for j in range(len(vars_num)):
      ar[i,j] = correlation_ratio(categories = data[vars_cat[i]], measurements = data[vars_num[j]])
  
  ar = pd.DataFrame(ar, columns = vars_num)
  ar.index = vars_cat
  
  if heatmap:
      plot_heatmap(data=np.round(ar, 2) , width = width, height = height, color_threshold = 0.75)
  else:
      return(np.round(ar, round) )
  

# correlation_ratio_matrix(data=data, vars_cat=['zrodlo_dochodu'], vars_num=['max_dpd_60','CB_PLN','doch_do_dysp','saldo_sum_dep'], heatmap=True)






# >>> pearson correlation <<<


def correlation_pearson(data, vars = None, heatmap = False, width = 10, height = 10, method = 'pearson', round = 2):
  """
  Purpose:
    Pearson correlation matrix for numerical variables
  
  Arguments:
  
  Output:
    DataFrame of matlibplot
  
  """
  if vars is None:
    if heatmap:
      plot_heatmap(data=np.round(data[vars].corr(method = method), 2), width = width, height = height, color_threshold = 0.05)
    else:
      return(np.round(self.data[self.num].corr(method = method), round))
  else:
    if heatmap:
      plot_heatmap(data=np.round(data[vars].corr(method = method), 2), width = width, height = height, color_threshold = 0.05)
    else:
      return(np.round(data[vars].corr(method = method), round))








###############
# Basic plots #
###############

# >>> density plot <<<

def plot_density(data, var_group = None, var_x = None, bw_method = 0.2, x_lim = None, fig_w = 8, fig_h = 8):
  """
  Purpose:
    Grouped density plot base on pandas.
  
  Arguments:
    data: DatFrame
    var_group: optional name of categorical variable for grouping plot
    var_x: numerical variable name
    bw_method: adjust coefficient
    x_lim: oprional boundries for numerical variable values.
    fig_w: plot width
    fig_h: plot hight
    
  Output:
    matlibplot
    
  Example of use:
    plot_density(data = data_set, var_group='FLAGA_RESPONSE', var_x = 'max_dpd_60', bw_method=0.3, x_lim=[0, 20])
  """
  plt.figure()
  fig, axes = plt.subplots(1, 2, figsize = (fig_w, fig_h))
  
  
  if x_lim is not None:
    data = data.loc[data[var_x].between(x_lim[0], x_lim[1]),]
  
  if var_group is not None:
    sns.boxplot(x=data[var_x],
                 ax = axes[0],
                 y=data[var_group],  
                 width=0.5,
                 orient='h')
    for label, df in data.groupby(var_group):
      df[var_x].plot(kind="kde", label=label, bw_method=bw_method, ax = axes[1])
      data[var_group] = data[var_group].astype('category')
  else:
    # data[var_group] = data[var_group].astype('category')
    sns.boxplot(x=data[var_x],
                 ax = axes[1],
                 width=0.5,
                 orient='h')
    data[var_x].plot(kind="kde", bw_method=bw_method, ax = axes[0])
  plt.legend()
  plt.show()



# plot_density(data.head(100000), var_x='saldo_sum_dep', var_group='FLAGA_RESPONSE', x_lim=[0,3000], bw_method=0.2)



# >>> density box <<<
def plot_box(data, var_group = None, var_x = None, x_lim = None, fig_w = 8, fig_h = 8):
      """
      Purpose:
        Grouped box plot base on pandas.
      
      Arguments:
        data: DatFrame
        var_group: optional name of categorical variable for grouping plot
        var_x: numerical variable name
        x_lim: oprional boundries for numerical variable values.
        fig_w: plot width
        fig_h: plot hight

      Output:
        matlibplot
      
      Example of use:
        plot_density(data = data_set, var_group='FLAGA_RESPONSE', var_x = 'max_dpd_60', bw_method=0.3, x_lim=[0, 20])
      """
      if x_lim is not None:
        data = data.loc[data[var_x].between(x_lim[0], x_lim[1]),]
      
      if var_group is None:
          sns.boxplot(y=var_x, 
                 data=data, 
                 width=0.5,
                 orient='h',
                 palette="colorblind")
      else:
        data[var_group] = data[var_group].astype('category')
        sns.boxplot(x=data[var_x],
                 y=data[var_group],  
                 width=0.5,
                 orient='h',
                 palette="colorblind")
      
      plt.show()



# plot_box(data.sample(frac = 0.05), var_x='CLS_CRR_PRINC_BALANCE_AMT_PLN', var_group = 'FLAGA_RESPONSE', x_lim=[-10,5000])


# >>> heatmap <<<

def plot_heatmap(data = None, width = 10, height = 10, color_threshold = 0.75):
  """
  Purpose:
    Plotting heatmap useful for correlation plots.
  
  Arguments:
    data: DataFrame or array. If DataFrame it is recommended that columns and Indicies are named.
    width: width of plot
    height: width of plot
    color_threshold: if absolute values in cell is above threshold then the color of the backgroud is red. Otherwise is blue.
  
  Output:
    matlibplot
  """
  
  plt.figure(figsize=(width, height))
  mask = np.zeros_like(data, dtype=np.bool)
  mask[np.triu_indices_from(mask, k=1)] = True
  ax = sns.heatmap(
      data,
      mask=mask,
      square=True,
      annot=True,
      fmt='.2f',
      cmap='Blues',
      vmin=0.,
      vmax=1,
      cbar_kws={'shrink': .75}
  )
  
  mask[np.abs(data) <= color_threshold] = True
  mask[np.triu_indices_from(mask)] = True
  
  sns.heatmap(
    data,
    mask=mask,
    square=True,
    annot=False,
    cmap='YlOrRd',
    cbar=False
  )
  
  lw = 2
  ax.axvline(x=0, color='k',linewidth=lw)
  ax.axhline(y=data.shape[1], color='k',linewidth=lw)
  ax.axhline(y=round((data.shape[1]*(data.shape[1]-1)-1.0)/data.shape[1]), color='k', linewidth=lw * 0.5)
  
  plt.tight_layout()
  plt.show()





#########################################
# Variables transformation and encoding #
#########################################


# >>> categorical encoding <<<

def categorical_features_recoding(data, var, dict_old_new, else_value = None, new_name = None):
  """
  Purpose:
    Easy way to rocode categorical variables with use of dictionary
    
  Arguments:
    var: name of column to recode
    dict_new_old: dictonary on how to rocode values of variable
    else_value: if provided all values not mapped in dictionary will be replaced with this value
    new_name: name of new recoded variable. If None then current variable will be overwritten.
    
  Output:
    DataFrame
    
  """
  if new_name is None:
    new_name = var
  data[new_name] = data[var].replace(to_replace = dict_old_new)
  if else_value is not None:
    data.loc[~data[new_name].isin(list(dict_old_new.values())), new_name] = else_value
  
  
  return(data)


# return(data)


# >>> target encoding <<<

def target_encoding(data, var_x, var_y, var_y_convert_to_float = True):
    data_ = copy.deepcopy(data)
    encode_ = ce.TargetEncoder(cols = var_x )
    if var_y_convert_to_float:
        t1 = encode_.fit(X=data_[var_x], y = data_[var_y].astype('float64'))
    else:
        t1 = encode_.fit(X=data_[var_x], y = data_[var_y])
    x = t1.transform(X=data_[var_x])
    data_[var_x] = pd.DataFrame(x, columns = var_x, index = data.index)
    return(data_)

# target_encoding(data = data_1, var_x=['czy_wynagrodzenie'], var_y=y)


# >>> numerical normalization <<<


def numerical_features_normalization(data, vars_to_normalize, method = 'standarization'):
    if method == 'standarization':
        data[vars_to_normalize] = pd.DataFrame(preprocessing.StandardScaler().fit(data[vars_to_normalize]).transform(data[vars_to_normalize]), columns = vars_to_normalize, index = data.index)
    if method == 'min_max_normalization':
        data[vars_to_normalize] = pd.DataFrame(preprocessing.MinMaxScaler().fit(data[vars_to_normalize]).transform(data[vars_to_normalize]), columns = vars_to_normalize, index = data.index)
    if method == 'robust_scaler':
        data[vars_to_normalize] = pd.DataFrame(preprocessing.RobustScaler().fit(data[vars_to_normalize]).transform(data[vars_to_normalize]), columns = vars_to_normalize, index = data.index)
    return(data)


# >>> numerical transformations <<<

def numerical_features_transform(ts, imputation = None, var_y = None, n_lag = None, diff_lag = None, log = False,  inv_log = None, asinh = None, box_cox = True, inv_box_cox = None, normalisation = True, new_names = True):
  """ 
  Purpose:
    To make many tranformation of numerical variable at once. Name for new tranformed variables are created automatically
    
  Arguments:
    ts: DataFrame
    imputation: imputation strategy (optional)
    var_y: name of numerical variable to transform
    n_lag: for creating lagged variable. Integer value tells how big lag should be.
    diff_lag: for creating differential variable. Integer value tells how big gap between two values shoud be.
    log: logaritm base of e.
    inv_log: inverse of logaritm
    asinh: arcus sinuous
    box_cox: box cox tranformation
    inv_box_cox: inverse box cox tranformation. Value for tranformation must be provided.
    normalisation: mean/std
    new_names: if tranform variables should be assing to new columns
  
  Output:
    DataFrame
  
  """
  
  # creating variables by lagging y
  if n_lag is not None:
    for i in n_lag:
      ts[var_y + '_lag_'+str(i)] = ts[var_y].shift(periods=i)
    lag_names = [ var_y + '_lag_' + str(i)  for i in n_lag]
  
  # creating variables by differenciating y
  if diff_lag is not None:
    for j in diff_lag:
      ts[var_y + '_diff_'+str(j)] = ts[var_y].diff(periods=j)
    diff_names = [var_y + '_diff_' + str(i)  for i in diff_lag]
  
  if log:
    ts[var_y + '_log'] = np.log(ts[var_y].values)
  
  if inv_log:
    ts[var_y + '_invlog'] = np.e**ts[var_y].values
    
  if asinh:
    ts[var_y + 'asinh'] = np.arcsinh(data[var_y].values)
  
  if box_cox:
    y_boxcox = stats.boxcox(ts[var_y].values)
    ts[var_y + '_box'] = y_boxcox[0]
    cox_box_coeff =  y_boxcox[1]
  else:
    cox_box_coeff = 'None'
  
  if inv_box_cox is not None:
    ts[var_y + 'inv_box'] = inv_boxcox(ts[var_y].values, inv_box_cox)
  
  if normalisation:
    ts[var_y + '_norm'] = (ts[var_y].values - np.nanmean(ts[var_y].values)) / np.nanstd(ts[var_y].values)
  
  
  if diff_lag is not None and n_lag is not None:
    all_new_names = lag_names + diff_names
  elif n_lag is not None:
    all_new_names = lag_names
  elif diff_lag is not None:
    all_new_names = diff_names
  else:
    all_new_names = None
  
  if all_new_names is not None and imputation is not None:
    my_imputer = Imputer(strategy=imputation, axis=0)
    my_imputer = my_imputer.fit(ts[all_new_names])
    imputed_df = pd.DataFrame(my_imputer.transform(ts[all_new_names]))
    ts[all_new_names] = imputed_df
  # print(cox_box_coeff)
  # self.data = ts
  return(ts)












#####################
# Features rankings #
#####################




# >>> rand_var <<<


def rand_var(data = None, prexid = 'rand_'):
  data_size = data.shape[0]
  data[prefix + 'binary']  = np.random.randint(0, 2, data_size)
  data[prefix + 'cat']     = np.random.randint(0, 5, data_size)
  data[prefix + 'uniform'] = np.random.uniform(0, 1, data_size)
  data[prefix + 'normal']  = np.random.normal(0, 1, data_size)
  
  return(data)






# >>> Feature importance from package catboot <<<

def feature_importance_class_CB(iterations = 30, learning_rate = 0.1,  depth = 5, cat_features = None, x_train = None, y_train = None):
    """ 
    Purpose:
    feature importance with CataBoost for categorical objective feature 
    """
    
    if cat_features is not None:
        x_train.loc[:,cat_features] = x_train.loc[:,cat_features].astype(str)
        #         x_train.loc[:,cat_features] = x_train.loc[:,cat_features].fillna('zzz')
        cat_indicies = [x_train.columns.get_loc(c) for c in cat_features if c in x_train]
    else:
        cat_indicies = None
    cb = CatBoostClassifier(  iterations = iterations, learning_rate=learning_rate, depth=depth)
    cb.fit(x_train, y_train, plot = False, cat_features = cat_indicies,  use_best_model=True, early_stopping_rounds=75,  silent=True)
    
    return(np.round(cb.get_feature_importance(prettified=True), 3))




# >>> Feature importance from sklearn based on RandomForest <<<

def feature_importance_class_RF(n_estimators, x_train, y_train):
  """ 
  Purpose:
    feature importance with RandomForest for categorical objective feature
  """
  rf = RandomForestClassifier(  n_estimators = n_estimators
                               ,n_jobs = -1
                               ,oob_score = True
                               ,bootstrap = True
                               ,random_state = 42)
  rf.fit(x_train, y_train)
      
  def r2(rf, x_train, y_train):
    return(r2_score(y_train, rf.predict(x_train)))
  
  return( np.round(permutation_importances(rf, x_train, y_train, r2), 2) )


# >>> Feature information from sklearn based on Permutation Importance <<<




# OUTLIERES

# >>> Isolation Forest <<< 

def outlieres_IF(n_estimators, x_train):
  """ 
  Purpose:
    Multidimentional analysis of outliers with Isolation Forest
    
  Arguments
    n_estimators: number of estimators for algorithm
    x_train: DataFrame
    
  Output:
    DataFrame with added extra column with witch idicate if observation is as an outlier or not.
  
  """
  iforest=IsolationForest(n_estimators=n_estimators, max_samples='auto', contamination=float(.12), max_features=1.0, bootstrap=False, n_jobs=5, random_state=42, verbose=0)
  iforest.fit(x_train)
  pred = iforest.predict(x_train)
  x_train['anomaly'] = pred
  return(x_train)




############################
# Missing values imputation #
############################


# >>> basic numerical imputation <<<

def Imputation_numeric(   data
                        , var               = None
                        , missing_indicator = True
                        , method_min        = True
                        , method_median     = False
                        , method_mean       = False
                        , method_prev_value = False
                        , method_value      = False
                        , value = None): # if 'method_value'  you have to provede this value
  """
  Purpose:
    Basic imputation technique for numerical variables. More than one type of imputation can be done at once. New names for imputed variable are created automatically.
    
  Arguments:
    data: DataFrame
    var: name of varible to impute
    ...
    
  Output:
    DataFrame
    
  Example of use:
   data_1 = pd.DataFrame({'a':[1,2,np.nan], 'b':['a','b','c']})
   Imputation_numeric(data_1, var='a', method_median=True, method_mean=True, method_prev_value=True, method_min=True)
    
  """
  # check if there is any missing value
  if sum(np.isnan(data[var])) == 0:
    raise ValueError('there are no missing values to impute')
    
  # indicator if value is missing
  if missing_indicator:
    data[var + '_is_missing'] = np.isnan(data[var])
  
  # impute with min(variable) - range(variable)
  # if method_min:
    value_to_imput = np.nanmin(data[var]) - (np.nanmax(data[var]) - np.nanmin(data[var]))
    data[var + '_imput_min'] = data[var].fillna(value = value_to_imput)
  
  # impute with median
  if method_median:
    value_to_imput = np.nanmedian(data[var])
    data[var + '_imput_median'] = data[var].fillna(value = value_to_imput)
  
  # impute with mean
  if method_mean:
    value_to_imput = np.nanmean(data[var])
    data[var + '_imput_mean'] = data[var].fillna(value = value_to_imput)
  
  # impute with previous value
  if method_prev_value:
    data[var + '_imput_prev'] = data[var].fillna(method='ffill')
    
  if method_value:
    data[var + '_imput_value'] = data[var].fillna(value)
    
  return(data)





###########################
# IML (under development) #
###########################

# >>> shapley value <<<

def shapley_value(model, x_train, y_train, sample_size = 1000):
    model.fit(x_train, y_train)

    x_train_samp = shap.sample(x_train, sample_size)

    explainer = shap.KernelExplainer(model.predict_proba, x_train_samp)

    shap_values = explainer.shap_values(x_train_samp)

    shap.summary_plot(shap_values, x_train_samp)
    
    shap.initjs()
    shap.summary_plot(shap_values[0], x_train_samp)
    
    for i in list(x_train_samp.columns):
        shap.dependence_plot(i, shap_values[0], x_train_samp)

    shap.initjs()
    display(shap.force_plot(explainer.expected_value[0], shap_values[0], x_train_samp))




# >>> Parial Dependence Plot (PDP plot)

def PDP_plot_RF(model, x_train, y_train, var):
    """
    Purpose:
    Partial Dependence Plot base on RandomForest
    """
    var_index = x_train.columns.get_loc(var)
    
    from sklearn.inspection import plot_partial_dependence
    from sklearn.inspection import partial_dependence
    
    model.fit(x_train, y_train)

    t1 = model.predict_proba(X=x_train)

    pdp, axes = partial_dependence(model, x_train, features = [var_index])

    pdp_df = pd.DataFrame({'probability':pdp[0], 'category':axes[0]})

    display(ggplot(pdp_df) + geom_line(aes(x='category', y='probability')))




# >>> L1 with logistic regression


def feature_importance_logistic_regression(x_train=None, y_train=None, plot = False, normalize_data = 'all', label_shift = -2, penalty = 'l1'):
    
    # check missing values in data set
    missing = x_train.isna().sum()
    if max(missing) > 0:
        print(missing)
    
    clf.fit(x_train, y_train)
    coef = clf.coef_
    imp_coef = pd.DataFrame(coef.T)
    imp_coef.columns = ['coef']
    imp_coef.index = list(x_train.columns)
    imp_coef['coef_abs'] = imp_coef['coef'].abs()
    imp_coef  = imp_coef.sort_values(['coef_abs'], ascending=True)
    imp_coef = imp_coef.reset_index(drop=False)
    imp_coef = np.round(imp_coef, 2)
    if plot==False:
        return(imp_coef)
    else:
        dodge_text = position_dodge(width=1.9)
        imp_coef['index_1'] = pd.Categorical(imp_coef['index'], categories=imp_coef['index'])
        return(ggplot(imp_coef, aes(x='index_1', y='coef_abs')) 
               + geom_bar(stat='identity', position='dodge', show_legend=False) 
               + coord_flip()
               + geom_text(aes(y = label_shift, label='coef_abs'), position=dodge_text,size=12, va='bottom', format_string='{}')) 




# >>> Elastic net with logistic regression <<<


from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing

def feature_importance_class_elastic_net(l1_ratio = 0.5, x_train=None, y_train=None, plot = False,  label_shift = -2):
    
    # check missing values in data set
    missing = x_train.isna().sum()
    if max(missing) > 0:
        print(missing)
    
    elastic = SGDClassifier(loss = 'log', penalty = 'elasticnet', l1_ratio=l1_ratio )
    elastic.fit(x_train, y_train)
    coef = pd.Series(elastic.coef_[0], index = list(x_train.columns))
    
    imp_coef = pd.DataFrame(coef)
    imp_coef.columns = ['coef']
    imp_coef['coef_abs'] = imp_coef['coef'].abs()
    imp_coef  = imp_coef.sort_values(['coef_abs'], ascending=True)
    imp_coef = imp_coef.reset_index(drop=False)
    imp_coef = np.round(imp_coef, 2)
    if plot==False:
        return(imp_coef)
    else:
        dodge_text = position_dodge(width=0.9)
        imp_coef['index_1'] = pd.Categorical(imp_coef['index'], categories=imp_coef['index'])
        return(ggplot(imp_coef, aes(x='index_1', y='coef_abs')) 
               + geom_bar(stat='identity', position='dodge', show_legend=False) 
               + coord_flip()
               + geom_text(aes(y = label_shift, label='coef_abs'), position=dodge_text,size=12, va='bottom', format_string='{}')) 





### Other things

def sbs_(dfs:list, captions:list):
    
    """
    Displaying many Data Frames with captions in one row.
    """
    
    from IPython.display import display, HTML
    display(
        HTML(data=""" 
    <style> 
        div#notebook-container    { width: 95%; }
        div#menubar-container     { width: 65%; }
        div#maintoolbar-container { width: 55%; }
    </style> 
    """))
    """Display tables side by side to save vertical space
    Input:
        dfs: list of pandas.DataFrame
        captions: list of table captions
    """
    output = ""
    combined = dict(zip(captions, dfs))
    for caption, df in combined.items():
        output += (df.style
                   .set_table_attributes("style='display:inline'").set_caption(caption)._repr_html_()
                   .format(formatter="{:,}")
                  )
        output += "\xa0\xa0\xa0"
    display(HTML(output))

