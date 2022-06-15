






import pandas      as pd
import numpy       as np
import sklearn     as skl
import statsmodels as stm
# import keras       as ker
import plotnine
import seaborn     as sb
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
import os
import itertools   as it
import sklearn.metrics as skm
import datetime
import math
import copy # deepcopies
import scipy
import imblearn   # problem of imbalanced samples

from scipy            import stats
from scipy.special    import boxcox, inv_boxcox

from rfpimp           import permutation_importances
from sklearn.metrics  import r2_score # scores for assessment model performance
from sklearn.ensemble import IsolationForest # for outliered detecting



from imblearn.over_sampling    import RandomOverSampler
from imblearn.over_sampling    import SMOTE
from imblearn.under_sampling   import RandomUnderSampler
from plotnine.themes.themeable import axis_line
from sklearn.model_selection   import train_test_split
from sklearn.ensemble          import AdaBoostClassifier
from sklearn.ensemble          import RandomForestClassifier
from sklearn.ensemble          import GradientBoostingClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.naive_bayes       import MultinomialNB
from lightgbm                  import LGBMClassifier
# from xgboost                   import XGBClassifier
from sklearn.svm               import SVC
from sklearn.preprocessing     import OneHotEncoder
from sklearn.calibration       import CalibratedClassifierCV
# from sklearn.preprocessing     import Imputer
from  lightgbm import LGBMClassifier
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing
from sklearn.impute import SimpleImputer as Imputer
import catboost
from catboost                  import CatBoostClassifier, Pool, cv
from plotnine                  import * # ggplot for python
import shap
# https://www.kaggle.com/discdiver/category-encoders-examples
import category_encoders as ce
from IPython.display import display_html

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc

from IPython.display import display, HTML

import gc

from statsmodels.stats.outliers_influence import variance_inflation_factor

# DISPLAY OPTIONS
import warnings
warnings.filterwarnings('ignore')
pd.set_option('precision', 3)
# pd.set_option('display.float_format', lambda x: '%20.3f' % x)
# pd.set_option('precision', 2)
# pd.options.display.float_format = '{:,}'.format
# pd.options.display.max_columns=100
# pd.options.display.max_rows=500
# pd.options.display.max_colwidth=30










#### start


# obiekt do zagadnien klasyfikacjnych (binary old multiclass)
class classification_model():
  """
  # metoda dzialania obiektu
  
  # - Podaje 2 zbiory: treningowy i testowy
  # - W ramach zbioru treningowego mozna przeprowadzać pelna estymacja (metoda FULL) oraz uzyc algorytmu z samplingiem (CV, HOLDOUD. Wtedy powstaja zbiory 'valid' poza 'test'). Bootstrap nie jest w tej chwili zaimplementowany.
  # - Wykonanie każdej symulacji oznacza odłozenie danych. W ramach poszczególnej symulacji mozna jednocześnie uzyc kilka metod statystycznych dostepnych w pakiecie sklearn i catboost (catboost aktualnie nie obsluguje zagadnienia wieloklasowego). Ale struktura zbioru, punkt odciencia itp. moga są róznic tylko miedzy symulacjami
  # - W ramach symulacji odkładane są przede wszystkim scory, prawdopodobieństwa, decyzje modelu i macierze pomylek, oraz różne metadane.
  
  # Uwagi:
  # W threshold podajemy punkty odciencia poprzez slownik. Wazne zeby nazwy kluczy byly tego samego typu (najlepiej 'str') co typ zmiennej objasnianej (y) w zbiorze
  # zaleca sie aby zmienna objasniana byla typu 'str', ale z mozliwoscia przekonwertowania na int ( uzywa tego teraz np. target encoding oraz wyliczanie scorow dla binarnego targetu).
  # obecnie target encoding nie obsluguje zmiennej objasnianej typu str. Dlatego na razie jest zaimplementowana automatyczne konwersja na 'int', ale to dziala oczywiscie pod warunkiem ze taka konwersja zmiennje objaśnianej jest mozliwe
  # kalibracja dziala na razie chyba tylko dla sklerna.
  # Catboost robi automatyczny target encoding. Dla tej metody opcjonalnie liste zmiennych kategorycznych podajemy w 'models_fit_parameters'.
  # Dla feature importance permutation (fip) mozna dac tylko jeden model.
  # Samplingi (hold, cv itp) sa robione tak aby kazdy model byl liczony na tych samych zbiorach
  # W przypadku oversamplingu nie sa zapisywane poprawnie inforamcje o indeksach. Jest to generalnie problem tworzenia danych synstetycznych
  
  
  
  # modele
    ### data = r.diamonds
    ### data = categorical_features_recoding(data, var='clarity', dict_old_new = {'I1':'0','SI2':'0','SI1':'0','VS2':'0','VS1':'1','VVS2':'1', 'VVS1':'1', 'IF':'1'}, else_value = None, new_name = 'target')
    ### data.to_csv('diamonds.txt')
    
    # data = pd.read_csv('diamonds.txt')
    
    # models_list = {'rand1':RandomForestClassifier(n_estimators=30, max_depth=6)}
    
    # models_parameters = None
    
    # # zmienne do modelu
    
    # y_var = ['target']
    # x_var = ['price', 'x', 'color','cut','y', 'z']
    
    # x_var_num = ['price', 'x', 'y', 'z']
    # x_var_cat = ['color','cut']
    
    # x_var = x_var_num + x_var_cat
    
    # # podstawowy zbior danych
    
    # x_train, x_test, y_train, y_test = skl.model_selection.train_test_split(data[x_var], data[y_var], test_size=0.33)
    # x_train = x_train.reset_index(drop=True)
    # x_test  = x_test.reset_index(drop=True)
    # y_train = y_train.reset_index(drop=True)
    # y_test  = y_test.reset_index(drop=True)
    
    # train_set = pd.concat([x_train, y_train], axis = 1)
    # test_set  = pd.concat([x_test, y_test], axis = 1)
    
    # train_set[y_var] = train_set[y_var].astype(str)
    # test_set[y_var]  = test_set[y_var].astype(str)
    
    
    
    # # EXAMPLE OF USE
    # # %run "F_modeling.py"
    # models_base = {  'rf' :RandomForestClassifier(n_estimators=10, max_depth=6)
    #                 ,'gb' :GradientBoostingClassifier(n_estimators=10)
    #                 # ,'cat':CatBoostClassifier(iterations=10,  random_seed=123)
    #                 ,'ab' :AdaBoostClassifier(n_estimators = 30)
    # #                 ,'svc':skl.svm.SVC(C=1.0, kernel = 'rbf', degree=3, probability=True) # It takes long time to calculate
    #                 ,'knn':skl.neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=10, p=2, metric='minkowski')
    #                 ,'nb' :skl.naive_bayes.GaussianNB(priors=None, var_smoothing=1e-09)
    # #                 ,'qda':skl.discriminant_analysis.QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0, store_covariance=False, tol=0.0001)
    # #                 ,'gp' :
    #                 }
    
    # simulation_name = 'sim_1'
    # simulation_description = 'symulacja'
    # models_list = {'rf':models_base['rf'], 'ab':models_base['ab'] }
    # threshold = {'0':0.5, '1':0.5}
    # cv_n  = 2
    # hold_n = 4; hold_train_sample_size = 0.1; hold_test_sample_size = 0.1
    # fip_n = 1
    # pos_label = '1' #
    
    
    # c1 = classification_model()
    
    # c1.simulation(  train_set = train_set, test_set = test_set, y_var = y_var, x_var_num = x_var_num, x_var_cat = x_var_cat
    #               , balancing_method='over'
    #               , cv_n=cv_n, fip_n=fip_n, hold_n = hold_n, hold_train_sample_size = hold_train_sample_size, hold_test_sample_size = hold_test_sample_size
    #               , data_check=True, models_list=models_list, correlation = True, model_fip = RandomForestClassifier(n_estimators=30, max_depth=7)
    #               , simulation_name = simulation_name, description = simulation_description)
    
    
    # c1.performance(simulation_name=simulation_name, threshold=threshold, priori =  None)

    # # h('meta informations about simulation')
    # c1.simulation_metainfo.loc[c1.simulation_metainfo['simulation_name'] == simulation_name]
    # # h('correlation between features')
    # c1.fast_correlation_matrices(simulation_name = simulation_name)
    # # h('target distribution, confussion matrix, scores and feature importance')
    # c1.fast_conf_matrix_scores(simulation_name = simulation_name)
    # # h('density plots')
    # c1.fast_dens_plot(simulation_name=simulation_name, sample_type = ['full'])
    # # h('density plot for Feature Importatnce Permutation')
    # c1.fast_dens_plot(simulation_name=simulation_name, sample_type = ['fip'])
    # # h('scores by cross validation')
    # c1.fast_scores_plot(simulation_name=simulation_name, sample_type = 'cv')
    # # h('scores by holdout')
    # c1.fast_scores_plot(simulation_name=simulation_name, sample_type = 'hold')
    # # h('pr and roc curve')
    # c1.plot_roc_pr_curves(simulation_name=simulation_name, set_type='test', sample_type = ['full'], y_category=pos_label)
    # # h('pr and roc curve for Feature Importatnce Permutation')
    # c1.plot_roc_pr_curves(simulation_name=simulation_name, set_type='test', sample_type = ['fip', 'full'], y_category=pos_label, group = 'model_name', facet = 'sample_nr~variable')
  
  """
  
  def __init__(self):
    """ 
    initialization of variables to store results of calculations.
    """
    
    # DIFFERENT METADATA : 
    
    # metadata about simulations
    self.simulation_metainfo = pd.DataFrame()
    
    # models hyperparameters
    self.models  = {}
    
    # priori weights and thresholds
    self.priori    = dict()
    self.threshold = dict()
    
    # labels of categories for dependent variable
    self.y_labels = []
    
    # type of classification problem
    self.classification_problem = ''
    
    # names and indicies for categorical and numerical variables
    self.cat_vars          = []
    self.cat_vars_indicies = []
    self.num_vars          = []
    self.num_vars_indicies = []
    
    # information about missing values and variables types in trai set
    self.data_check_result = pd.DataFrame()
    
    # count
    self.y_train_count = pd.DataFrame()
    self.y_test_count  = pd.DataFrame()
    
    
    # SIMULATION AND PERFORMANCE RESULTS : 
    
    
    # correlation matrices
    self.correlation_numeric_pearson      = dict()
    self.correlation_numeric_kendall      = dict()
    self.correlation_mix_ratio            = dict()
    self.correlation_categorical_v_cramer = dict()
    self.vif = dict()
    
    # scores
    self.scores             = pd.DataFrame()
    
    # probabilities
    self.probabilities      = pd.DataFrame()
    
    # classification decisions
    self.classification     = pd.DataFrame()
    
    # confusion matrices
    self.confusion_matrix   = pd.DataFrame()
    
    # feature importatnce
    self.feature_importance = pd.DataFrame()
    self.feature_importance_enet = pd.DataFrame()
  
  
  
  def data_check( self
                , data
                , set_name = 'train'
                , x_var    = None
                , y_var    = None
                , x_var_cat = None):
      
    """
    checking number of missing values and columns types on train set. 
    """
    
    # if duplicated index
    if any(data.index.duplicated()):
      raise ValueError('LUCAS: ' +  set_name +  ' data sets has duplicated indicies')
    
    
    # if variable is a DataFrame
    if str(type(data)).find('DataFrame') < 1:
      raise ValueError('LUCAS: ' +  set_name +  ' data sets is not Pandas DataFrame')
    
    # checking data length
    if len(data)==0:
        raise ValueError('LUCAS: length of '+ set_name + ' data set is 0')
    
    # checking if data includes all variables
    if y_var is None or x_var is None:
      raise ValueError('LUCAS: variables names not provided')
    if set(x_var) != set(x_var).intersection(set(data.columns)):
      raise ValueError('LUCAS: not all features variables are included in ' + set_name + ' set')
    elif set([y_var]) != set([y_var]).intersection(set(data.columns)):
      raise ValueError('LUCAS: target variable is not included in ' + set_name + ' set')
    
    # checking if all 'object' columns are included in 'x_var_cat'
    cols_types = pd.DataFrame(data.dtypes, columns = ['type']).reset_index(drop = False)
    cols_types = cols_types.loc[(cols_types['type'] == 'object') & (cols_types['index'] != y_var ) , :]
    if len(x_var_cat) > 0:
        if x_var_cat is None:
            raise ValueError('LUCAS: Data contains string columns but not x_var_cat is specified')
        elif len(set(cols_types['index'])-set(x_var_cat)) > 0 :
            raise ValueError('LUCAS: In data set there are object columns not specified in "x_var_cat"')
    
    
    # checking if number o target categories > 1
    if len(data[y_var].drop_duplicates()) < 2:
      raise ValueError('LUCAS: In target variable there is only 1 category. For classification task at least 2 are required')
    
    # checking missing value and data type
    df_na = data.apply(lambda x : sum(pd.isnull(x))  ).T
    df_types = data.dtypes
    
    df_na = pd.concat([df_na, df_types], axis = 1)
    df_na.columns = ['missing','type']
    self.data_check_result = df_na
    if len(df_na.loc[ df_na['missing'] > 0, :]) > 0:
      print(df_na)
      raise ValueError('LUCAS: there are missing values in data set. Get attibute "data_check_result" to see details' )
    
    # checking infinity values
    df_num = data.select_dtypes('number')
    df_inf = pd.DataFrame(np.isinf(df_num), columns = list(df_num.columns)).sum()
    if len(df_inf[df_inf == True]) > 0:
      print(df_inf)
      raise ValueError('LUCAS: there are infinite values in data set')
    
    return([df_na, df_inf])
    
  
  
  
  def remove_simulations(self, simulations_names, keep = False):
    
    """ 
    removing listed simulations from all objects
    """
    
    if keep:
      simulations_names = set(list(self.simulation_metainfo['simulation_name'])).diffecence(set(simulations_names))
      simulations_names = list(simulations_names)
    
    # self.simulation_metainfo.drop(self.simulation_metainfo[ self.simulation_metainfo['simulation_name'].isin(simulations_names) ].index, inplace=True ) # ku przetrodze
    self.simulation_metainfo = self.simulation_metainfo[ ~self.simulation_metainfo['simulation_name'].isin(simulations_names) ] # ku przestrodze
    for x in simulations_names:
      try:
        del self.models[x]
      except:
        print('no ' + x + ' found in self.models' )
      try:
        del self.priori[x]
      except:
        print('no ' + x + ' found in self.priori' )
      try:
        del self.threshold[x]
      except:
        print('no ' + x+ ' found in self.threshold' )
      try:
        del self.correlation_numeric_pearson[x]
      except:
        print('no ' + x+ ' found in self.correlation_numeric_pearson' )
      try:
        del self.correlation_numeric_kendall[x]
      except:
        print('no ' + x+ ' found in self.correlation_numeric_kendall' )
      try:
        del self.correlation_numeric_ratio[x]
      except:
        print('no ' + x+ ' found in self.correlation_numeric_ratio' )
      try:
        del self.correlation_numeric_v_cramer[x]
      except:
        print('no ' + x+ ' found in self.correlation_numeric_v_cramer' )
    
   
    try:
      self.y_train_count = self.y_train_count[ ~self.y_train_count['simulation_name'].isin(simulations_names)]
    except:
      print('no ' + x + ' found in self.y_train_count' )
    
    try:
      self.y_test_count = self.y_test_count[ ~self.y_test_count['simulation_name'].isin(simulations_names)]
    except:
      print('no ' + x + ' found in self.y_test_count' )
    
    try:
      self.scores           = self.scores[ ~self.scores['simulation_name'].isin(simulations_names)]
    except: 
      print('no ' + x + ' found in self.scores' )
    
    try:
      if len(self.probabilities) == 0:
        print('no ' + x + ' found in self.probabilities' )
      self.probabilities    = self.probabilities[ ~self.probabilities['simulation_name'].isin(simulations_names) ]
    except: 
      print('no ' + x + ' found in self.probabilities' )
    
    try:
      if len(self.classification) == 0:
        print('no ' + x + ' found in self.classification' )
      self.classification   = self.classification[ ~self.classification['simulation_name'].isin(simulations_names) ]
    except: 
      print('no ' + x + ' found in self.classification' )
    
    try:
      self.confusion_matrix = self.confusion_matrix[ ~self.confusion_matrix['simulation_name'].isin(simulations_names) ]
    except: 
      print('no ' + x + ' found in self.confusion_matrix' )
    
    try:
      self.feature_importance           = self.feature_importance[~self.feature_importance['simulation_name'].isin(simulations_names) ]
    except:
      print('no ' + x + ' found in self.feature_importance' )
  
  
  def probabilities_determining(self
                              , y_true
                              , prob
                              , simulation_name
                              , model_name
                              , sample_type = None
                              , sample_nr   = None
                              , sample_name = None
                              , set_type    = None):
      """
      Determining probabilities from the model
      """
      
      prob.insert(0, column='index' , value = prob.index)
      
      prob = prob.reset_index(drop=True)
      
      prob.insert(0, column = 'y_true',          value = y_true.values)
      prob.insert(0, column = 'set_type',        value = set_type)
      prob.insert(0, column = 'sample_nr',       value = sample_nr)
      prob.insert(0, column = 'sample_name',     value = sample_name)
      prob.insert(0, column = 'sample_type',     value = sample_type)
      prob.insert(0, column = 'model_name',      value = model_name)
      prob.insert(0, column = 'simulation_name', value = simulation_name)

      self.probabilities = pd.concat([self.probabilities, prob], ignore_index=True)
      
      
  
  
  def classification_decision_automatic(self
                              , x
                              , y_true
                              , model
                              , simulation_name
                              , model_name
                              , sample_type = None
                              , sample_nr   = None
                              , sample_name = None
                              , set_type    = None):
      """
      Making model classification with use of default 'proba' method
      """
      classification = pd.DataFrame(model.predict(x), columns=['y_pred'])
      classification.insert(0, column = 'y_true',          value = y_true.values)
      classification.insert(0, column = 'index',           value = x.index)
      classification.insert(0, column = 'set_type',        value = set_type)
      classification.insert(0, column = 'sample_nr',       value = sample_nr)
      classification.insert(0, column = 'sample_name',     value = sample_name)
      classification.insert(0, column = 'sample_type',     value = sample_type)
      classification.insert(0, column = 'model_name',      value = model_name)
      classification.insert(0, column = 'simulation_name', value = simulation_name)
      classification.insert(0, column = 'if_automatic',    value = 1)
      classification.insert(0, column = 'threshold_priori_id',    value = '')
      self.classification = pd.concat([self.classification, classification], ignore_index=True)
  
  
  def classification_decision(  self
                              , priori      = None
                              , threshold   = None
                              , simulation_name     = None
                              , threshold_priori_id = None
                              ):
    """ 
    Dokonywanie klasyfikacji przez model w oparciu o zadane thresholdy (thresholdy tylko dla zmiennych binarnych), prawdopodobienstwa i wagi dla poszczegolnych kategorii zmiennej objasnianej (opcjonalne) 
    """
    
    # table with probabilities
    prob_f = copy.deepcopy(self.probabilities.loc[self.probabilities['simulation_name']==simulation_name])
    prob_f = prob_f.reset_index(drop=True)
    
    
    prob = prob_f[self.y_labels] # extracting columns with probabilities
    prob_f = prob_f.drop(self.y_labels, axis=1) # extracting columns with indicies
    
    # weighing probabilties
    if priori is not None:
      for p in self.y_labels :
        priori_p = priori[p]
        prob.loc[:,p] = prob.loc[:,p] * (priori_p)
      prob_sum = prob.apply(lambda x: 1/sum(x), axis = 1)
      for p in self.y_labels :
        prob.loc[:,p] = prob.loc[:,p]*prob_sum 
    
    # classification based on threshold (if threshold is provided and target is binary)
    if len(self.y_labels) == 2 and threshold is not None:
      var_1 = list(threshold.keys())[0]
      var_2 = list(threshold.keys())[1]
      threshold_var_1 = list(threshold.values())[0]
      threshold_var_2 = list(threshold.values())[1]
      classification  = pd.DataFrame([var_1 if x >= threshold_var_1 else var_2 for x in prob[str(var_1)] ])
    else: # classification by idmax
      classification =  pd.DataFrame(prob.idxmax(axis = 1)) # zapis jako DataFrame by dalej zrobić 'concat' ### ???
    
    classification   = classification.reset_index(drop=True)
    prob_f['y_pred'] = classification
    prob_f['threshold_priori_id'] = threshold_priori_id
    
    prob_f.insert(0, column = 'if_automatic',    value = 0)
    # sprawdzic 
    if len(self.classification) > 0:
      self.classification = pd.concat([self.classification.loc[(self.classification['if_automatic']!=0) | (self.classification['simulation_name']!=simulation_name) | (self.classification['threshold_priori_id']!=threshold_priori_id) ,:], prob_f], ignore_index=True)
    else:
      self.classification = prob_f
  
  def scores_determining(self, y_true, y_pred, pos_label = 1):
    
    """
    calculation scores for models predictions. Scores are not saved here. It is done in f:performance
    """
    
    # accuracy
    try:
        accuracy = skm.accuracy_score(y_true = y_true, y_pred = y_pred )
    except:
        accuracy = np.nan
    
    
    # balanced accuracy
    try:
        balanced_accuracy = skm.balanced_accuracy_score(y_true = y_true, y_pred = y_pred)
    except:
        balanced_accuracy = np.nan
    
    
    # recall
    try:
      if len(self.y_labels) < 3:
        recall  = skm.recall_score(y_true = y_true.astype(int), y_pred = y_pred.astype(int), average = 'binary', pos_label = int(pos_label))
      else:
        recall = skm.recall_score(y_true = y_true.astype(str), y_pred = y_pred.astype(str), average = 'weighted')
    except:
        recall = np.nan
        
    
    
    # precision
    try:
      if len(self.y_labels) < 3:
        precision  = skm.precision_score(y_true = y_true.astype(int), y_pred = y_pred.astype(int), average = 'binary', pos_label = int(pos_label))
      else:
        precision = skm.precision_score(y_true = y_true.astype(str), y_pred = y_pred.astype(str), average = 'weighted')
    except:
        precision = np.nan
    
    # f1
    try:
      if len(self.y_labels) < 3:
        f1 = skm.f1_score(y_true = y_true, y_pred = y_pred, average = 'binary', pos_label = str(pos_label))
      else:
        f1 = skm.f1_score(y_true = y_true, y_pred = y_pred, average = 'weighted')
    except:
      f1 = np.nan
    
    # putting all scores into Dataframe
    scores = pd.DataFrame({'score_name' :['accuracy','balanced_accuracy', 'recall', 'precision', 'f1'], 
                           'score_value':[ accuracy , balanced_accuracy ,  recall ,  precision ,  f1]})
    
    return(scores)
  
  
  
  def performance(  self
                  , simulation_name      = 'sim_1'
                  , priori               = None
                  , threshold            = None
                  , threshold_priori_id  = None
                  , pos_label            = '1'
                  , filter_test_indicies = None
                  , loops_progress       = False
                  ):
    
    """
    calculating models performance (scores and confusion matrices)
    Currently can't use self.y_label, self.x_var ,  self_x_num ect.
    """
    
    time_start = str(datetime.datetime.now().replace(microsecond=0))
    
    
    # classification decision (not automatic)
    self.classification_decision(priori = priori, threshold = threshold, simulation_name = simulation_name, threshold_priori_id = threshold_priori_id)
    
    # print('THRESHOLD')
    # print(threshold)
    
    # saving information about priori and threshold in a list
    if priori is not None:
        self.priori[simulation_name] = priori
    else:
        self.priori[simulation_name] = 'no defined'

    if threshold is not None:
        self.threshold[simulation_name] = threshold
    else:
        self.threshold[simulation_name] = 'no defined'
    
    # getting data with classification decision (brak automatic - w petli nie jest podany threshold_priori_id='' dla automatica)
    cls = self.classification.loc[(self.classification['simulation_name']==simulation_name) & (self.classification['threshold_priori_id']==threshold_priori_id), : ]
    
    
    # indicies filter for test set
    if filter_test_indicies is not None:
      cls_other = cls[cls['set_type']!='test']
      cls_test  = cls[cls['set_type']=='test'] 
      cls_test = cls_test[cls_test['index'].isin(filter_test_indicies)]
      
      cls = pd.concat([cls_other, cls_test], axis = 0)
    
    cls = cls.drop(columns='index') # nie potrzebujemy indeksu
    
    # data grouping with classification decision
    group_structure = ['simulation_name', 'model_name', 'set_type', 'sample_type', 'sample_nr', 'sample_name',  'if_automatic', 'threshold_priori_id']
    gr = cls.groupby(group_structure)
    
    # empty matrices to collect results
    scores = pd.DataFrame()
    confusion_matrix = pd.DataFrame()
    
    # loop over groups
    for name, group in gr:
        
        # printing progress of the loop
        if loops_progress: print(name)
        
        # scores 
        scores_i     = self.scores_determining( y_true = group['y_true'], y_pred = group['y_pred'], pos_label = pos_label )
        
        # scores group
        scores_group = pd.DataFrame([list(name)], columns = group_structure)
        
        # connecting scores and it's group
        scores_i     = pd.concat([scores_group, scores_i], axis = 1)
        
        # filling scores group
        scores_i[group_structure] = scores_i[group_structure].fillna(method = 'ffill')
        
        # adding full information about scores to collecting variable
        scores       = pd.concat([scores, scores_i])
        
        
        
        # confusion matrix
        try:
            cf_1 = pd.DataFrame([list(name)], columns = group_structure )
        except:
            cf_1 = pd.DataFrame()
        
        try:
            cf_2 = pd.DataFrame([skl.metrics.confusion_matrix(y_true = group['y_true'] , y_pred = group['y_pred']).flatten()])
        except:
            cf_2 = pd.DataFrame()
        
        # if confusion matrix are not empty save it
        if cf_1.shape[0] != 0 and cf_2.shape[0] != 0 :
            confusion_matrix_i = pd.concat([cf_1, cf_2], axis = 1)
            confusion_matrix = pd.concat([confusion_matrix, confusion_matrix_i], axis = 0 )
    
    # inforamtion about threshold and priori identifier
    # scores['threshold_priori_id'] = threshold_priori_id
    # confusion_matrix['threshold_priori_id'] = threshold_priori_id
    
    # saving scores
    if len(self.scores) != 0:
      self.scores = pd.concat([self.scores.loc[(self.scores['simulation_name'] != simulation_name) | (self.scores['threshold_priori_id'] != threshold_priori_id), : ], scores])
    else:
      self.scores = scores
    
    
    
    # saving confussion matrix
    if confusion_matrix.shape[0] != 0:
        if len(self.confusion_matrix) != 0:
          self.confusion_matrix = pd.concat([self.confusion_matrix.loc[(self.confusion_matrix['simulation_name'] != simulation_name)  | (self.confusion_matrix['threshold_priori_id'] != threshold_priori_id), : ], confusion_matrix])
        else:
          self.confusion_matrix = confusion_matrix
    
    time_end = str(datetime.datetime.now().replace(microsecond=0))
    
    # saving meta data
    self.simulation_metainfo.loc[self.simulation_metainfo['simulation_name']==simulation_name, 'pos_label']            = pos_label
    self.simulation_metainfo.loc[self.simulation_metainfo['simulation_name']==simulation_name, 'threshold_priori_id']  = threshold_priori_id
    self.simulation_metainfo.loc[self.simulation_metainfo['simulation_name']==simulation_name, 'performance start']    = time_start
    self.simulation_metainfo.loc[self.simulation_metainfo['simulation_name']==simulation_name, 'performance end']      = time_end
  
  
  def performance_multi_thr(self, thresholds_priori_list = None, simulation_name = None, pos_label = '1', filter_test_indicies = None):
    """
    thresholds_priori_list ={  'p1':[{'0':0.5, '1':0.5},{'0':0.2, '1':0.8}]
      , 'p2':[{'0':0.3, '1':0.7},{'0':0.2, '1':0.8}]}
    """
    
    gc.collect()
    
    
    for i, threshold_priori_id, in enumerate(thresholds_priori_list.keys()):
        threshold = thresholds_priori_list[threshold_priori_id][0]
        priori    = thresholds_priori_list[threshold_priori_id][1]
        # print(threshold)
        # print(priori)
        self.performance(simulation_name=simulation_name, threshold = threshold, priori = priori, pos_label = pos_label, threshold_priori_id = threshold_priori_id, filter_test_indicies = filter_test_indicies)
  
  
  def set_balancing(self, x_set, y_set, method='over'):
    """
    balancing data set
    """
    
    # checking if selected balancing method is available
    if method not in ['under', 'over']:
      raise ValueError('LUCAS: only permitted balancing methods are currently "over" and "under" ')
    
    
    # selecting balancing method
    if method == 'over':
      ros = RandomOverSampler(random_state=0)
    if method == 'under':
      ros = RandomUnderSampler(random_state=0)
    # if method == 'smote':
    #   ros = SMOTE(random_state=0)
    
    
    # creating balanced data set
    x_set_balanced, y_set_balanced = ros.fit_resample(x_set, y_set)
    x_set_balanced = pd.DataFrame(x_set_balanced, columns = list(x_set.columns) )
    y_set_balanced = pd.Series(y_set_balanced)
    
    # oryginalne indeksy (dla innych metod nie da sie odzyskac tej informacji)
    if method == 'under':
      new_idx = x_set.iloc[ros.sample_indices_].index # posortowanie po indeksach z under samplingu
      x_set_balanced.index = new_idx
      y_set_balanced.index = new_idx
    
    
    return([x_set_balanced, y_set_balanced])
  
  
  
  def cat_encoding( self
                  , x_train   = None
                  , y_train   = None
                  , x_valid   = None
                  , x_test    = None
                  , cat_vars = None
                  , cat_encoding_method = 'target'
                  , y_to_int = True):
    """
    supervised encoding of categorical variables
    # https://www.kaggle.com/discdiver/category-encoders-examples
    """
     # converting target variable into 'int' type (currently TargetEncoder does not support string target)
    if y_to_int:
      y_train = y_train.astype(float).astype('int64')
    if cat_encoding_method == 'target':
      encode = ce.TargetEncoder(cols = cat_vars )
    
    # fitting encoder
    encode.fit(x_train, y_train)
    
    # transforming train data set with encoded values
    data_encoded_train = encode.transform(X = x_train, y = y_train)
    
    # transforming valid and test data sets with encoded values
    if x_valid is not None:
      data_encoded_valid = encode.transform(X = x_valid, y = None)
    else: 
      data_encoded_valid = None
    if x_test is not None:
      data_encoded_test = encode.transform(X = x_test, y = None)
    else: 
      data_encoded_test = None # None a nie pusty DataFrame bo potem inne funkcje spradzaja warunek na istnienie x_test i y_test
    
    return([data_encoded_train, data_encoded_valid, data_encoded_test])
  
  
  def model_framework_identify(self, model):
      """
      Identifying which package does the model comes from
      """
      
      model_framework = str(type(model)) # checking only first model !!!
      
      if model_framework.find('sklearn') > 0:
        model_framework = 'sklearn'
      elif model_framework.find('catboost') > 0: 
        model_framework = 'catboost'
      elif model_framework.find('xgboost') > 0:
         model_framework = 'xgboost'
      elif models_framework.find('keras') > 0:
        model_framework = 'keras'
      else: 
        model_framework = 'not identified'
      
      return(model_framework)
  
  
  def feature_importance_class_CB(  self
                                  , iterations    = 10
                                  , learning_rate = 0.1
                                  , depth         = 5
                                  , cat_features  = None
                                  , x_train       = None
                                  , y_train       = None
                                  , x_test        = None
                                  , y_test        = None):
    """ 
    feature importance with CatBoost (based on test set if provided)
    """
    
    # getting indicies of categorical variables
    if cat_features is not None:
        x_train.loc[:,cat_features] = x_train.loc[:,cat_features].astype(str)
        #         x_train.loc[:,cat_features] = x_train.loc[:,cat_features].fillna('zzz')
        cat_indicies = [x_train.columns.get_loc(c) for c in cat_features if c in x_train]
    else:
        cat_indicies = None
    
    # creating and fitting the catboost model
    cb = CatBoostClassifier(  iterations = iterations, learning_rate=learning_rate, depth=depth)
    cb.fit(   x_train
            , y_train
            , plot                  = False
            , cat_features          = cat_indicies
            , use_best_model        = True
            , early_stopping_rounds = 75
            , silent                = True)
    
    # feature importance determining
    if y_test is None: # for test set
      return(np.round(cb.get_feature_importance(prettified=True), 3))
    else: # for train set
      return(np.round(cb.get_feature_importance(Pool(x_test, label=y_test, cat_features=cat_indicies), prettified=True), 3))
    
  
  def feature_importance_class_RF(  self
                                  , n_estimators
                                  , max_depth
                                  , x_train
                                  , y_train
                                  , x_test = None
                                  , y_test = None):
    """ 
      feature importance with RandomForest
    """
    
    # creating and fitting RandomForest model
    rf = RandomForestClassifier(  n_estimators = 50
                                 ,max_depth    = 10
                                 ,n_jobs       = 10
                                 ,oob_score    = True
                                 ,bootstrap    = True
                                 ,random_state = 42)
    rf.fit(x_train, y_train)
    
    # regular Feature Importatnce determining
    rf_importance = rf.feature_importances_
    rf_importance = sorted(zip(x_train.columns, rf.feature_importances_), key=lambda x: x[1] * -1)
    rf_importance = np.round(pd.DataFrame(rf_importance),2)
    rf_importance['method'] = 'Random_Forest_Importance'
    rf_importance.columns = ['Feature','Importance', 'method']
    
    def r2(rf, x_train, y_train):
      return(r2_score(y_train, rf.predict(x_train) ) )
    
    # Permutation Importance determining with use of test set (if provided)
    if y_test is None:
      rf_importance_permutation = pd.DataFrame(np.round(permutation_importances(rf, x_train, y_train, r2), 2) )
    else:
      rf_importance_permutation = pd.DataFrame(np.round(permutation_importances(rf, x_test,  y_test,  r2), 2) )
    rf_importance_permutation['method'] = 'RandomForest_permutation'
    rf_importance_permutation = rf_importance_permutation.reset_index(drop = False)
    
    
    return(pd.concat([rf_importance, rf_importance_permutation], axis = 0))
  
  
  
  def feature_importance_elastic_net(self, l1_ratio = [0.1, 0.5, 0.9], x_train=None, y_train=None):
      
      # data normalisation
      x_train = pd.DataFrame(preprocessing.StandardScaler().fit(x_train).transform(x_train), index = x_train.index, columns = list(x_train.columns))
      
      imp_coef = pd.DataFrame()
      
      for i in l1_ratio:
        elastic_i = SGDClassifier(loss = 'log', penalty = 'elasticnet', l1_ratio = i )
        elastic_i.fit(x_train, y_train)
        coef_i = pd.Series(elastic_i.coef_[0], index = list(x_train.columns))
        
        imp_coef_i = pd.DataFrame(coef_i)
        imp_coef_i.columns = ['Importance']
        imp_coef_i['Importance_abs'] = imp_coef_i['Importance'].abs()
        imp_coef_i['l1'] = str(i)
        imp_coef_i  = imp_coef_i.sort_values(['Importance_abs'], ascending=False)
        imp_coef = pd.concat([imp_coef, imp_coef_i])
        
      
      imp_coef = imp_coef.reset_index(drop=False)
      imp_coef = imp_coef.rename(columns = {'index':'Feature'})
      imp_coef = np.round(imp_coef, 2)
      
      return(imp_coef)
  
  
  def correlation_cramers_v(self, var_1, var_2, round = 2):
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
    n    = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k  = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1)) / (n-1))
    rcorr = r - ((r-1)**2) / (n-1)
    kcorr = k - ((k-1)**2) / (n-1)
    
    v_cramer = np.round( np.sqrt(phi2corr / min((kcorr-1), (rcorr-1))), round)
    
    return( v_cramer )
  
  
  def correlation_cramer_v_matrix(self, data, vars, round = 2):
    """
    A wrapper for 'correlation_cramers_v' function. Returns matrix of values for any given set of binary variables.
    """
    n = len(vars)
    ar = np.empty(shape=[n, n])
    
    for i in range(n):
      for j in range(n):
        if i > j:
          ar[i,j] = self.correlation_cramers_v(var_1 = data[vars[i]], var_2 = data[vars[j]], round = round)
        elif i == j:
          ar[i,j] = 1
        else:
          ar[i,j] = np.nan
    
    ar = pd.DataFrame(ar, columns = vars)
    ar.index = vars
    
    return( ar)
  
  
  def correlation_ratio(self, categories, measurements, round = 2):
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
    fcat, _     = pd.factorize(categories)
    cat_num     = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array     = np.zeros(cat_num)
    
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    
    y_total_avg = np.nansum(np.multiply(y_avg_array,n_array))/np.nansum(n_array)
    numerator   = np.nansum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.nansum(np.power(np.subtract(measurements,y_total_avg),2))
    
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)
    
    return(np.round(eta, round))
  
  
  def correlation_ratio_matrix(self, data, vars_cat, vars_num, round = 2):
    """ 
      A wrapper for 'correlation_ratio' function. Returns matrix of values for given set of variables.
    """
    
    ar = np.empty(shape=[len(vars_cat), len(vars_num)])
    
    for i in range(len(vars_cat)):
      for j in range(len(vars_num)):
        ar[i,j] = self.correlation_ratio(categories = data[vars_cat[i]], measurements = data[vars_num[j]])
    
    ar       = pd.DataFrame(ar, columns = vars_num)
    ar.index = vars_cat
    
    return(np.round(ar, round) )
  
  
  def correlation_pearson(self, data, vars = None, method = 'pearson', round = 2):
    """
    Pearson correlation matrix for numerical variables
    """
    if vars is None:
      return(np.round(self.data[self.num].corr(method = method), round))
    else:
      return(np.round(data[vars].corr(method = method), round))
  
  
  def simulation(  self
                  # data sets
                  ,train_set = None
                  ,test_set  = None
                  ,y_var     = None
                  ,x_var_num = None
                  ,x_var_cat = None
                  ,y_labels  = None
                  # categotical variables (for catboost list of cat vars is specified in 'models_fit_parameters' )
                  ,cat_encoding_method = 'target' # implemented methods : 'target'
                  # calibration and balancing
                  ,calibration_method =  None # 'isotonic'. Not recommended for catboost. If 'None' then no calibration
                  ,calibration_cv     = 3
                  ,balancing_method = None # 'over', 'under', 'smote'
                  # models and simulation
                  ,models_list     = None
                  ,models_fit_parameters = None # Warning: The same parameter is for all models. So you cannot use models with different f:fit syntax in one simulation !!! 
                  ,simulation_name = 'simulation_1'
                  ,description     = ' '
                  ,models_framework = None # 'sklearn', 'catboost', 'keras', 'xgboost'. We can set framework manually. If not function will try to identify it automatically
                  # CV parameters
                  ,cv_n = 5 # number of folds in CV. If 'None' no KFolding is performed
                  ,cv_save_prob  = True
                  ,cv_save_class = False
                  ,cv_save_confussion_matrix = False
                  # Holdout parameters
                  ,hold_n = None # number of holdoud sampling. If 'None' no sampling
                  ,hold_train_sample_size = 0.1
                  ,hold_test_sample_size  = 0.1 # fraction of observation drawn for each sample
                  ,hold_save_prob  = True
                  ,hold_save_class = False
                  ,hold_save_confussion_matrix = False
                  # Full parameters
                  ,full            = True
                  ,full_save_prob  = True
                  ,full_save_class = True
                  ,full_save_confussion_matrix = True
                  ,full_importance_n = [15,6,0.1] # [n_estimators, max_dept, learning_rate]
                  # by col
                  ,by_var = None # list with one column
                  ,by_var_save_prob = True
                  ,by_var_save_class = True
                  # Feature importance Permutation (FIP)
                  ,fip_n = None # number of tries in future importatnce
                  ,fip_sample_size = 0.5 # fraction of observation drawn for each sample
                  ,fip_type = 'remove' # remove, permutate_train
                  ,model_fip = RandomForestClassifier(n_estimators=30, max_depth=7) # for fip you can provide only one model
                  ,fip_save_prob  = True
                  ,fip_save_class = True
                  ,fip_save_confussion_matrix = True
                  # other options
                  ,data_check     = False
                  ,correlation    = False
                  ,loops_progress = False
                  ):
    """
    Main function of the object. It's basic task is to fit models using difference sampling techniques and then calculate probabilities on train, validation and test sets.
    """
    
    # PART 1: DATA BASIC PREPARATIONS

    
    x_var = x_var_num + x_var_cat
    
    x_var_num = x_var_num[:]
    x_var_cat = x_var_cat[:]
    
    # train_set    = train_set[x_var + y_var]
    # test_set     = test_set[x_var  + y_var]
    
    
    print(train_set.shape)
    print(test_set.shape)
    
    # extracting y_var if is list
    if type(y_var) == list:
      y_var = y_var[0]
    
    # checking if data structure is correct
    if data_check:
      self.data_check(data = train_set, set_name = 'train', y_var = y_var, x_var = x_var, x_var_cat = x_var_cat)
      if test_set is not None:
        self.data_check(data = test_set, set_name = 'train', y_var = y_var, x_var = x_var, x_var_cat = x_var_cat)
      else:
        print('LUCAS: Warning: test data set not provided. Calculations will be carried out without it')
    
    # if 'test' and 'train' have the same columns structure
    if test_set is not None:
      if any(train_set.columns != test_set.columns ):
        raise ValueError('LUCAS: x_train has different columns than x_test')
    
    # resetting indicies to be sure that duplicates are ramoved and converting target variable to str
    # train_set = train_set.sample(frac = 1)
    train_set[y_var] = train_set[y_var].astype(str) 
    
    if test_set is not None:
      # test_set  = test_set.sample(frac = 1)
      test_set[y_var]  = test_set[y_var].astype(str)
    
    
    # y counts
    y_train_count = pd.DataFrame(train_set[y_var].value_counts())
    y_train_count['proc'] = np.round(100 * y_train_count[y_var]/len(train_set), 2)
    y_train_count['simulation_name'] = simulation_name
    y_train_count = y_train_count.reset_index(drop = False)
    
    self.y_train_count = pd.concat([self.y_train_count, y_train_count], axis = 0)
    
    if test_set is not None:
      y_test_count = pd.DataFrame(test_set[y_var].value_counts())
      y_test_count['proc'] = np.round(100 * y_test_count[y_var]/len(test_set), 2)
      y_test_count['simulation_name'] = simulation_name
      y_test_count = y_test_count.reset_index(drop = False)
      self.y_test_count = pd.concat([self.y_test_count, y_test_count], axis = 0)
    
    
    
    # rebuilding sets if we have 'by_var'
    if by_var is not None:
      # separating by_var column
      train_set_by = pd.DataFrame( {by_var:train_set[by_var]} )
      test_set_by  = pd.DataFrame( {by_var:test_set[by_var]} )
      train_set    = train_set.drop(columns=[by_var])
      test_set     = test_set.drop(columns=[by_var])
      x_var.remove(by_var)
      x_var_cat.remove(by_var)
    
    # splitting data sets into target and features
    y_train = train_set[y_var]
    x_train = train_set[x_var]
    
    if test_set is not None:
      x_test = test_set[x_var]
      y_test = test_set[y_var]
    else:
      x_test = None
      y_test = None
      
    del train_set, test_set
    
    # type of classification problem
    self.classification_problem = skl.utils.multiclass.type_of_target(y_train)
    
    # checking if the simulation name exist in current list of simulations
    if self.simulation_metainfo.shape != (0,0):
      if simulation_name in list(self.simulation_metainfo.simulation_name):
        raise Exception('LUCAS: Simultion "' + simulation_name + '" already exists. Change name of your new simulation or remove existing simulation "' + simulation_name +'"')
    time_start = str(datetime.datetime.now().replace(microsecond=0))
    
    # determining categories for dependent variable
    if y_labels is None:
      y_labels        = list(y_train.drop_duplicates())
      y_labels.sort()
      self.y_labels   = y_labels
    else:
      self.y_labels = y_labels
    
    # names and positions of categorical variables (positions are used for exemple in catboost to give columns indicies), and names of numerical variables
    if x_var_cat is not None:
      self.cat_vars = x_var_cat
      self.cat_vars_indicies = [x_train.columns.get_loc(c) for c in x_var_cat if c in x_train]
      # num_vars = list(set(x_var) -  set(cat_vars))
      num_vars = x_var_num
      if len(num_vars)==0:
        self.num_vars = None
      else:
        self.num_vars = num_vars
    else:
      self.num_vars = x_var
      self.cat_vars = None
    
    # if cat encoding will be performed (not performed for catboost)
    use_cat_enc = len(x_var_cat) > 0 and cat_encoding_method is not None
    
    
    # setting variables for stacking
    
    
    # CV, HOLDOUT i FULL is implemented. All type of simulations can be carried out at once. 
    # bootstrap (currently not implemented) : # https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/modules/generated/sklearn.cross_validation.Bootstrap.html
    
    
    
    # saving raw data sets without any modifications
    x_train_b = copy.deepcopy(x_train)
    x_test_b  = copy.deepcopy(x_test)
    y_train_b = copy.deepcopy(y_train)
    y_test_b  = copy.deepcopy(y_test)
    
    
    # PART 2: CV - cross validation
    if cv_n is not None:
        
        sample_type = 'cv'
        
        # getting indicies for split od data set
        kf = skl.model_selection.KFold(n_splits = cv_n)
        kf.get_n_splits(x_train_b)
        
        #  loop over models list
        for j in range(len(models_list)):
          
            # printing information about current iteration of loop to see the calculations progress
            if loops_progress: print('cv ' + str(j))
            
            # model and it's features
            model_j_name = list(models_list.keys())[j]
            model_j = list(models_list.values())[j]
            
            if models_framework is None:
                model_framework = self.model_framework_identify(model_j)
            else:
                model_framework = models_framework
            
            # rewritting data with raw set
            x_train = copy.deepcopy(x_train_b)
            x_test  = copy.deepcopy(x_test_b)
            y_train = copy.deepcopy(y_train_b)
            y_test  = copy.deepcopy(y_test_b)
            
            # loop over K-Folds
            f = 0
            for train_index_i, valid_index_i in kf.split(x_train) :
                
                
                test_index_i = x_test.index
                
                if loops_progress: print(f)
                
                # building sets based on split indicies
                x_train_i, x_valid_i = x_train.iloc[train_index_i], x_train.iloc[valid_index_i]
                y_train_i, y_valid_i = y_train.iloc[train_index_i], y_train.iloc[valid_index_i]
                
                
                # information about simulation
                simulation_info = {'simulation_name':simulation_name, 'model_name':[model_j_name],'sample_type':[sample_type], 'sample_nr':[f]}
                
                
                # categorical variables encoding (optional)
                if  use_cat_enc and model_framework not in  ['catboost']:
                    x_train_i, x_valid_i, x_test = self.cat_encoding(x_train = x_train_i , y_train = y_train_i, x_valid = x_valid_i, x_test = x_test,  cat_vars = x_var_cat, cat_encoding_method = cat_encoding_method)
                
                # sample balancing (optional)
                if balancing_method is not None:
                    x_train_i, y_train_i = self.set_balancing(x_train_i, y_train_i, method=balancing_method)
                
                
                
                # model fitting
                if models_fit_parameters is not None:
                    model_j.fit(X=x_train_i, y=y_train_i, **models_fit_parameters )
                else:
                    if model_framework in ['sklearn']:
                        model_j.fit(X=x_train_i, y=y_train_i)
                    elif model_framework in ['catboost']:
                        model_j.fit(X = x_train_i, y = y_train_i,  plot = False, cat_features = self.cat_vars_indicies, use_best_model=False, early_stopping_rounds=75, logging_level='Silent')
                
                # probability calibration (optional)
                if calibration_method is not None and model_framework  in ['sklearn']:
                    model_j = CalibratedClassifierCV(model_j, cv=calibration_cv, method=calibration_method)
                    model_j.fit(x_train_i, y_train_i) # , sw_train
                
                # probabilities prediction calculation and saving
                
                
                if cv_save_prob:
                    prob_train = pd.DataFrame(model_j.predict_proba(x_train_i), columns = [str(x) for x in model_j.classes_], index = x_train_i.index)
                    # print(prob_train.shape)
                    self.probabilities_determining(y_true = y_train_i
                                                  , prob = prob_train, simulation_name = simulation_name, model_name = model_j_name, sample_type = sample_type, sample_nr = f, sample_name = '_', set_type = 'train')
                    
                    prob_valid = pd.DataFrame(model_j.predict_proba(x_valid_i), columns = [str(x) for x in model_j.classes_], index = x_valid_i.index)        
                    # print(prob_valid.shape)
                    self.probabilities_determining(y_true = y_valid_i
                                                  , prob = prob_valid, simulation_name = simulation_name, model_name = model_j_name, sample_type = sample_type, sample_nr = f, sample_name = '_', set_type = 'valid' )
                    if y_test is not None:
                        prob_test = pd.DataFrame(model_j.predict_proba(x_test), columns = [str(x) for x in model_j.classes_], index = x_test.index)  
                        # print(prob_test.shape)
                        self.probabilities_determining(y_true = y_test
                                                      , prob = prob_test, simulation_name = simulation_name, model_name = model_j_name, sample_type = sample_type, sample_nr = f, sample_name = '_', set_type = 'test')
                
                # classification calculation and saving (automatic based on default setting in 'proba' function)
                if cv_save_class:
                  
                    self.classification_decision_automatic( x = x_train_i, y_true = y_train_i
                                                          , model = model_j, simulation_name = simulation_name, model_name = model_j_name, sample_type = sample_type, sample_nr = f, sample_name = '_', set_type = 'train')
                                                          
                    self.classification_decision_automatic( x = x_valid_i, y_true = y_valid_i
                                                          , model = model_j, simulation_name = simulation_name, model_name = model_j_name, sample_type = sample_type, sample_nr = f, sample_name = '_', set_type = 'valid')
                    
                    if y_test is not None:
                        self.classification_decision_automatic( x = x_test, y_true = y_test
                                                              , model = model_j, simulation_name = simulation_name, model_name = model_j_name, sample_type = sample_type, sample_nr = f, sample_name = '_', set_type = 'test')
                f = f+1
    
    
    # PART 3: HOLD - holdout
    if hold_n is not None:
        
        sample_type = 'hold'
        
        # getting indicies for split od data set
        smp = skl.model_selection.ShuffleSplit(n_splits=hold_n, train_size=hold_train_sample_size, test_size = hold_test_sample_size,  random_state = None )
        smp.get_n_splits(x_train_b)
        
        train_index_all = {}
        valid_index_all = {}
        
        k = 0
        for train_index_i, valid_index_i in smp.split(x_train):
          
          train_index_all[k] = train_index_i
          valid_index_all[k] = valid_index_i
          k = k+1
        
        
        
        # loop over models
        for j in range(len(models_list)):
            
            # printing information about current iteration of loop to see the calculations progress
            if loops_progress: print('hold ' + str(j))
            
            # model and it's features
            model_j_name = list(models_list.keys())[j]
            model_j = list(models_list.values())[j]
            if models_framework is None:
                model_framework = self.model_framework_identify(model_j)
            else:
                model_framework = models_framework
            
            # rewritting data with raw set
            x_train = copy.deepcopy(x_train_b)
            x_test  = copy.deepcopy(x_test_b)
            y_train = copy.deepcopy(y_train_b)
            y_test  = copy.deepcopy(y_test_b)
            
            f = 0
            # loop over sampled sets
            for i in train_index_all.keys():
                
                # indicies for i-th loop
                train_index_i = train_index_all[i]
                valid_index_i = valid_index_all[i]
                test_index_i  = x_test.index
                
                # building sets based on split indicies
                x_train_i, x_valid_i = x_train.iloc[train_index_i], x_train.iloc[valid_index_i]
                y_train_i, y_valid_i = y_train.iloc[train_index_i], y_train.iloc[valid_index_i]
                
                # information about simulation
                simulation_info = {'simulation_name':simulation_name, 'model_name':[model_j_name],'sample_type':[sample_type], 'sample_nr':[f]}
                
                # categorical variables encoding
                if  use_cat_enc and model_framework not in  ['catboost']:
                    x_train_i, x_valid_i, x_test = self.cat_encoding(x_train = x_train_i , y_train = y_train_i, x_valid = x_valid_i, x_test = x_test_b,  cat_vars = x_var_cat, cat_encoding_method = cat_encoding_method)
                
                # sample balancing (optional)
                if balancing_method is not None:
                    x_train_i, y_train_i = self.set_balancing(x_train_i, y_train_i, method=balancing_method)
                
                # model fitting
                if models_fit_parameters is not None:
                    model_j.fit(X=x_train_i, y=y_train_i, **models_fit_parameters)
                else:
                    if model_framework in ['sklearn']:
                        model_j.fit(X=x_train_i, y=y_train_i)
                    elif model_framework in ['catboost']:
                        model_j.fit(X=x_train_i, y=y_train_i, plot=False, cat_features=self.cat_vars_indicies, use_best_model=True, early_stopping_rounds=75, logging_level='Silent'  )
                
                # probability calibration (optional)
                if calibration_method is not None and model_framework  in ['sklearn']:
                    model_j = CalibratedClassifierCV(model_j, cv=calibration_cv, method=calibration_method)
                    model_j.fit(x_train_i, y_train_i) # , sw_train
                
                # probabilities prediction calculation and saving
                if hold_save_prob:
                    
                    prob_train = pd.DataFrame(model_j.predict_proba(x_train_i), columns = [str(x) for x in model_j.classes_], index = x_train_i.index)
                    # print(prob_train.shape)
                    self.probabilities_determining(y_true = y_train_i
                                                  , prob = prob_train, simulation_name = simulation_name, model_name = model_j_name, sample_type = sample_type, sample_nr = f, sample_name = '_', set_type = 'train')
                    
                    prob_valid = pd.DataFrame(model_j.predict_proba(x_valid_i), columns = [str(x) for x in model_j.classes_], index = x_valid_i.index)        
                    # print(prob_valid.shape)
                    self.probabilities_determining(y_true = y_valid_i
                                                  , prob = prob_valid, simulation_name = simulation_name, model_name = model_j_name, sample_type = sample_type, sample_nr = f, sample_name = '_', set_type = 'valid' )
                    if y_test is not None:
                        prob_test = pd.DataFrame(model_j.predict_proba(x_test), columns = [str(x) for x in model_j.classes_], index = x_test.index)  
                        # print(prob_test.shape)
                        self.probabilities_determining(y_true = y_test
                                                      , prob = prob_test, simulation_name = simulation_name, model_name = model_j_name, sample_type = sample_type, sample_nr = f, sample_name = '_', set_type = 'test')
                
                # classification calculation and saving (automatic based on default setting in 'proba' function)
                if hold_save_class:
                  
                    self.classification_decision_automatic( x = x_train_i, y_true = y_train_i
                                                          , model = model_j, simulation_name = simulation_name, model_name = model_j_name, sample_type = sample_type, sample_nr = f, sample_name = '_', set_type = 'train')
                                                          
                    self.classification_decision_automatic( x = x_valid_i, y_true = y_valid_i
                                                          , model = model_j, simulation_name = simulation_name, model_name = model_j_name, sample_type = sample_type, sample_nr = f, sample_name = '_', set_type = 'valid')
                    
                    if y_test is not None:
                        self.classification_decision_automatic( x = x_test, y_true = y_test
                                                              , model = model_j, simulation_name = simulation_name, model_name = model_j_name, sample_type = sample_type, sample_nr = f, sample_name = '_', set_type = 'test')
                f = f+1
                
                
    # PART 4: BY_VAR - grouping by columns (tutaj x_test jest baza do robienie x_valid_i !!! sam x_test tez jest ale pod nazwa x_test_i bo jest problem ze zmienna by_var)
    if by_var is not None:
      
      sample_type = 'by_var'
      
      by_unique_train = train_set_by.drop_duplicates().sort_values(by_var).values[:,0].tolist()
      by_unique_test  = test_set_by.drop_duplicates().sort_values(by_var).values[:,0].tolist()
      
      if by_unique_train != by_unique_test:
        raise ValueError('LUCAS: categories for "by_var" in train and test set are different')
      else :
        by_unique = by_unique_train
      
      
      # loop over models
      for j in range(len(models_list)):
          
          # printing information about current iteration of loop to see the calculations progress
          if loops_progress: print('by_var ' + str(j))
          
          # model and it's features
          model_j_name = list(models_list.keys())[j]
          model_j = list(models_list.values())[j]
          if models_framework is None:
              model_framework = self.model_framework_identify(model_j)
          else:
              model_framework = models_framework
          
          # rewritting data with raw set
          x_train = copy.deepcopy(x_train_b)
          x_test  = copy.deepcopy(x_test_b)
          
          x_train = pd.concat([x_train, train_set_by], axis = 1)
          x_test  = pd.concat([x_test,  test_set_by], axis = 1)
          
          y_train = copy.deepcopy(y_train_b)
          y_test  = copy.deepcopy(y_test_b)
          
          
          # loop over sampled sets
          for f, i in enumerate(by_unique): 
            x_train_i = x_train.loc[x_train[by_var] == i ]
            x_valid_i = x_test.loc[x_test[by_var]   == i ] # valid jest czescia test-u
            y_train_i = pd.merge(y_train, x_train_i, left_index = True, right_index = True)[y_var]
            y_valid_i = pd.merge(y_test,  x_valid_i, left_index = True, right_index = True)[y_var]
            
            # indicies fot i-th loop
            train_index_i = x_train_i.index 
            valid_index_i = x_valid_i.index
            test_index_i  = x_test.index
            
            x_test_i = copy.deepcopy(x_test)
            y_test_i = copy.deepcopy(y_test)
            
            x_train_i = x_train_i.drop(columns = by_var)
            x_valid_i = x_valid_i.drop(columns = by_var)
            x_test_i = x_test_i.drop(columns = by_var)
            
            # information about simulation
            simulation_info = {'simulation_name':simulation_name, 'model_name':[model_j_name],'sample_type':[sample_type], 'sample_nr':[f], 'sample_name':[i]}
            
            # categorical variables encoding
            
            
            if  use_cat_enc and model_framework not in  ['catboost']:
                # UWAGA: x_test = x_test_i
                x_train_i,  x_valid_i, x_test_i = self.cat_encoding(  x_train = x_train_i , y_train = y_train_i, x_valid = x_valid_i, x_test = x_test_i
                                                                ,  cat_vars = x_var_cat, cat_encoding_method = cat_encoding_method)
            
            # sample balancing (optional)
            if balancing_method is not None:
                x_train_i, y_train_i = self.set_balancing(x_train_i, y_train_i, method=balancing_method)
            
            # model fitting
            if models_fit_parameters is not None:
                model_j.fit(X=x_train_i, y=y_train_i, **models_fit_parameters)
            else:
                if model_framework in ['sklearn']:
                    model_j.fit(X=x_train_i, y=y_train_i)
                elif model_framework in ['catboost']:
                    model_j.fit(X=x_train_i, y=y_train_i, plot=False, cat_features=self.cat_vars_indicies, use_best_model=True, early_stopping_rounds=75, logging_level='Silent'  )
            
            # probability calibration (optional)
            if calibration_method is not None and model_framework  in ['sklearn']:
                model_j = CalibratedClassifierCV(model_j, cv=calibration_cv, method=calibration_method)
                model_j.fit(x_train_i, y_train_i) # , sw_train
            
            # probabilities prediction calculation and saving
            if by_var_save_prob:
                
                prob_train = pd.DataFrame(model_j.predict_proba(x_train_i), columns = [str(x) for x in model_j.classes_], index = x_train_i.index)
                # print(prob_train.shape)
                self.probabilities_determining(y_true = y_train_i
                                              , prob = prob_train, simulation_name = simulation_name, model_name = model_j_name, sample_type = sample_type, sample_nr = f, sample_name = str(i), set_type = 'train')
                
                prob_valid = pd.DataFrame(model_j.predict_proba(x_valid_i), columns = [str(x) for x in model_j.classes_], index = x_valid_i.index)
                # print(prob_valid.shape)
                self.probabilities_determining(y_true = y_valid_i
                                              , prob = prob_valid, simulation_name = simulation_name, model_name = model_j_name, sample_type = sample_type, sample_nr = f, sample_name = str(i), set_type = 'valid' )
                
                if y_test is not None:
                    prob_test = pd.DataFrame(model_j.predict_proba(x_test_i), columns = [str(x) for x in model_j.classes_], index = x_test_i.index)
                    # print(prob_test.shape)
                    self.probabilities_determining(y_true = y_test
                                                  , prob = prob_test, simulation_name = simulation_name, model_name = model_j_name, sample_type = sample_type, sample_nr = f, sample_name = str(i), set_type = 'test')
            
            # classification calculation and saving (automatic based on default setting in 'proba' function)
            if by_var_save_class:
              
                self.classification_decision_automatic( x = x_train_i, y_true = y_train_i
                                                      , model = model_j, simulation_name = simulation_name, model_name = model_j_name, sample_type = sample_type, sample_nr = f, sample_name = str(i), set_type = 'train')
                                                      
                self.classification_decision_automatic( x = x_valid_i, y_true = y_valid_i
                                                      , model = model_j, simulation_name = simulation_name, model_name = model_j_name, sample_type = sample_type, sample_nr = f, sample_name = str(i), set_type = 'valid')
                
                if y_test is not None:
                    self.classification_decision_automatic( x = x_test_i, y_true = y_test
                                                          , model = model_j, simulation_name = simulation_name, model_name = model_j_name, sample_type = sample_type, sample_nr = f, sample_name = str(i), set_type = 'test')
            
            
    
    
    
    # PART 5: FULL - fitting on full train set without sampling and creating validation sets.
    if full is not None:
      
        sample_type = 'full'
        
        # loop over models
        f = 0  # there is only one iteration so 'f' is fixed
        for j in range(len(models_list)):
            # printing information about current iteration of loop to see the calculations progress
            if loops_progress: print('full ' + str(j))
            
            # model and it's features
            model_j_name = list(models_list.keys())[j]
            model_j = list(models_list.values())[j]
            if models_framework is None:
                model_framework = self.model_framework_identify(model_j)
            else:
                model_framework = models_framework
            simulation_info  = {'simulation_name':simulation_name, 'model_name':[model_j_name],'sample_type':[sample_type], 'sample_nr':[f]} # sample_nr='0' na sztywno bo dla FULL jest tylko jedna iteracja
            
            # rewritting data with raw set
            x_train = copy.deepcopy(x_train_b)
            x_test  = copy.deepcopy(x_test_b)
            y_train = copy.deepcopy(y_train_b)
            y_test  = copy.deepcopy(y_test_b)
            
            # indicies
            train_index_i = x_train.index
            valid_index_i = None
            test_index_i  = x_test.index
            
            # cat encoding (optional)
            if  use_cat_enc and model_framework not in  ['catboost']:
                x_train, x_test, x_test = self.cat_encoding(   x_train = x_train , y_train = y_train, x_valid = None, x_test=x_test
                                                            ,  cat_vars = x_var_cat, cat_encoding_method = cat_encoding_method)
            
            # sample balancing (optional)
            if balancing_method is not None:
                x_train, y_train = self.set_balancing(x_train, y_train, method=balancing_method)
            
            # model fitting
            if models_fit_parameters is not None:
                model_j.fit(X=x_train, y=y_train, **models_fit_parameters )
            else:
                if model_framework in ['sklearn']:
                    model_j.fit(X=x_train, y=y_train)
                elif model_framework in ['catboost']:
                    model_j.fit(X=x_train, y=y_train, plot=False, cat_features=self.cat_vars_indicies, use_best_model=True, early_stopping_rounds=75, logging_level='Silent' )
            
            # probability calibration (optional)
            if calibration_method is not None and model_framework in ['sklearn']:
                model_j = CalibratedClassifierCV(model_j, cv=calibration_cv, method=calibration_method)
                model_j.fit(x_train, y_train) # , sw_train
            
            # probabilities prediction calculation and saving
            if full_save_prob:
              
                    prob_train = pd.DataFrame(model_j.predict_proba(x_train), columns = [str(x) for x in model_j.classes_], index = x_train.index)
                    self.probabilities_determining(y_true = y_train
                                                  , prob = prob_train, simulation_name = simulation_name, model_name = model_j_name, sample_type = sample_type, sample_nr = f, sample_name = '_', set_type = 'train')
                    
                    if y_test is not None:
                        prob_test = pd.DataFrame(model_j.predict_proba(x_test), columns = [str(x) for x in model_j.classes_], index = x_test.index)  
                        self.probabilities_determining(y_true = y_test
                                                      , prob = prob_test, simulation_name = simulation_name, model_name = model_j_name, sample_type = sample_type, sample_nr = f, sample_name = '_', set_type = 'test')
            
            # classification calculation and saving (automatic based on default setting in 'proba' function)
            if full_save_class:
                
                self.classification_decision_automatic( x = x_train, y_true = y_train
                                                      , model = model_j, simulation_name = simulation_name, model_name = model_j_name, sample_type = sample_type, sample_nr = f, sample_name = '_', set_type = 'train')
                
                if y_test is not None:
                    self.classification_decision_automatic( x = x_test, y_true = y_test
                                                          , model = model_j, simulation_name = simulation_name, model_name = model_j_name, sample_type = sample_type, sample_nr = f, sample_name = '_', set_type = 'test')
        
        # feature importance calculations
        if full_importance_n is not None:
          
          if loops_progress: print('feature_importance')
          
          # rewritting data with raw set
          x_train = copy.deepcopy(x_train_b)
          x_test  = copy.deepcopy(x_test_b)
          y_train = copy.deepcopy(y_train_b)
          y_test  = copy.deepcopy(y_test_b)
          
          # adding noise variables
          x_train['NOISE_uniform'] = np.random.uniform(-10, 10, len(x_train))
          x_train['NOISE_normal']  = np.random.normal(0, 10, len(x_train))
          
          if x_test is not None:
            x_test['NOISE_uniform'] = np.random.uniform(-10, 10, len(x_test))
            x_test['NOISE_normal']  = np.random.normal(0, 10, len(x_test))
          
          # sample balancing (optional)
          if balancing_method is not None:
              x_train, y_train = self.set_balancing(x_train, y_train, method=balancing_method)
          
          # CATBOOST (cat boost currently does not suppor multi-classification)
          if len(self.y_labels) == 2:
            
            feature_importance_cat = self.feature_importance_class_CB(  iterations = full_importance_n[0], learning_rate = full_importance_n[2],  depth = full_importance_n[1], cat_features = x_var_cat
                                                                      , x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test)
                                                        
            feature_importance_cat = feature_importance_cat.rename(columns={'Importances': 'Importance', 'Feature Id':'Feature'}) # for structer compatibility with results from randomforest feature importatnce
            feature_importance_cat['method']='catboost'
          else:
            feature_importance_cat = pd.DataFrame()
          
          
          # RANDOM FOREST
          
          # cat encoding (optional)
          if use_cat_enc:
            x_train, x_test, x_test = self.cat_encoding(x_train = x_train , y_train = y_train, x_valid = None, x_test=x_test,  cat_vars = x_var_cat, cat_encoding_method = cat_encoding_method)
          
          feature_importance_rt = self.feature_importance_class_RF( n_estimators = full_importance_n[0], max_depth = full_importance_n[1]
                                                                  , x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test)
          feature_importance_rt = feature_importance_rt.reset_index(drop = False)
          
          
          # Elastic net (only for binary variables - cat_encoding required so I get from random forest above)  
          if len(self.y_labels) == 2:
            feature_importance_enet = self.feature_importance_elastic_net(l1_ratio = [0.1, 0.5, 0.9], x_train = x_train, y_train = y_train)
            feature_importance_enet['method'] = 'elastic_net'
          else:
            feature_importance_enet = pd.DataFrame()
          
          
          # gathering all results
          feature_importance = pd.concat([feature_importance_rt, feature_importance_cat], axis = 0)
          feature_importance['simulation_name']      = simulation_name
          if len(feature_importance_enet) > 0:
            feature_importance_enet['simulation_name'] = simulation_name
          
          # columns reordering
          feature_importance = feature_importance[['simulation_name','method','Feature','Importance']]
          
          # saving results
          self.feature_importance = pd.concat([self.feature_importance, feature_importance])
          if len(feature_importance_enet) > 0 :
            self.feature_importance_enet = pd.concat([self.feature_importance_enet, feature_importance_enet])
    
    # PART 6: FIP- feature importance permutation
    if fip_n is not None:
        
        sample_type = 'fip'
        
        # loop over samples # not implemented
        for f in range(fip_n):
            if loops_progress: print('fip '+ str(f))
            
            # rewritting data with raw set
            x_train = copy.deepcopy(x_train_b)
            x_test  = copy.deepcopy(x_test_b)
            y_train = copy.deepcopy(y_train_b)
            y_test  = copy.deepcopy(y_test_b)
            
            # split zbioru na testowy i uczący
            x_train_fip, x_valid_fip, y_train_fip, y_valid_fip = skl.model_selection.train_test_split(x_train, y_train, test_size=fip_sample_size)
            
            
            # indicies
            train_index_i = x_train_fip.index 
            valid_index_i = x_valid_fip.index
            test_index_i  = x_test.index
            
            
            # loop over feature variables  (instead of models loop !!!)
            for column in x_train.columns:
                # model and it's features
                model_fip_name = column
                model_framework = self.model_framework_identify(model_fip)
                
                # simulation information
                simulation_info = {'simulation_name':simulation_name, 'model_name':[model_fip_name],'sample_type':[sample_type], 'sample_nr':[f]}
                

                # sample balancing (optional)
                if balancing_method is not None:
                  x_train_fip, y_train_fip = self.set_balancing(x_train_fip, y_train_fip, method=balancing_method)
                if fip_type == 'permutate_train':
                    # variable permutation
                    x_train_fip_perm = copy.deepcopy(x_train_fip)
                    x_valid_fip_perm = copy.deepcopy(x_valid_fip)
                    x_train_fip_perm[column] =  np.random.permutation(x_train_fip_perm[column])
                    # x_valid_fip_perm[column] =  np.random.permutation(x_valid_fip_perm[column])
                    if y_test is not None:
                         x_test_perm         = copy.deepcopy(x_test)
                    #     x_test_perm[column] = np.random.permutation(x_test_perm[column])
                    # przeliczenie na nowo współrzędnych zmiennych jakościowych (np. dla catboost-a) - akurat tutaj zostaje po staremu
                    cat_vars_indicies = self.cat_vars_indicies
                elif fip_type == 'remove':
                    x_train_fip_perm = copy.deepcopy(x_train_fip.drop(columns=column))
                    x_valid_fip_perm = copy.deepcopy(x_valid_fip.drop(columns=column))
                    if y_test is not None:
                        x_test_perm = copy.deepcopy(x_test.drop(columns=column))
                    # przeliczenie na nowo współrzędnych zmiennych jakościowych (np. dla catboost-a)
                    x_var_cat = self.cat_vars
                    x_var_cat = list(set(x_var_cat) - set([column]))
                    cat_vars_indicies = [x_train_fip_perm.columns.get_loc(c) for c in x_var_cat if c in x_train_fip_perm]

                # categorical variables encoding (optional)
                if  use_cat_enc  and model_framework not in  ['catboost']:
                    x_train_fip_perm, x_valid_fip_perm, x_test_perm = self.cat_encoding(x_train = x_train_fip_perm , y_train = y_train_fip, x_valid = x_valid_fip_perm, x_test = x_test_perm,  cat_vars = x_var_cat, cat_encoding_method = cat_encoding_method)


                # model fitting
                if models_fit_parameters is not None:
                    model_fip.fit(X=x_train_fip_perm, y=y_train_fip, **models_fit_parameters )
                else:
                    if model_framework in ['sklearn']:
                        model_fip.fit(X=x_train_fip_perm, y=y_train_fip)
                    elif model_framework in ['catboost']:
                        model_fip.fit(X=x_train_fip_perm, y=y_train_fip, plot=False, cat_features=self.cat_vars_indicies, use_best_model=True, early_stopping_rounds=75, logging_level='Silent'  )
                
                # probability calibration (optional)
                if calibration_method is not None and models_framework in ['sklearn']:
                    model_fip = CalibratedClassifierCV(model_fip, cv=calibration_cv, method=calibration_method)
                    model_fip.fit(x_train_fip_perm, y_train_fip) # , sw_train
                
                # probabilities prediction calculation and saving
                if fip_save_prob:
                    
                    # UWAGA: aktualnie permutuje tylko x_train
                    # UWAGA przy przeklejaniu : tutaj zbiory maja w nazwie 'fip'
                    
                    
                    
                    prob_train = pd.DataFrame(model_fip.predict_proba(x_train_fip_perm), columns = [str(x) for x in model_fip.classes_], index = x_train_fip_perm.index)
                    self.probabilities_determining(y_true = y_train_fip
                                                  , prob = prob_train, simulation_name = simulation_name, model_name = model_fip_name, sample_type = sample_type, sample_nr = f, sample_name = column , set_type = 'train')
                    
                    prob_valid = pd.DataFrame(model_fip.predict_proba(x_valid_fip_perm), columns = [str(x) for x in model_fip.classes_], index = x_valid_fip.index)
                    self.probabilities_determining(y_true = y_valid_fip
                                                  , prob = prob_valid, simulation_name = simulation_name, model_name = model_fip_name, sample_type = sample_type, sample_nr = f, sample_name = column, set_type = 'valid' )
                    
                    if y_test is not None:
                        prob_test = pd.DataFrame(model_fip.predict_proba(x_test_perm), columns = [str(x) for x in model_fip.classes_], index=x_test_perm.index)
                        self.probabilities_determining(y_true = y_test
                                                      , prob = prob_test, simulation_name = simulation_name, model_name = model_fip_name, sample_type = sample_type, sample_nr = f, sample_name = column, set_type = 'test')
                
                # classification calculation and saving (automatic based on default setting in 'proba' function)
                if fip_save_class:
                    
                    self.classification_decision_automatic( x = x_train_fip_perm, y_true = y_train_fip
                                                          , model = model_fip, simulation_name = simulation_name, model_name = model_fip_name, sample_type = sample_type, sample_nr = f, sample_name = column, set_type = 'train')
                    
                    self.classification_decision_automatic(  x = x_valid_fip_perm, y_true = y_valid_fip
                                                           , model = model_fip, simulation_name = simulation_name, model_name = model_fip_name, sample_type = sample_type, sample_nr = f, sample_name = column, set_type = 'valid')
                    if y_test is not None:
                        self.classification_decision_automatic(  x = x_test_perm, y_true = y_test
                                                               , model = model_fip, simulation_name = simulation_name, model_name = model_fip_name, sample_type = sample_type, sample_nr = f, sample_name = column, set_type = 'test')
    
    
    
    # PART 7: CORRELATIONS determining (target variable not included)

    if correlation:

      x_train_correlation = copy.deepcopy(x_train_b)
      y_train_correlation = copy.deepcopy(y_train_b)


      # zmiennej bez 'self' uzywam opcjonalnie jezeli nie chce sie odwolywac do globalnych ustawien
      num_vars = self.num_vars
      cat_vars = self.cat_vars

      if use_cat_enc and len(self.cat_vars) > 0:
          encode = ce.TargetEncoder(cols=self.cat_vars)
          encode.fit(X=x_train_correlation[self.cat_vars], y=y_train_correlation.astype(float).astype('int64'))  #
          x_train_correlation[self.cat_vars] = encode.transform(X=x_train_correlation[self.cat_vars], y=y_train_correlation.astype(float).astype('int64'))
          num_vars = num_vars + cat_vars


      # VIF
      if len(num_vars) > 0:

          vif = pd.DataFrame()
          vif['feature'] = num_vars
          vif['VIF'] = [variance_inflation_factor(x_train_correlation[num_vars].values, i) for i in range(len(num_vars))]
          self.vif[simulation_name] = vif

      # correlation for numeric variables
      if self.num_vars is not None:
        
        correlation_numeric_pearson = self.correlation_pearson(data = x_train_correlation, vars = num_vars, method = 'pearson', round = 2)
        correlation_numeric_kendall = self.correlation_pearson(data = x_train_correlation, vars = num_vars, method = 'kendall', round = 2)
        
        self.correlation_numeric_pearson[simulation_name]  = correlation_numeric_pearson
        self.correlation_numeric_kendall[simulation_name]  = correlation_numeric_kendall
      
      # correlation for categorical variables
      if self.cat_vars is not None:
        
        correlation_categorical_v_cramer = self.correlation_cramer_v_matrix(data=x_train_correlation, vars= cat_vars,  round = 2)
        
        self.correlation_categorical_v_cramer[simulation_name] = correlation_categorical_v_cramer
      
      # correlation for mix of categorical and numerical variables
      if self.cat_vars is not None and self.num_vars is not None:
        
        correlation_mix_ratio = self.correlation_ratio_matrix(data = x_train_correlation.reset_index(drop=True), vars_cat = cat_vars, vars_num = self.num_vars, round = 2) # tutaj zostawiam self.num_vars pierwotny
        
        self.correlation_mix_ratio[simulation_name] = correlation_mix_ratio
    
    
    # PART 8: METADATA about simulation
    time_end = str(datetime.datetime.now().replace(microsecond=0))
    num_vars_str = 'None' if self.num_vars is None else ', '.join(self.num_vars) # names of numeric features
    cat_vars_str = 'None' if self.cat_vars is None else ', '.join(self.cat_vars) # names of categorical features
    x_var_n = len(x_var) # number of feature variables
    simulation_metainfo_new      = pd.DataFrame({   'simulation_name': [simulation_name] 
                                                  , 'description'    : [description]
                                                  , 'x_n'      : x_var_n
                                                  , 'y_name'   : y_var
                                                  , 'y_labels' : [', '.join(self.y_labels)]
                                                  , 'x_num'    : [num_vars_str]
                                                  , 'x_cat'    : [cat_vars_str]
                                                  , 'cat_encoding'    : cat_encoding_method
                                                  , 'balancing'       : balancing_method
                                                  , 'calibration'     : calibration_method
                                                  , 'simulation_start': [time_start]
                                                  , 'simulation_end'  : [time_end],})
    # empty slots for performance metadata
    simulation_metainfo_new['pos_label']  = np.nan
    simulation_metainfo_new['priori']     = np.nan
    simulation_metainfo_new['threshold']  = np.nan
    simulation_metainfo_new['performance start'] = np.nan
    simulation_metainfo_new['performance end']   = np.nan
    
    # final save of metadata
    self.simulation_metainfo     = pd.concat([self.simulation_metainfo, simulation_metainfo_new])
    self.models[simulation_name] = models_list
  
  
  
  
  def plot_probabilities_density(   self
                                  , data_new = None
                                  , x_var  = ['1']
                                  , filter = None
                                  , fill_var ='y_true'
                                  , facet  = None
                                  , x_lim  = [np.nan, np.nan]
                                  , adjust = 0.5
                                  , alpha  = 0.5
                                  , size   = 2
                                  , frac   = 1
                                  , title  = ''
                                  , fig_w  = 15
                                  , fig_h  = 5):
    """
    plotting density plot of probabilities
    """
    # plot size
    plotnine.options.figure_size = (fig_w, fig_h)
    
    # choosing data set for plot
    if data_new is not None:
      data = data_new
    else:
      data = self.probabilities
    
    # filtering data
    if filter is not None:
      for i in filter.keys():
        data = data.loc[ data[i].isin(filter[i]) ,:]
    
    # variable used for 'fill 'converted into string (plotnine requirement)
    if fill_var != np.nan:
      data[fill_var] = data[fill_var].astype(str)
    
    # faceting of plot
    if facet is None:
      facet = facet_null()
    else:
      facet = facet_grid(facets=facet, scales='free_y')
    
    # sampling data (plotting big data set can be impossible) 
    if len(data) < 50000 and len(data) > 1:
      data = data.sample(frac=frac)
    else:
      data = data.sample(n = 50000)
    
    if len(data)== 0:
      print('LUCAS: after filtering no data to display. Suggestion is to check "set_type" if exists')
    else:
      return(ggplot(data = data ) + 
             geom_density(aes(x = x_var, colour = fill_var, fill = fill_var), adjust = adjust, size = size, alpha = alpha) + 
             facet +
             ggtitle(title))
  
  
  
  def plot_calibration_curve(   self
                              , simulation_name = None
                              , model_name      = None
                              , sample_type     = 'full'
                              , set_type        = 'test' 
                              , sample_nr       = 0
                              , pos_label       = '1'):
    
    """
    calibration plot. Only for binary classifiers
    """
    
    # checking if target is binary
    if len(self.y_labels) > 2:
      raise ValueError('LUCAS: calibration plot currently is only for binary classifiers. The target variable has more than two labels')
    
    pos_label = str(pos_label)
    
    # getting probabilities data
    data = self.probabilities.loc[(self.probabilities['simulation_name']== simulation_name) & 
                                  (self.probabilities['model_name']     == model_name)      & 
                                  (self.probabilities['sample_type']    == sample_type)     & 
                                  (self.probabilities['set_type']       == set_type)        & 
                                  (self.probabilities['sample_nr']      == sample_nr), ['y_true', pos_label] ]
    
    # determining calibration curve
    curve = skl.calibration.calibration_curve(y_true = data['y_true'], y_prob=data[pos_label], normalize=False, n_bins=10, strategy='uniform')
    curve = pd.DataFrame(curve).T
    curve.columns = [str(x) for x in list(curve.columns)]
    
    return(ggplot(data=curve) + geom_line(aes(x = '0', y = '1'), color = 'red') + xlim([0,1]) + ylim([0,1]) )
    
    
    
  
  def plot_scores_bar_full_fip( self
                              , score
                              , simulation_name
                              , set_type = 'test'):
        """
        Bar plot with scores for 'full' and 'fip' sample types
        """
        
        scores = self.scores
        scores['score_value'] = np.round(scores['score_value'], 3)
        scores.sort_values(by=['score_value'])
        
        if len(scores) > 0:
          return( ggplot(data = scores.loc[(scores['simulation_name'] == simulation_name) & (scores['set_type']==set_type) & (scores['if_automatic']==0) & ((scores['sample_type']=='full') | (scores['sample_type']=='fip')) & (scores['score_name']==score),:]) + 
                  geom_bar(aes(x='model_name',y='score_value', fill = 'sample_nr'), stat='identity') + 
                  geom_label(aes(x='model_name',y='score_value', label='score_value', fill = 'sample_nr')) + 
                  ylim([0,1]) + 
                  coord_flip())


  def scores_volatility(self, simulation_name, sample_type, set_type):
      scores = self.scores

      scores_filtered = scores.loc[(scores['simulation_name'] == simulation_name) & (scores['if_automatic'] == 0)  & (scores['sample_type']==sample_type) & (scores['set_type'].isin(set_type))]

      scores_group = scores_filtered.groupby(
          ['model_name', 'set_type', 'sample_type', 'threshold_priori_id', 'score_name']).agg(
          max=pd.NamedAgg(column='score_value', aggfunc='max'),
          min=pd.NamedAgg(column='score_value', aggfunc=lambda x: min(x)),
          mean=pd.NamedAgg(column='score_value', aggfunc=lambda x: np.nanmean(x)),
          std=pd.NamedAgg(column='score_value', aggfunc=lambda x: np.nanstd(x))
      )

      scores_group = scores_group.reset_index()
      return(scores_group)

  
  def plot_scores( self
                , data_new = None
                , filter = {'if_automatic':[0], 'sample_type':['cv'], 'set_type' : ['test']}
                , facet = 'score_name~simulation_name'
                , fill_var = 'model_name'
                , x_var = 'sample_nr'
                , y_var = 'score_value'
                , fig_w = 15
                , fig_h = 5):
    """
    Line plot with scores
    """
    
    # plot size
    plotnine.options.figure_size = (fig_w, fig_h)
    
    # choosing data set for plot
    if data_new is not None:
      data = data_new
    else:
      data = self.scores
    
    # filtering data
    if filter is not None:
      for i in filter.keys():
        data = data.loc[ data[i].isin(filter[i]) ,:]
    
    # converting into string variable for fill (plotnine requrement)
    if fill_var != np.nan:
      data[fill_var] = data[fill_var].astype(str)
    
    # faceting of plot
    if facet is None:
      facet = facet_null()
    else:
      facet = facet_grid(facets=facet)
    
    if len(data)== 0:
      # raise ValueError('LUCAS: after filtering no data to display. Suggestion is to check "set_type" if exists')
      print('LUCAS: after filtering no data to display. Suggestion is to check "set_type" if exists')
    else:
      return((ggplot(data=data ) + 
              geom_point( aes(x = x_var, y = y_var,  color = fill_var)) + 
              geom_line(  aes(x = x_var, y = y_var,  color = fill_var)) + 
              facet ) + theme(strip_text_y = element_text(angle = 0,              # change facet text angle
                                        ha = 'left'             # change text alignment
                                       ),  strip_background_y = element_text(color = '#969dff' # change background colour of facet background
                                              , width = 0.2     # adjust width of facet background to fit facet text
                                             ) ))
  
  
  
  def plot_roc_pr_curves( self
                        , simulation_name
                        , model_name  = None
                        , sample_type = ['full']
                        , set_type    = 'test'
                        , sample_nr   = None
                        , y_category  = None
                        , group = 'variable'
                        , facet = 'model_name~sample_nr'):
    
    """
    Step plot of "roc-curve" and "precision-recall-curve".
    """
    # getting probabilities data
    dt = self.probabilities.loc[(self.probabilities['simulation_name']==simulation_name) & (self.probabilities['sample_type'].isin(sample_type))  & (self.probabilities['set_type']==set_type) , :] # if automatic not filtered because it is not included in 'probabilities'

    dt['sample_nr'] = dt['sample_nr'].astype(str)  # 'str' for for gruping in ggplot

    # filtering model_name 
    if model_name is not None:
        if type(model_name) != list:
            model_name = [model_name]
        dt = dt.loc[dt['model_name'].isin(model_name),:]
    
    # filtering samples_nr
    if sample_nr is not None:
        if type(sample_nr) != list:
            sample_nr = [sample_nr]
        dt = dt.loc[dt['sample_nr'].isin(sample_nr),:]
    
    if len(dt) == 0:
        raise ValueError('LUCAS: after applying all filters now rows left in data. Check if you provided combination of existing values for parameters.')
    
    sample_type_list = sample_type
    model_name_list = list(dt['model_name'].drop_duplicates())
    sample_nr_list  = list(dt['sample_nr'].drop_duplicates())
    
    if y_category is None:
        y_category_list = self.y_labels
    else:
        if type(y_category) != list:
            y_category = [y_category]
        y_category_list = y_category
    
    
    # empty data frames to gather results
    pr  = pd.DataFrame()
    roc = pd.DataFrame()
    
    roc_auc = dict()
    average_precision = dict()
    
    # loop over model_name
    for m in model_name_list:
        print(m)
        # loop over sample_nr
        sample_nr_list = list(dt.loc[dt['model_name'] == m, 'sample_nr'].drop_duplicates())
        for s in sample_nr_list:
            
            precision = dict()
            recall = dict()
            fpr = dict()
            tpr = dict()
            
            # loop over y_category
            for i in y_category_list:
                
                # filtering rows for i-th iteration
                dt_i = dt.loc[(dt['model_name']==m) & (dt['sample_nr']==s),:]
                
                # filtering two columns for i-th iteration ( 'y_true' and probabilities for current category)
                dt_i = dt_i[['y_true'] + [i]]
                
                
                # adding dummy variable for current category of target (current label (1) vs other labels (0) )
                dt_i_a = dt_i.loc[dt_i['y_true']==i,:]
                dt_i_b = dt_i.loc[dt_i['y_true']!=i,:]
                
                dt_i_a['y_true'] = 1
                dt_i_b['y_true'] = 0
                
                dt_i = pd.concat([dt_i_a, dt_i_b])
                
                
                # precision curve
                precision[i], recall[i], _ = precision_recall_curve(dt_i['y_true'] , dt_i[i])
                
                # average precision
                average_precision[str(m) + ' - ' + str(s) + ' - '  +  str(i)] = average_precision_score(dt_i['y_true'], dt_i[i])
                
                # FPR and TPR for ROC curve
                fpr[i], tpr[i], _ = roc_curve(dt_i['y_true'] , dt_i[i])
                
                # AUC ROC
                roc_auc[str(m) + ' - ' + str(s) + ' - ' +  str(i)] = auc(fpr[i], tpr[i])
            
            # collecting result
            precision = pd.DataFrame.from_dict(precision,orient = 'index').T
            precision = precision.melt(value_name='precision')
            recall    = pd.DataFrame.from_dict(recall,orient = 'index').T
            recall    = recall.melt(value_name = 'recall')
            
            pr_i = pd.concat([precision, recall['recall']], axis = 1)
            pr_i['model_name'] = m
            pr_i['sample_nr']  = s
            pr = pd.concat([pr, pr_i])
            
            fpr = pd.DataFrame.from_dict(fpr,orient = 'index').T
            fpr = fpr.melt(value_name = 'fpr')
            tpr = pd.DataFrame.from_dict(tpr,orient = 'index').T
            tpr = tpr.melt(value_name = 'tpr')
            
            roc_i = pd.concat([fpr, tpr['tpr']], axis = 1)
            roc_i['model_name'] = m
            roc_i['sample_nr']  = s
            roc = pd.concat([roc, roc_i])
        
    
    # data_plot_line = pd.DataFrame({'x':[0,1], 'y':[0,1]})
    # + geom_line(data = data_plot_line, mapping = aes(x='x', y='y' )) 

    # przygotowanie tabeli roc_auc do przeksztalcen i wyswietlenie
    roc_auc = pd.DataFrame(roc_auc.items(), columns=['model', 'roc_auc'])
    roc_auc_col_split = roc_auc['model'].str.split('-', n=3, expand=True)
    roc_auc_col_split.columns = ['model', 'sample', 'label']
    roc_auc = pd.concat([roc_auc_col_split, roc_auc['roc_auc']], axis=1)

    # przygotowanie tabeli average_precision do przeksztalcen i wyswietlenie
    average_precision = pd.DataFrame(average_precision.items(), columns=['model', 'average_precision'])
    average_precision_col_split = average_precision['model'].str.split('-', n=3, expand=True)
    average_precision_col_split.columns = ['model', 'sample', 'label']
    average_precision = pd.concat([average_precision_col_split, average_precision['average_precision']], axis=1)

    # statystyki podsumowujace tabele roc_auc
    roc_auc_summary = roc_auc.groupby('model').agg(mean=pd.NamedAgg(column='roc_auc', aggfunc=lambda x: np.nanmean(x))
                                                   , median=pd.NamedAgg(column='roc_auc',
                                                                        aggfunc=lambda x: np.nanmedian(x))
                                                   , std=pd.NamedAgg(column='roc_auc', aggfunc=lambda x: np.nanstd(
            x))).reset_index().sort_values(['mean'], ascending=[False])

    # statystyki podsumowujace tabele average_precision
    average_precision_summary = average_precision.groupby('model').agg(
        mean=pd.NamedAgg(column='average_precision', aggfunc=lambda x: np.nanmean(x))
        , median=pd.NamedAgg(column='average_precision', aggfunc=lambda x: np.nanmedian(x))
        , std=pd.NamedAgg(column='average_precision', aggfunc=lambda x: np.nanstd(x))).reset_index().sort_values(
        ['mean'], ascending=[False])



    display(ggplot() + geom_step(data = pr,  mapping = aes(x='precision', y='recall', color =  group )) + facet_grid(facet)        + ggtitle('precision-recall'))
    display(ggplot() + geom_step(data = roc, mapping = aes(x='fpr',       y='tpr',    color =  group))  + facet_grid(facets=facet) + ggtitle('roc'))

    print('roc_auc')
    print(np.round(roc_auc,3))
    print('average_precision')
    print(np.round(average_precision,3))
    print('roc_auc stability')
    print(np.round(roc_auc_summary,3))
    print('average precision stability')
    print(np.round(average_precision_summary,3))
  
  def show_conf_matrix(  self
                       , simulation_name
                       , model_name
                       , sample_nr = 0
                       , set_type = 'test'
                       , sample_type = 'full'
                       , thresholds_priori_id = None
                       , if_automatic = 0):
                         
    """
    Printing confusion matrix in nice readable way
    """
    # getting confusion matrix data and reshaping it into square shape
    conf_vec         = self.confusion_matrix.loc[ (self.confusion_matrix['if_automatic']==if_automatic) &  (self.confusion_matrix['simulation_name']==simulation_name) & (self.confusion_matrix['model_name']==model_name) & (self.confusion_matrix['set_type']==set_type)  & (self.confusion_matrix['sample_nr']==sample_nr) & (self.confusion_matrix['sample_type']==sample_type) & (self.confusion_matrix['threshold_priori_id']==thresholds_priori_id), ]
    col_num          = conf_vec.shape[1]
    conf_vec         = conf_vec.iloc[:,8:col_num]
    col_num          = conf_vec.shape[1]
    conf_matrix_size = int(col_num**0.5)
    conf_vec         = np.array(conf_vec)
    confusion_matrix = conf_vec.reshape((conf_matrix_size, conf_matrix_size))
    
    # diagonal elements of confusion matrix (right predicted by the model)
    diagonal = list()
    for i in range(confusion_matrix.shape[1]):
      diagonal.append(confusion_matrix[i,i])
    
    # sums by rows and columns of confusion matrix
    sum_row = confusion_matrix.sum(axis=0)
    sum_col = confusion_matrix.sum(axis=1)
    
    # percentage of right predicted by rows and columns
    percent_row = pd.Series(np.around(diagonal/sum_row * 100, 2))
    percent_col = pd.Series(np.around(diagonal/sum_col * 100, 2))
    
    # Putting all elements into one matrix.
    sum_row = pd.Series(sum_row)
    sum_col = pd.Series(sum_col)
    
    confusion_matrix = pd.DataFrame(confusion_matrix)
    
    confusion_matrix = confusion_matrix.append(sum_row,ignore_index=True)
    confusion_matrix = confusion_matrix.append(percent_row,ignore_index=True)
    
    confusion_matrix = confusion_matrix.assign(__sum__=sum_col)
    confusion_matrix = confusion_matrix.assign(__error__=percent_col)
    
    confusion_matrix.columns = self.y_labels + ['_sum_'] + ['_correctness_']
    confusion_matrix.index = self.y_labels + ['_sum_'] + ['_correctness_']
    
    return(confusion_matrix)
  
  
  def print_simulation_metainfo(simulation_name = 's1'):
    
    """
    Printing metainformations
    """
    
    sim = self.simulation_metainfo.loc[self.simulation_metainfo['simulation_name'] == simulation_name,:]
    mod = self.models[simulation_name]
    print('SIMULATION BASIC INFO: \n')
    print(sim)
    print('\n')
    print('MODELS PARAMETERS: \n')
    for i in mod:
      print(i)
      print(mod[i])
      print('\n')
      
  
  def save_results(self, path, override = True):
    
    """
    currenlty only overriding is implemented
    """
    
    if override == True:
      import pickle
      data = {}
      
      data['classification'] = self.classification
      data['probabilities'] = self.probabilities
      data['scores']    = self.scores
      data['confusion_matrix'] = self.confusion_matrix
      
      data['simulation_metainfo'] = self.simulation_metainfo
      
      data['priori']    = self.priori
      data['threshold'] = self.threshold
      
      data['y_train_count'] = self.y_train_count
      data['y_test_count']  = self.y_test_count
      
      data['feature_importance']      = self.feature_importance
      data['feature_importance_enet'] = self.feature_importance_enet
      
      data['models'] = self.models
      
      data['correlation_numeric_pearson']      = self.correlation_numeric_pearson
      data['correlation_numeric_kendall']      = self.correlation_numeric_kendall
      data['correlation_mix_ratio']            = self.correlation_mix_ratio
      data['correlation_categorical_v_cramer'] = self.correlation_categorical_v_cramer
      data['vif'] = self.vif
      
      with open(path, 'wb') as file:
        pickle.dump(data, file)
      file.close()
      
  
  def load_results(self, path):
    
    """
    Loading pickle file with simulations restuls (saved before with function 'save results')
    """
    
    import pickle
    
    file = open(path, 'rb')
    data = pickle.load(file)
    file.close()
    
    self.classification   = data['classification']
    self.probabilities    = data['probabilities']
    self.scores           = data['scores']
    self.confusion_matrix = data['confusion_matrix'] 
    
    self.priori    = data['priori']
    self.threshold = data['threshold']
    
    self.y_train_count = data['y_train_count']
    self.y_test_count  = data['y_test_count']
    
    self.feature_importance      = data['feature_importance']
    self.feature_importance_enet = data['feature_importance_enet']
    
    
    self.simulation_metainfo = data['simulation_metainfo']
    
    self.models = data['models']
    
    self.correlation_numeric_pearson  = data['correlation_numeric_pearson']   
    self.correlation_numeric_kendall  = data['correlation_numeric_kendall']  
    self.correlation_mix_ratio        = data['correlation_mix_ratio']    
    self.correlation_categorical_v_cramer = data['correlation_categorical_v_cramer']
    self.vif = data['vif']
  
  

  def stacking_between_sim(self, simulation_new_name = 'sim_new', simulation_params = [], method = 'mean'):
    """
    only for 'full'
    simulation_params = [[sim, model],[sim, model]]
    """
    
    
    # loop over simulations
    for i, sim_ in enumerate(simulation_params):
      
        sim = sim_[0]
        data_i   = self.probabilities.loc[ (self.probabilities['sample_type']=='full') & (self.probabilities['simulation_name']==sim), :]
        models_i = sim_[1:]
        
        # loop over models in simulations
        for j, model in enumerate(models_i):
          if (i == 0) and (j == 0): # pierwsza iteracja petli
              data_identifiers = data_i.loc[data_i['model_name'] == model, data_i.columns.difference(self.y_labels) ]
              data_identifiers = data_identifiers.reset_index(drop = True)
              prob_array_3D = np.array(data_i.loc[data_i['model_name'] == model, self.y_labels])
              prob_array_3D = np.expand_dims(prob_array_3D, 0)
          else:
              prob_array_3D_i_j = data_i.loc[data_i['model_name'] == model, self.y_labels]
              prob_array_3D_i_j = np.expand_dims(prob_array_3D_i_j, 0)
              prob_array_3D = np.append(prob_array_3D, prob_array_3D_i_j, axis=0)
    
    # probability average over dim=0
    prob_array_mean = np.apply_over_axes(func=lambda x,y: np.mean(x,y) , a=prob_array_3D, axes=[0])
    
    # removing third dimension (after calculeting mean it is no more necessary)
    prob_array_mean = np.squeeze(prob_array_mean, axis=0)
    
    # tranform to DataFrame and add identifiers
    prob_mean = pd.DataFrame(prob_array_mean, columns = self.y_labels)
    prob_mean = pd.concat([data_identifiers, prob_mean], axis = 1)
    
    # name of stacked model and simulation
    prob_mean['model_name'] = 'model_stack'
    prob_mean['simulation_name'] = simulation_new_name
    
    # saving results
    self.probabilities = pd.concat([ self.probabilities[self.probabilities['simulation_name']!=simulation_new_name ], prob_mean])
  
  def scores_cf_agg_full(self, simulation_params = [], if_automatic = 0, pos_label = '1'):
    
    """
    Calculating scores for aggregated result from different simulations (only 'full' sample_type).
    From each simulation only one model can be used.
    simulation_params = [['sim_1','cat','tp1'], ['sim_2','xgb','tp2']]
    """
    
    # getting classification data
    data_f = self.classification.loc[(self.classification['if_automatic']==if_automatic) & (self.classification['sample_type']=='full'), :]
    data_f.drop(columns='index') # nie potrzebujemy indeksu bo wyniki sa agregacja
    
    # filtering data by models and simulations)
    data = pd.DataFrame()
    model_name_agg = ''
    simulation_name_agg = ''
    threshold_priori_id_agg = ''
    for params in simulation_params:
        data = pd.concat([data, data_f.loc[(data_f['simulation_name'] == params[0]) & (data_f['model_name'] == params[1]) & (data_f['threshold_priori_id'] == params[2]), : ]], axis = 0)
        simulation_name_agg = simulation_name_agg + '_' + params[0]
        model_name_agg      = model_name_agg + '_' + params[1]
        threshold_priori_id_agg = threshold_priori_id_agg + params[2]
    # splitting data into train and test (simulations and models are mixed here as we want)
    train = data.loc[data['set_type'] == 'train', : ]
    test  = data.loc[data['set_type'] == 'test', : ]
    
    
    
    # creating indicies columns for aggregated results
    group_structure = ['if_automatic', 'model_name', 'sample_nr', 'sample_name', 'sample_type', 'set_type', 'simulation_name','threshold_priori_id']
    
    # model_name_agg = '_'.join(list(simulation_params.values()))
    # simulation_name_agg = '_'.join(list(simulation_params.keys()))
    
    
    # creating empty (with only indicies) data frames to collect aggregated results
    train_group = pd.DataFrame([[if_automatic, model_name_agg, 0, '_', 'full', 'train', simulation_name_agg, threshold_priori_id_agg]], columns = group_structure )
    test_group  = pd.DataFrame([[if_automatic, model_name_agg, 0, '_', 'full', 'test',  simulation_name_agg, threshold_priori_id_agg]], columns = group_structure )
    
    
    
    # scores
    scores_train = self.scores_determining( y_true = train['y_true'], y_pred = train['y_pred'], pos_label = pos_label )
    scores_test  = self.scores_determining( y_true = test['y_true'],  y_pred = test['y_pred'],  pos_label = pos_label )
    
    # dodanie zmiennych grupujacych
    scores_train = pd.concat([train_group, scores_train], axis = 1)
    scores_test  = pd.concat([test_group,  scores_test], axis = 1)
    
    # wypelnienie zmiennych grupujacych
    scores_train[group_structure] = scores_train[group_structure].fillna(method = 'ffill')
    scores_test[group_structure]  = scores_test[group_structure].fillna(method = 'ffill')
    
    
    
    # confusion matrices
    cf_train = pd.DataFrame([skl.metrics.confusion_matrix(y_true = train['y_true'] , y_pred = train['y_pred']).flatten()])
    cf_test  = pd.DataFrame([skl.metrics.confusion_matrix(y_true = test['y_true']  , y_pred = test['y_pred']).flatten()])
    
    cf_train = pd.concat([train_group, cf_train], axis = 1)
    cf_test  = pd.concat([test_group,  cf_test],  axis = 1)
    
    
    # saving scores
    if len(self.scores) != 0:
      self.scores = pd.concat([self.scores.loc[self.scores['simulation_name'] != simulation_name_agg, : ], scores_train, scores_test])
    
    
    
    # saving confussion matrix
    if cf_train.shape[0] != 0:
        if len(self.confusion_matrix) != 0:
          self.confusion_matrix = pd.concat([self.confusion_matrix.loc[self.confusion_matrix['simulation_name'] != simulation_name_agg, : ], cf_train, cf_test])
    
    
  
  
  
  # FAST PLOTS
  
  def fast_correlation_matrices(self, simulation_name):
    
    """
    Printing set of tables with correlations between feature variables
    """
    
    
    return( self.sbs(
      [self.correlation_numeric_pearson.get(simulation_name, pd.DataFrame())
      ,self.correlation_numeric_kendall.get(simulation_name, pd.DataFrame())
      ,self.correlation_mix_ratio.get(simulation_name, pd.DataFrame())
      ,self.correlation_categorical_v_cramer.get(simulation_name, pd.DataFrame())
      ,self.vif.get(simulation_name, pd.DataFrame())], ['pearson', 'kendall', 'ratio', 'v_cramer','vif']))
  
  
  def fast_dens_plot( self
                    , simulation_name = None 
                    , y_category      = '1'
                    , score           = 'balanced_accuracy'
                    , plot_density_sample_frac = 0.5
                    , sample_type = 'full'
                    , facet = 'model_name~.'
                    , fig_w = 15
                    , fig_h = 5):
    """
    Density plots for train and test sets only (sample_type == 'full') 
    """
    
    # return without list !!!. If not, plot will be duplicated
    if type(sample_type) != list:
      sample_type = [sample_type]
    
    if sample_type[0] in list(self.probabilities['sample_type'].drop_duplicates()):
      display(self.plot_probabilities_density(data_new = None
                                      , x_var  = y_category
                                      , filter = {'set_type' : ['train'], 'sample_type':sample_type, 'simulation_name':[simulation_name]}
                                      , fill_var = 'y_true'
                                      , facet  = facet
                                      , x_lim  = [np.nan, np.nan]
                                      , adjust = 0.5
                                      , alpha  = 0.5
                                      , size   = 0.5
                                      , frac   = plot_density_sample_frac
                                      , title  = 'train'
                                      , fig_w  = fig_w
                                      , fig_h  = fig_h) # train
      ,self.plot_probabilities_density(data_new = None
                                      , x_var  = y_category
                                      , filter = {'set_type' : ['test'], 'sample_type':sample_type, 'simulation_name':[simulation_name]}
                                      , fill_var = 'y_true'
                                      , facet  = facet
                                      , x_lim  = [np.nan, np.nan]
                                      , adjust = 0.5
                                      , alpha  = 0.5
                                      , size   = 0.5
                                      , frac   = plot_density_sample_frac
                                      , title  = 'test'
                                      , fig_w  = fig_w
                                      , fig_h  = fig_h ) # train # test
      

      )
  
  def fast_scores_plot( self
                      , simulation_name = None 
                      , y_category  = '1'
                      , score       = 'balanced_accuracy'
                      , sample_type = 'cv'):
    
    """
    Plot line plot with scores for train and test test (sample_type can be selected). This is wrapper for method 'plot_scores'
    """
    
    display(#self.plot_scores(filter = {'if_automatic':[0], 'sample_type':[sample_type], 'set_type' : ['train'], 'simulation_name':[simulation_name]}, x_var = 'sample_nr', y_var='score_value', fill_var = 'model_name', facet = 'threshold_priori_id~score_name')
            #,self.plot_scores(filter = {'if_automatic':[0], 'sample_type':[sample_type], 'set_type' : ['valid'], 'simulation_name':[simulation_name]}, x_var = 'sample_nr', y_var='score_value', fill_var = 'model_name', facet = 'threshold_priori_id~score_name')
            self.plot_scores(filter = {'if_automatic':[0], 'sample_type':[sample_type], 'set_type' : ['test'],  'simulation_name':[simulation_name]}, x_var = 'sample_nr', y_var='score_value', fill_var = 'model_name', facet = 'threshold_priori_id~score_name'))

  def fast_conf_matrix_scores_flat_version(self, simulation_name, fig_w = 10, fig_h = 2.5):
    # flat version of conf matrix and score
    cf_matrix = self.confusion_matrix[(self.confusion_matrix['simulation_name']==simulation_name) & (self.confusion_matrix['sample_type']=='full') & (self.confusion_matrix['if_automatic']==0) ].drop(columns={'simulation_name','if_automatic','sample_name','sample_nr','sample_type'})
    scores_matrix = self.scores[(self.scores['simulation_name']==simulation_name) & (self.scores['sample_type']=='full') & (self.scores['if_automatic']==0) ].drop(columns={'simulation_name','if_automatic','sample_name','sample_nr','sample_type'})
    scores_matrix = scores_matrix.rename(columns = {'score_value':'_'})

    scores_matrix = scores_matrix.set_index(['model_name','set_type','threshold_priori_id','score_name'])
    scores_matrix = scores_matrix.unstack(3)

    # usuwanie indeksów wierszowych
    scores_matrix = scores_matrix.reset_index(drop=False)

    # laczenie indeksow kolumnowych
    scores_matrix.columns = [''.join(col).strip() for col in scores_matrix.columns.values]

    # poprawka nazw indeksów kolumnowych
    scores_matrix = scores_matrix.rename(columns={'model_name_':'model_name','set_type_':'set_type','threshold_priori_id_':'threshold_priori_id'})
    cf_scores_matrix = pd.merge(cf_matrix, scores_matrix, on=['model_name','set_type','threshold_priori_id'])
    cf_scores_matrix.columns = cf_scores_matrix.columns.astype(str)

    cf_scores_matrix_test = cf_scores_matrix.loc[cf_scores_matrix['set_type']=='test']
    plotnine.options.figure_size = (fig_w, fig_h)
    display(ggplot(data=cf_scores_matrix_test) + geom_line(aes(x='threshold_priori_id', y='3', color='model_name', fill='model_name', group='model_name')) + ggtitle('TP (test)'))
    display(ggplot(data=cf_scores_matrix_test) + geom_line(aes(x='threshold_priori_id', y='_recall', color='model_name', fill='model_name',group='model_name')) + geom_point(aes(x='threshold_priori_id', y='_recall', color='model_name', fill='model_name'))  + ggtitle('_recall  (test)'))
    display(ggplot(data=cf_scores_matrix_test) + geom_line(aes(x='threshold_priori_id', y='_precision', color='model_name', fill='model_name', group='model_name')) + geom_point(aes(x='threshold_priori_id', y='_precision', color='model_name', fill='model_name')) + ggtitle('precision (test)'))

    return(cf_scores_matrix)


  def fast_conf_matrix_scores(self, simulation_name):
    
    """
    Printing confusion matrix, scores for sample_type = 'full' but for all models and all cut_off/priori. Additionaly information about y_counts and threshold are printted.
    """
    
    
    #(1) confussion matrices (only train and test from 'full' but for all models)
    models_list = self.confusion_matrix.loc[(self.confusion_matrix['simulation_name']==simulation_name) & (self.confusion_matrix['sample_type']=='full') , 'model_name'].drop_duplicates()
    
    # threshold_priori list for model: all model have the same thresholds so is not necessery to nest this code in loop
    thresholds_priori_list = self.confusion_matrix.loc[(self.confusion_matrix['simulation_name']==simulation_name) & (self.confusion_matrix['sample_type']=='full') , 'threshold_priori_id'].drop_duplicates()
    
    
    cf_train_list = []
    cf_test_list  = []
    
    labels_to_display = []
    
    for i in models_list:
      for j in thresholds_priori_list:
        if len(self.confusion_matrix.loc[(self.confusion_matrix['sample_type']=='full') & (self.confusion_matrix['model_name']==i)  & (self.confusion_matrix['threshold_priori_id']==j),:] ) != 0: # protection for 'fip'
          cf_train = self.show_conf_matrix( simulation_name = simulation_name
                         , model_name   = i
                         , sample_nr    = 0
                         , set_type     = 'train'
                         , sample_type  = 'full'
                         , thresholds_priori_id = j
                         , if_automatic = 0)
          
          cf_test = self.show_conf_matrix( simulation_name = simulation_name
                   , model_name   = i
                   , sample_nr    = 0
                   , set_type     = 'test'
                   , sample_type  = 'full'
                   , thresholds_priori_id = j
                   , if_automatic = 0)
          
          labels_to_display = labels_to_display + [i + '__' + j] 
          
          cf_train = cf_train.fillna(0)
          cf_test = cf_test.fillna(0)
          
          cf_train_list = cf_train_list + [cf_train.astype(int)]  
          cf_test_list  = cf_test_list  + [cf_test.astype(int)]
    
    
    #(2) feature_importance
    feature_importance = self.feature_importance.loc[self.feature_importance['simulation_name']==simulation_name,:]
    feature_importance = feature_importance.reset_index(drop=True) # index reset so function 'sbs' works properly
    
    feature_importance_rf     = feature_importance.loc[feature_importance['method'] == 'Random_Forest_Importance', ['Feature', 'Importance']]
    feature_importance_rf_per = feature_importance.loc[feature_importance['method'] == 'RandomForest_permutation', ['Feature', 'Importance']]
    feature_importance_cat    = feature_importance.loc[feature_importance['method'] == 'catboost', ['Feature', 'Importance']]
    if len(self.feature_importance_enet) > 0 : # warunek jest sprawdzany bo np. dla zmiennych niebinarnych jest nieliczony
      feature_importance_enet   = self.feature_importance_enet.loc[self.feature_importance_enet['simulation_name'] == simulation_name,['Feature', 'Importance', 'Importance_abs', 'l1']]
    else:
      feature_importance_enet = pd.DataFrame({'elastic_net':['not calculated']})
    
    #(3) scores
    scores_names_list = self.scores.loc[self.scores['simulation_name']==simulation_name, 'score_name'].drop_duplicates()
    
    scores_list  = []
    
    for i in scores_names_list:
      
      scores_i = self.scores.loc[(self.scores['if_automatic']==0) & (self.scores['simulation_name']==simulation_name) & (self.scores['sample_type']=='full') & (self.scores['score_name']==i),['model_name',	'set_type', 'threshold_priori_id', 'score_value']] 
      
      scores_i = scores_i.reset_index(drop=True) # index reset so function 'sbs' works properly
      
      scores_list = scores_list + [scores_i]
    
    
    #(4) y counts
    y_train_count = self.y_train_count.loc[self.y_train_count['simulation_name'] == simulation_name, :]
    y_train_count = y_train_count.drop('simulation_name', axis = 1)
    
    y_test_count  = self.y_test_count.loc[self.y_test_count['simulation_name'] == simulation_name, :] 
    y_test_count  = y_test_count.drop('simulation_name', axis = 1)
    
    #(5) thresholds
    # thresholds = self.threshold.get(simulation_name, {'no_thresholds':'NaN'})
    # thresholds = pd.DataFrame(list(thresholds.values()), index = list(thresholds.keys()))
    # 
    # priori = self.priori.get(simulation_name, {'no_priori':'NaN'})
    # priori = pd.DataFrame(list(priori.values()), index = list(priori.keys()))
    
    
    
    
    ['Confusion matrix  TRAIN ,   Model:  '   + x for x in models_list]
    
    # displaying all elements (1-5)
    
    return([ self.sbs( [y_train_count, y_test_count]       ,['target train','target test'] )
            ,self.sbs( cf_train_list                                  ,['Confusion matrix  TRAIN ,   Model:  ' + x for x in labels_to_display] )
            ,self.sbs( cf_test_list                                   ,['Confusion matrix  TEST,    Model:  '  + x for x in labels_to_display] )
            #,self.sbs( scores_list                                    ,['score: ' + x for x in scores_names_list]) # na razie wyłączone. Za dużo tabel które można połączyć w jedną mniejszą tabelę.
            ,self.sbs( [feature_importance_enet, feature_importance_rf, feature_importance_rf_per, feature_importance_cat],['Feature Importance e-net', 'Feature Importance RF','Feature Permutation Importance RF', 'Feature Importance CATBOOST'])
            ,'recall = TP / TP + FN;   precision = TP / (TP + FP);    f1 = ( 2*TP ) / (2*TP + FP + FN );   balanced_accuracy = (TP/(TP+FN) + TN/(TN+FP)) / 2 '])
  
  
      
    
  def map_quantile_threshold(self, simulation_name, model_name, label_to_map = '1', quantiles = [0.5], set_type = 'test', filter_test_indicies = None):
    """
    map_quantile_threshold(simulation_name, model_name, '0', [0.5, 0.2], 'test')
    używać do klasyfikacji binarnej. W zagadnieniach wieloklasowych raczej nie stosuje się punktu odciecia
    """
    
    labels = self.y_labels
    thr_list = []
    for quantile in quantiles:
    
        pr = self.probabilities.loc[(self.probabilities['simulation_name']==simulation_name) & 
                         (self.probabilities['model_name']==model_name) &
                        (self.probabilities['sample_type']=='full') &
                        (self.probabilities['set_type']==set_type),
                         ['y_true', 'index', labels[0], labels[1] ] ]
        
        if set_type == 'test' and filter_test_indicies != None:
          pr = pr[pr['index'].isin(filter_test_indicies)]
        
        
        pr = pr.reset_index(drop = True)
        pr['quantile_label'] = pd.qcut(pr[label_to_map].values, q=[0,quantile,1], labels = ['under_q','above_q'])
        thr = np.min(pr.loc[pr['quantile_label']=='above_q',label_to_map].values)
        thr_list = thr_list + [thr] 
    return(thr_list)
  
  
  def summary_big(self, simulation_name, pos_label, cv_n, hold_n, by_var):
    display(self.h('meta informations about simulation'))
    display(self.simulation_metainfo.loc[self.simulation_metainfo['simulation_name'] == simulation_name])
    display(self.h('correlation between features'))
    display(self.fast_correlation_matrices(simulation_name = simulation_name))
    display(self.h('target distribution, confussion matrix, scores and feature importance'))
    display(self.fast_conf_matrix_scores(simulation_name = simulation_name))
    display(self.h('flat confusion matrix and scores'))
    display(self.fast_conf_matrix_scores_flat_version(simulation_name=simulation_name))
    display(self.h('density plots'))
    display(self.fast_dens_plot(simulation_name=simulation_name, sample_type = ['full'], fig_w=15, fig_h=5))
    display(self.h('density plot for Feature Importance Permutation'))
    display(self.fast_dens_plot(simulation_name=simulation_name, sample_type = ['fip','full'], fig_w=15, fig_h=10))
    if cv_n is not None:
        display(self.h('density plot for cv'))
        display(self.fast_dens_plot(simulation_name=simulation_name, sample_type=['cv'], facet = 'sample_nr~model_name', fig_w=15, fig_h=10))
    if hold_n:
        display(self.h('density plot for hold'))
        display(self.fast_dens_plot(simulation_name=simulation_name, sample_type=['hold'], facet = 'sample_nr~model_name', fig_w=15, fig_h=10))
    if cv_n is not None:
      display(self.h('scores by cross validation'))
      display(self.fast_scores_plot(simulation_name=simulation_name, sample_type = 'cv'))
      display(self.scores_volatility(simulation_name=simulation_name, sample_type='cv', set_type = ['test']))
    if hold_n is not None:
      display(self.h('scores by holdout'))
      display(self.fast_scores_plot(simulation_name=simulation_name, sample_type = 'hold'))
      display(self.scores_volatility(simulation_name=simulation_name, sample_type='hold', set_type = ['test']))
    if by_var is not None:
        display(self.h('scores by by_var'))
        display(self.fast_scores_plot(simulation_name=simulation_name, sample_type = 'by_var'))
        display(self.scores_volatility(simulation_name=simulation_name, sample_type='by_var', set_type = ['test']))

    display(self.h('pr and roc curve'))
    display(self.plot_roc_pr_curves(simulation_name=simulation_name, set_type='test', sample_type = ['full'], y_category=pos_label))
    display(self.h('pr and roc curve for Feature Importatnce Permutation'))
    display(self.plot_roc_pr_curves(simulation_name=simulation_name, set_type='test', sample_type = ['fip', 'full'], y_category=pos_label, group = 'model_name', facet = 'sample_nr~variable'))
    if cv_n is not None:
        display(self.h('pr and roc curve for cv'))
        display(self.plot_roc_pr_curves(simulation_name=simulation_name, set_type='test', sample_type=['cv'], y_category=pos_label, group='sample_nr', facet='model_name~variable'))
    if hold_n is not None:
        display(self.h('pr and roc curve for hold'))
        display(self.plot_roc_pr_curves(simulation_name=simulation_name, set_type='test', sample_type=['hold'], y_category=pos_label, group='sample_nr', facet='model_name~variable'))
  
  
  
  def h(self, text, size = 3, bold = True, color = 'blue'):
    if not bold:
        return(HTML('<font size = "' + str(size) + '" color= "' + color + '" >' + text + '</font>' ))
    else:
        return(HTML('<font size = " ' + str(size) + '" color= "' + color + '" ><b>' + text + '</b></font>' ))
  
  
  def sbs(self, dfs:list, captions:list):
    
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
                   # .format(formatter="{:,}")
                   )
        
        output += "\xa0\xa0\xa0"
    display(HTML(output))



 













