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
import copy  # deepcopies
import scipy
import imblearn  # problem of imbalanced samples
import ennemi

from scipy import stats
from scipy.special import boxcox, inv_boxcox

from rfpimp import permutation_importances
from sklearn.ensemble import IsolationForest  # for outliered detecting

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from plotnine.themes.themeable import axis_line
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMRegressor
# from xgboost                   import XGBRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing     import Imputer
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn import preprocessing
from sklearn.impute import SimpleImputer as Imputer
import catboost
from catboost import CatBoostRegressor, Pool, cv
from plotnine import *  # ggplot for python
import shap
# https://www.kaggle.com/discdiver/category-encoders-examples
import category_encoders as ce
from IPython.display import display_html

from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_poisson_deviance
from sklearn.metrics import mean_gamma_deviance
from sklearn.metrics import mean_tweedie_deviance


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
      # cv_n  = 2
      # hold_n = 4; hold_train_sample_size = 0.1; hold_test_sample_size = 0.1
      # fip_n = 1


      # c1 = classification_model()

      # c1.simulation(  train_set = train_set, test_set = test_set, y_var = y_var, x_var_num = x_var_num, x_var_cat = x_var_cat
      #               , cv_n=cv_n, fip_n=fip_n, hold_n = hold_n, hold_train_sample_size = hold_train_sample_size, hold_test_sample_size = hold_test_sample_size
      #               , data_check=True, models_list=models_list, correlation = True, model_fip = RandomForestClassifier(n_estimators=30, max_depth=7)
      #               , simulation_name = simulation_name, description = simulation_description)


      # c1.performance(simulation_name=simulation_name)



    """

    def __init__(self):
        """
        initialization of variables to store results of calculations.
        """

        # DIFFERENT METADATA :

        # metadata about simulations
        self.simulation_metainfo = pd.DataFrame()

        # models hyperparameters
        self.models = {}


        # labels of categories for dependent variable
        self.y_labels = []


        # names and indicies for categorical and numerical variables
        self.cat_vars = []
        self.cat_vars_indicies = []
        self.num_vars = []
        self.num_vars_indicies = []

        # information about missing values and variables types in trai set
        self.data_check_result = pd.DataFrame()

        # count
        self.y_train_count = pd.DataFrame()
        self.y_test_count = pd.DataFrame()

        # SIMULATION AND PERFORMANCE RESULTS :

        # correlation matrices
        self.correlation_numeric_pearson = dict()
        self.correlation_numeric_kendall = dict()
        self.correlation_mix_ratio = dict()
        self.correlation_categorical_v_cramer = dict()
        self.vif = dict()

        # scores
        self.scores = pd.DataFrame()

        # prediction (istead of probabilities)
        self.prediction = pd.DataFrame()


        # feature importatnce
        self.feature_importance = pd.DataFrame()
        self.feature_importance_enet = pd.DataFrame()

    def data_check(self
                   , data
                   , set_name='train'
                   , x_var=None
                   , y_var=None
                   , x_var_cat=None):

        """
        checking number of missing values and columns types on train set.
        """

        # if duplicated index
        if any(data.index.duplicated()):
            raise ValueError('LUCAS: ' + set_name + ' data sets has duplicated indicies')

        # if variable is a DataFrame
        if str(type(data)).find('DataFrame') < 1:
            raise ValueError('LUCAS: ' + set_name + ' data sets is not Pandas DataFrame')

        # checking data length
        if len(data) == 0:
            raise ValueError('LUCAS: length of ' + set_name + ' data set is 0')

        # checking if data includes all variables
        if y_var is None or x_var is None:
            raise ValueError('LUCAS: variables names not provided')
        if set(x_var) != set(x_var).intersection(set(data.columns)):
            raise ValueError('LUCAS: not all features variables are included in ' + set_name + ' set')
        elif set([y_var]) != set([y_var]).intersection(set(data.columns)):
            raise ValueError('LUCAS: target variable is not included in ' + set_name + ' set')

        # checking if all 'object' columns are included in 'x_var_cat'
        cols_types = pd.DataFrame(data.dtypes, columns=['type']).reset_index(drop=False)
        cols_types = cols_types.loc[(cols_types['type'] == 'object') & (cols_types['index'] != y_var), :]
        if len(x_var_cat) > 0:
            if x_var_cat is None:
                raise ValueError('LUCAS: Data contains string columns but not x_var_cat is specified')
            elif len(set(cols_types['index']) - set(x_var_cat)) > 0:
                raise ValueError('LUCAS: In data set there are object columns not specified in "x_var_cat"')

        # checking if number o target categories > 1
        if len(data[y_var].drop_duplicates()) < 2:
            raise ValueError(
                'LUCAS: In target variable there is only 1 category. For regression task at least 2 are required')

        # checking missing value and data type
        df_na = data.apply(lambda x: sum(pd.isnull(x))).T
        df_types = data.dtypes

        df_na = pd.concat([df_na, df_types], axis=1)
        df_na.columns = ['missing', 'type']
        self.data_check_result = df_na
        if len(df_na.loc[df_na['missing'] > 0, :]) > 0:
            print(df_na)
            raise ValueError(
                'LUCAS: there are missing values in data set. Get attibute "data_check_result" to see details')

        # checking infinity values
        df_num = data.select_dtypes('number')
        df_inf = pd.DataFrame(np.isinf(df_num), columns=list(df_num.columns)).sum()
        if len(df_inf[df_inf == True]) > 0:
            print(df_inf)
            raise ValueError('LUCAS: there are infinite values in data set')

        return ([df_na, df_inf])

    def remove_simulations(self, simulations_names, keep=False):

        """
        removing listed simulations from all objects
        """

        if keep:
            simulations_names = set(list(self.simulation_metainfo['simulation_name'])).diffecence(
                set(simulations_names))
            simulations_names = list(simulations_names)

        # self.simulation_metainfo.drop(self.simulation_metainfo[ self.simulation_metainfo['simulation_name'].isin(simulations_names) ].index, inplace=True ) # ku przetrodze
        self.simulation_metainfo = self.simulation_metainfo[
            ~self.simulation_metainfo['simulation_name'].isin(simulations_names)]  # ku przestrodze
        for x in simulations_names:
            try:
                del self.models[x]
            except:
                print('no ' + x + ' found in self.models')
            try:
                del self.correlation_numeric_pearson[x]
            except:
                print('no ' + x + ' found in self.correlation_numeric_pearson')
            try:
                del self.correlation_numeric_kendall[x]
            except:
                print('no ' + x + ' found in self.correlation_numeric_kendall')
            try:
                del self.correlation_numeric_ratio[x]
            except:
                print('no ' + x + ' found in self.correlation_numeric_ratio')
            try:
                del self.correlation_numeric_v_cramer[x]
            except:
                print('no ' + x + ' found in self.correlation_numeric_v_cramer')

        try:
            self.y_train_count = self.y_train_count[~self.y_train_count['simulation_name'].isin(simulations_names)]
        except:
            print('no ' + x + ' found in self.y_train_count')

        try:
            self.y_test_count = self.y_test_count[~self.y_test_count['simulation_name'].isin(simulations_names)]
        except:
            print('no ' + x + ' found in self.y_test_count')

        try:
            self.scores = self.scores[~self.scores['simulation_name'].isin(simulations_names)]
        except:
            print('no ' + x + ' found in self.scores')

        try:
            if len(self.prediction) == 0:
                print('no ' + x + ' found in self.prediction')
            self.prediction = self.prediction[~self.prediction['simulation_name'].isin(simulations_names)]
        except:
            print('no ' + x + ' found in self.prediction')

        try:
            self.feature_importance = self.feature_importance[
                ~self.feature_importance['simulation_name'].isin(simulations_names)]
        except:
            print('no ' + x + ' found in self.feature_importance')

    def prediction_determining(self
                                  , y_true
                                  , pred
                                  , simulation_name
                                  , model_name
                                  , sample_type = None
                                  , sample_nr = None
                                  , sample_name = None
                                  , set_type = None):
        """
        Determining prediction from the model
        """

        pred.insert(0, column='index', value=pred.index)

        pred = pred.reset_index(drop=True)

        pred.insert(0, column = 'y_true'          ,value = y_true.values)
        pred.insert(0, column = 'set_type'        ,value = set_type)
        pred.insert(0, column = 'sample_nr'       ,value = sample_nr)
        pred.insert(0, column = 'sample_name'     ,value = sample_name)
        pred.insert(0, column = 'sample_type'     ,value = sample_type)
        pred.insert(0, column = 'model_name'      ,value = model_name)
        pred.insert(0, column = 'simulation_name' ,value = simulation_name)



        pred['error'] = pred['y_true'] - pred['y_pred']
        pred['error_abs'] = pred['error'].abs()
        pred.loc[pred['y_true'] == 0, 'error_proc'] = np.inf
        pred.loc[pred['y_true'] != 0, 'error_proc'] = pred.loc[pred['y_true'] != 0, 'error'] / pred.loc[pred['y_true'] != 0, 'y_true']

        pred_error_std  = np.nanstd(pred['error'])
        pred_error_mean = np.nanmean(pred['error'])

        pred['error_centered'] = pred['error'] - pred_error_mean

        if pred_error_std !=0:
            pred['error_normalized'] = pred['error_centered'] / pred_error_std
        else:
            pred['error_normalized'] = np.inf



        self.prediction = pd.concat([self.prediction, pred], ignore_index=True)



    def prediction_tranformation(self
                                , simulation_name = None
                                , tranformation   = None
                                ):
        """
        Dodatkowe tranformacje wynikow predykcji
        """


    def scores_determining(self, y_true, y_pred):

        """
        calculation scores for models predictions. Scores are not saved here. It is done in f:performance
        """

        # mean_squared_error
        try:
            mean_squared_er = skm.mean_squared_error(y_true=y_true, y_pred=y_pred)
        except:
            mean_squared_er = np.nan

        # mean_squared_log_error
        try:
            mean_squared_log_er = skm.mean_squared_log_error(y_true=y_true, y_pred=y_pred)
        except:
            mean_squared_log_er = np.nan


        # explained_variance_score
        try:
            explained_var_score = skm.explained_variance_score(y_true=y_true, y_pred=y_pred)
        except:
            explained_var_score = np.nan

        # max_error
        try:
            max_error = skm.max_error(y_true=y_true, y_pred=y_pred)
        except:
            max_error = np.nan


        # median_absolute_error
        try:
            median_absolute_er = skm.median_absolute_error(y_true=y_true, y_pred=y_pred)
        except:
            median_absolute_er = np.nan


        # mean_ae
        try:
            mean_absolute_er = skm.mean_absolute_error(y_true=y_true, y_pred=y_pred)
        except:
            mean_absolute_er = np.nan

        # mean_poisson_deviance
        try:
           mean_poisson_dev = skm.mean_poisson_deviance(y_true=y_true, y_pred=y_pred)
        except:
            mean_poisson_dev = np.nan


        # mean_gamma_deviance
        try:
           mean_gamma_dev = skm.mean_gamma_deviance(y_true=y_true, y_pred=y_pred)
        except:
            mean_gamma_dev = np.nan



        # mean_tweedie_deviance
        try:
            mean_tweedie_dev = skm.mean_tweedie_deviance(y_true=y_true, y_pred=y_pred)
        except:
            mean_tweedie_dev = np.nan


        # putting all scores into Dataframe
        scores = pd.DataFrame({'score_name': [   'mean_squared_er'
                                                ,'mean_squared_log_er'
                                                ,'explained_var_score'
                                                ,'max_error'
                                                ,'median_absolute_er'
                                                ,'mean_absolute_er'
                                                ,'mean_poisson_dev'
                                                ,'mean_gamma_dev'
                                                ,'mean_tweedie_dev'],
                               'score_value': [  mean_squared_er
                                                ,mean_squared_log_er
                                                ,explained_var_score
                                                ,max_error
                                                ,median_absolute_er
                                                ,mean_absolute_er
                                                ,mean_poisson_dev
                                                ,mean_gamma_dev
                                                ,mean_tweedie_dev]})

        return (scores)


    def performance(self
                    , simulation_name='sim_1'
                    , filter_test_indicies=None
                    , loops_progress=False
                    ):

        """
        calculating models performance (scores)
        Currently can't use self.y_label, self.x_var ,  self_x_num ect.
        """

        time_start = str(datetime.datetime.now().replace(microsecond=0))


        # getting data with classification decision (brak automatic - w petli nie jest podany threshold_priori_id='' dla automatica)
        pred = self.prediction.loc[(self.prediction['simulation_name'] == simulation_name), :]

        pred = pred.drop(columns='index')  # nie potrzebujemy indeksu

        # data grouping with classification decision
        group_structure = ['simulation_name', 'model_name', 'set_type', 'sample_type', 'sample_nr', 'sample_name']  #if automatic threshold usuniete
        gr = pred.groupby(group_structure)

        # empty matrices to collect results
        scores = pd.DataFrame()

        # loop over groups
        for name, group in gr:

            # printing progress of the loop
            if loops_progress: print(name)

            # scores
            scores_i = self.scores_determining(y_true=group['y_true'], y_pred=group['y_pred'])

            # scores group
            scores_group = pd.DataFrame([list(name)], columns=group_structure)

            # connecting scores and it's group
            scores_i = pd.concat([scores_group, scores_i], axis=1)

            # filling scores group
            scores_i[group_structure] = scores_i[group_structure].fillna(method='ffill')

            # adding full information about scores to collecting variable
            scores = pd.concat([scores, scores_i])




        # saving scores
        if len(self.scores) != 0:
            self.scores = pd.concat([self.scores.loc[(self.scores['simulation_name'] != simulation_name) , :], scores])
        else:
            self.scores = scores



        time_end = str(datetime.datetime.now().replace(microsecond=0))

        # saving meta data
        self.simulation_metainfo.loc[
            self.simulation_metainfo['simulation_name'] == simulation_name, 'performance start'] = time_start
        self.simulation_metainfo.loc[
            self.simulation_metainfo['simulation_name'] == simulation_name, 'performance end'] = time_end




    def cat_encoding(self
                     , x_train=None
                     , y_train=None
                     , x_valid=None
                     , x_test=None
                     , cat_vars=None
                     , cat_encoding_method='target'):
        """
        supervised encoding of categorical variables
        # https://www.kaggle.com/discdiver/category-encoders-examples
        """
        # converting target variable into 'int' type (currently TargetEncoder does not support string target)

        if cat_encoding_method == 'target':
            encode = ce.TargetEncoder(cols=cat_vars)

        # fitting encoder
        encode.fit(x_train, y_train)

        # transforming train data set with encoded values
        data_encoded_train = encode.transform(X=x_train, y=y_train)

        # transforming valid and test data sets with encoded values
        if x_valid is not None:
            data_encoded_valid = encode.transform(X=x_valid, y=None)
        else:
            data_encoded_valid = None
        if x_test is not None:
            data_encoded_test = encode.transform(X=x_test, y=None)
        else:
            data_encoded_test = None  # None a nie pusty DataFrame bo potem inne funkcje spradzaja warunek na istnienie x_test i y_test

        return ([data_encoded_train, data_encoded_valid, data_encoded_test])

    def model_framework_identify(self, model):
        """
        Identifying which package does the model comes from
        """

        model_framework = str(type(model))  # checking only first model !!!

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

        return (model_framework)

    def feature_importance_class_CB(self
                                    , iterations=10
                                    , learning_rate=0.1
                                    , depth=5
                                    , cat_features=None
                                    , x_train=None
                                    , y_train=None
                                    , x_test=None
                                    , y_test=None):
        """
        feature importance with CatBoost (based on test set if provided)
        """

        # getting indicies of categorical variables
        if cat_features is not None:
            x_train.loc[:, cat_features] = x_train.loc[:, cat_features].astype(str)
            #         x_train.loc[:,cat_features] = x_train.loc[:,cat_features].fillna('zzz')
            cat_indicies = [x_train.columns.get_loc(c) for c in cat_features if c in x_train]
        else:
            cat_indicies = None

        # creating and fitting the catboost model
        cb = CatBoostRegressor(iterations=iterations, learning_rate=learning_rate, depth=depth)
        #print(cb)
        #cb = CatBoostRegressor(iterations=20, learning_rate=0.1, depth=5)
        cb.fit(x_train
               , y_train
               , plot=False
               , cat_features=cat_indicies
               , use_best_model=True
               , early_stopping_rounds=75
               , silent=True)
        # feature importance determining
        if y_test is None:  # for test set
            return (np.round(cb.get_feature_importance(prettified=True), 3))
        else:  # for train set
            return (np.round(cb.get_feature_importance(Pool(x_test, label=y_test, cat_features=cat_indicies), prettified=True), 3))

    def feature_importance_class_RF(self
                                    , n_estimators
                                    , max_depth
                                    , x_train
                                    , y_train
                                    , x_test=None
                                    , y_test=None):
        """
          feature importance with RandomForest
        """

        # creating and fitting RandomForest model
        rf = RandomForestRegressor(n_estimators=50
                                    , max_depth=10
                                    , n_jobs=10
                                    , oob_score=True
                                    , bootstrap=True
                                    , random_state=42)
        rf.fit(x_train, y_train)

        # regular Feature Importatnce determining
        rf_importance = rf.feature_importances_
        rf_importance = sorted(zip(x_train.columns, rf.feature_importances_), key=lambda x: x[1] * -1)
        rf_importance = np.round(pd.DataFrame(rf_importance), 2)
        rf_importance['method'] = 'Random_Forest_Importance'
        rf_importance.columns = ['Feature', 'Importance', 'method']

        def r2(rf, x_train, y_train):
            return (r2_score(y_train, rf.predict(x_train)))

        # Permutation Importance determining with use of test set (if provided)
        if y_test is None:
            rf_importance_permutation = pd.DataFrame(np.round(permutation_importances(rf, x_train, y_train, r2), 2))
        else:
            rf_importance_permutation = pd.DataFrame(np.round(permutation_importances(rf, x_test, y_test, r2), 2))
        rf_importance_permutation['method'] = 'RandomForest_permutation'
        rf_importance_permutation = rf_importance_permutation.reset_index(drop=False)

        return (pd.concat([rf_importance, rf_importance_permutation], axis=0))

    def feature_importance_elastic_net(self, l1_ratio=[0.1, 0.5, 0.9], x_train=None, y_train=None):

        # data normalisation
        x_train = pd.DataFrame(preprocessing.StandardScaler().fit(x_train).transform(x_train), index=x_train.index,
                               columns=list(x_train.columns))

        imp_coef = pd.DataFrame()

        for i in l1_ratio:
            elastic_i = SGDRegressor(loss='squared_loss', penalty='elasticnet', l1_ratio=i)
            elastic_i.fit(x_train, y_train)
            coef_i = pd.Series(elastic_i.coef_, index=list(x_train.columns))
            imp_coef_i = pd.DataFrame(coef_i)
            imp_coef_i.columns = ['Importance']
            imp_coef_i['Importance_abs'] = imp_coef_i['Importance'].abs()
            imp_coef_i['l1'] = str(i)
            imp_coef_i = imp_coef_i.sort_values(['Importance_abs'], ascending=False)
            imp_coef = pd.concat([imp_coef, imp_coef_i])

        imp_coef = imp_coef.reset_index(drop=False)
        imp_coef = imp_coef.rename(columns={'index': 'Feature'})
        imp_coef = np.round(imp_coef, 2)

        return (imp_coef)

    def correlation_cramers_v(self, var_1, var_2, round=2):
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
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)

        v_cramer = np.round(np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1))), round)

        return (v_cramer)

    def correlation_cramer_v_matrix(self, data, vars, round=2):
        """
        A wrapper for 'correlation_cramers_v' function. Returns matrix of values for any given set of binary variables.
        """
        n = len(vars)
        ar = np.empty(shape=[n, n])

        for i in range(n):
            for j in range(n):
                if i > j:
                    ar[i, j] = self.correlation_cramers_v(var_1=data[vars[i]], var_2=data[vars[j]], round=round)
                elif i == j:
                    ar[i, j] = 1
                else:
                    ar[i, j] = np.nan

        ar = pd.DataFrame(ar, columns=vars)
        ar.index = vars

        return (ar)

    def correlation_ratio(self, categories, measurements, round=2):
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

        for i in range(0, cat_num):
            cat_measures = measurements[np.argwhere(fcat == i).flatten()]
            n_array[i] = len(cat_measures)
            y_avg_array[i] = np.average(cat_measures)

        y_total_avg = np.nansum(np.multiply(y_avg_array, n_array)) / np.nansum(n_array)
        numerator = np.nansum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
        denominator = np.nansum(np.power(np.subtract(measurements, y_total_avg), 2))

        if numerator == 0:
            eta = 0.0
        else:
            eta = np.sqrt(numerator / denominator)

        return (np.round(eta, round))

    def correlation_ratio_matrix(self, data, vars_cat, vars_num, round=2):
        """
          A wrapper for 'correlation_ratio' function. Returns matrix of values for given set of variables.
        """

        ar = np.empty(shape=[len(vars_cat), len(vars_num)])

        for i in range(len(vars_cat)):
            for j in range(len(vars_num)):
                ar[i, j] = self.correlation_ratio(categories=data[vars_cat[i]], measurements=data[vars_num[j]])

        ar = pd.DataFrame(ar, columns=vars_num)
        ar.index = vars_cat

        return (np.round(ar, round))

    def correlation_pearson(self, data, vars=None, method='pearson', round=2):
        """
        Pearson correlation matrix for numerical variables
        """
        if vars is None:
            return (np.round(self.data[self.num].corr(method=method), round))
        else:
            return (np.round(data[vars].corr(method=method), round))

    def simulation(self
                   # data sets
                   , train_set=None
                   , test_set=None
                   , y_var=None
                   , x_var_num=None
                   , x_var_cat=None
                   , y_labels=None
                   # categotical variables (for catboost list of cat vars is specified in 'models_fit_parameters' )
                   , cat_encoding_method='target'  # implemented methods : 'target'
                   # models and simulation
                   , models_list=None
                   , models_fit_parameters=None
                   # Warning: The same parameter is for all models. So you cannot use models with different f:fit syntax in one simulation !!!
                   , simulation_name='simulation_1'
                   , description=' '
                   , models_framework=None
                   # 'sklearn', 'catboost', 'keras', 'xgboost'. We can set framework manually. If not function will try to identify it automatically
                   # CV parameters
                   , cv_n=5  # number of folds in CV. If 'None' no KFolding is performed
                   , cv_save_pred=True
                   # Holdout parameters
                   , hold_n=None  # number of holdoud sampling. If 'None' no sampling
                   , hold_train_sample_size=0.1
                   , hold_test_sample_size=0.1  # fraction of observation drawn for each sample
                   , hold_save_pred=True
                   # Full parameters
                   , full=True
                   , full_save_pred=True
                   , full_importance_n=[15, 6, 0.1]  # [n_estimators, max_dept, learning_rate]
                   # by col
                   , by_var=None  # list with one column
                   , by_var_save_pred=True
                   # Feature importance Permutation (FIP)
                   , fip_n=None  # number of tries in future importatnce
                   , fip_sample_size=0.5  # fraction of observation drawn for each sample
                   , fip_type='remove'  # remove, permutate_train
                   , model_fip=RandomForestRegressor(n_estimators=30, max_depth=7)
                   # for fip you can provide only one model
                   , fip_save_prob=True
                   , fip_save_class=True
                   , fip_save_confussion_matrix=True
                   # other options
                   , data_check=False
                   , correlation=False
                   , loops_progress=False
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
            self.data_check(data=train_set, set_name='train', y_var=y_var, x_var=x_var, x_var_cat=x_var_cat)
            if test_set is not None:
                self.data_check(data=test_set, set_name='train', y_var=y_var, x_var=x_var, x_var_cat=x_var_cat)
            else:
                print('LUCAS: Warning: test data set not provided. Calculations will be carried out without it')

        # if 'test' and 'train' have the same columns structure
        if test_set is not None:
            if any(train_set.columns != test_set.columns):
                raise ValueError('LUCAS: x_train has different columns than x_test')

        # resetting indicies to be sure that duplicates are ramoved and converting target variable to str
        # train_set = train_set.sample(frac = 1)




        # rebuilding sets if we have 'by_var'
        if by_var is not None:
            # separating by_var column
            train_set_by = pd.DataFrame({by_var: train_set[by_var]})
            test_set_by = pd.DataFrame({by_var: test_set[by_var]})
            train_set = train_set.drop(columns=[by_var])
            test_set = test_set.drop(columns=[by_var])
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


        # checking if the simulation name exist in current list of simulations
        if self.simulation_metainfo.shape != (0, 0):
            if simulation_name in list(self.simulation_metainfo.simulation_name):
                raise Exception(
                    'LUCAS: Simultion "' + simulation_name + '" already exists. Change name of your new simulation or remove existing simulation "' + simulation_name + '"')
        time_start = str(datetime.datetime.now().replace(microsecond=0))

        # determining categories for dependent variable
        if y_labels is None:
            y_labels = list(y_train.drop_duplicates())
            y_labels.sort()
            self.y_labels = y_labels
        else:
            self.y_labels = y_labels

        # names and positions of categorical variables (positions are used for exemple in catboost to give columns indicies), and names of numerical variables
        if x_var_cat is not None:
            self.cat_vars = x_var_cat
            self.cat_vars_indicies = [x_train.columns.get_loc(c) for c in x_var_cat if c in x_train]
            # num_vars = list(set(x_var) -  set(cat_vars))
            num_vars = x_var_num
            if len(num_vars) == 0:
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
        x_test_b = copy.deepcopy(x_test)
        y_train_b = copy.deepcopy(y_train)
        y_test_b = copy.deepcopy(y_test)

        # PART 2: CV - cross validation
        if cv_n is not None:

            sample_type = 'cv'

            # getting indicies for split od data set
            kf = skl.model_selection.KFold(n_splits=cv_n)
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
                for train_index_i, valid_index_i in kf.split(x_train):

                    test_index_i = x_test.index

                    if loops_progress: print(f)

                    # building sets based on split indicies
                    x_train_i, x_valid_i = x_train.iloc[train_index_i], x_train.iloc[valid_index_i]
                    y_train_i, y_valid_i = y_train.iloc[train_index_i], y_train.iloc[valid_index_i]

                    # information about simulation
                    simulation_info = {'simulation_name': simulation_name, 'model_name': [model_j_name],
                                       'sample_type': [sample_type], 'sample_nr': [f]}

                    # categorical variables encoding (optional)
                    print(y_train_i.dtype)
                    if use_cat_enc and model_framework not in ['catboost']:
                        x_train_i, x_valid_i, x_test = self.cat_encoding(x_train=x_train_i, y_train=y_train_i,
                                                                         x_valid=x_valid_i, x_test=x_test,
                                                                         cat_vars=x_var_cat,
                                                                         cat_encoding_method=cat_encoding_method)


                    # model fitting
                    if models_fit_parameters is not None:
                        model_j.fit(X=x_train_i, y=y_train_i, **models_fit_parameters)
                    else:
                        if model_framework in ['sklearn']:
                            model_j.fit(X=x_train_i, y=y_train_i)
                        elif model_framework in ['catboost']:
                            model_j.fit(X=x_train_i, y=y_train_i, plot=False, cat_features=self.cat_vars_indicies,
                                        use_best_model=False, early_stopping_rounds=75, logging_level='Silent')



                    # prediction calculation and saving

                    if cv_save_pred:
                        pred_train = pd.DataFrame(model_j.predict(x_train_i),
                                                  columns=['y_pred'], index=x_train_i.index)
                        # print(prob_train.shape)
                        self.prediction_determining(y_true=y_train_i
                                                       ,pred=pred_train, simulation_name=simulation_name,
                                                       model_name=model_j_name, sample_type=sample_type, sample_nr=f,
                                                       sample_name='_', set_type='train')

                        pred_valid = pd.DataFrame(model_j.predict(x_valid_i),
                                                  columns=['y_pred'], index=x_valid_i.index)
                        # print(pred_valid.shape)
                        self.prediction_determining(y_true=y_valid_i
                                                       , pred=pred_valid, simulation_name=simulation_name,
                                                       model_name=model_j_name, sample_type=sample_type, sample_nr=f,
                                                       sample_name='_', set_type='valid')
                        if y_test is not None:
                            pred_test = pd.DataFrame(model_j.predict(x_test),
                                                     columns=['y_pred'], index=x_test.index)
                            # print(prob_test.shape)
                            self.prediction_determining(y_true=y_test
                                                           ,pred=pred_test, simulation_name=simulation_name,
                                                           model_name=model_j_name, sample_type=sample_type,
                                                           sample_nr=f, sample_name='_', set_type='test')

                    f = f + 1

        # PART 3: HOLD - holdout
        if hold_n is not None:

            sample_type = 'hold'

            # getting indicies for split od data set
            smp = skl.model_selection.ShuffleSplit(n_splits=hold_n, train_size=hold_train_sample_size,
                                                   test_size=hold_test_sample_size, random_state=None)
            smp.get_n_splits(x_train_b)

            train_index_all = {}
            valid_index_all = {}

            k = 0
            for train_index_i, valid_index_i in smp.split(x_train):
                train_index_all[k] = train_index_i
                valid_index_all[k] = valid_index_i
                k = k + 1

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
                x_test = copy.deepcopy(x_test_b)
                y_train = copy.deepcopy(y_train_b)
                y_test = copy.deepcopy(y_test_b)

                f = 0
                # loop over sampled sets
                for i in train_index_all.keys():

                    # indicies for i-th loop
                    train_index_i = train_index_all[i]
                    valid_index_i = valid_index_all[i]
                    test_index_i = x_test.index

                    # building sets based on split indicies
                    x_train_i, x_valid_i = x_train.iloc[train_index_i], x_train.iloc[valid_index_i]
                    y_train_i, y_valid_i = y_train.iloc[train_index_i], y_train.iloc[valid_index_i]

                    # information about simulation
                    simulation_info = {'simulation_name': simulation_name, 'model_name': [model_j_name],
                                       'sample_type': [sample_type], 'sample_nr': [f]}

                    # categorical variables encoding
                    if use_cat_enc and model_framework not in ['catboost']:
                        x_train_i, x_valid_i, x_test = self.cat_encoding(x_train=x_train_i, y_train=y_train_i,
                                                                         x_valid=x_valid_i, x_test=x_test_b,
                                                                         cat_vars=x_var_cat,
                                                                         cat_encoding_method=cat_encoding_method)



                    # model fitting
                    if models_fit_parameters is not None:
                        model_j.fit(X=x_train_i, y=y_train_i, **models_fit_parameters)
                    else:
                        if model_framework in ['sklearn']:
                            model_j.fit(X=x_train_i, y=y_train_i)
                        elif model_framework in ['catboost']:
                            model_j.fit(X=x_train_i, y=y_train_i, plot=False, cat_features=self.cat_vars_indicies,
                                        use_best_model=True, early_stopping_rounds=75, logging_level='Silent')



                    # prediction calculation and saving
                    if hold_save_pred:

                        pred_train = pd.DataFrame(model_j.predict(x_train_i),
                                                  columns=['y_pred'], index=x_train_i.index)
                        # print(pred_train.shape)
                        self.prediction_determining(y_true=y_train_i
                                                       , pred=pred_train, simulation_name=simulation_name,
                                                       model_name=model_j_name, sample_type=sample_type, sample_nr=f,
                                                       sample_name='_', set_type='train')

                        pred_valid = pd.DataFrame(model_j.predict(x_valid_i),
                                                  columns=['y_pred'], index=x_valid_i.index)
                        # print(pred_valid.shape)
                        self.prediction_determining(y_true=y_valid_i
                                                       , pred=pred_valid, simulation_name=simulation_name,
                                                       model_name=model_j_name, sample_type=sample_type, sample_nr=f,
                                                       sample_name='_', set_type='valid')
                        if y_test is not None:
                            pred_test = pd.DataFrame(model_j.predict(x_test),
                                                     columns=['y_pred'], index=x_test.index)
                            # print(pred_test.shape)
                            self.prediction_determining(y_true=y_test
                                                           , pred=pred_test, simulation_name=simulation_name,
                                                           model_name=model_j_name, sample_type=sample_type,
                                                           sample_nr=f, sample_name='_', set_type='test')


                    f = f + 1

        # PART 4: BY_VAR - grouping by columns (tutaj x_test jest baza do robienie x_valid_i !!! sam x_test tez jest ale pod nazwa x_test_i bo jest problem ze zmienna by_var)
        if by_var is not None:

            sample_type = 'by_var'

            by_unique_train = train_set_by.drop_duplicates().sort_values(by_var).values[:, 0].tolist()
            by_unique_test = test_set_by.drop_duplicates().sort_values(by_var).values[:, 0].tolist()

            if by_unique_train != by_unique_test:
                raise ValueError('LUCAS: categories for "by_var" in train and test set are different')
            else:
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
                x_test = copy.deepcopy(x_test_b)

                x_train = pd.concat([x_train, train_set_by], axis=1)
                x_test = pd.concat([x_test, test_set_by], axis=1)

                y_train = copy.deepcopy(y_train_b)
                y_test = copy.deepcopy(y_test_b)

                # loop over sampled sets
                for f, i in enumerate(by_unique):
                    x_train_i = x_train.loc[x_train[by_var] == i]
                    x_valid_i = x_test.loc[x_test[by_var] == i]  # valid jest czescia test-u
                    y_train_i = pd.merge(y_train, x_train_i, left_index=True, right_index=True)[y_var]
                    y_valid_i = pd.merge(y_test, x_valid_i, left_index=True, right_index=True)[y_var]

                    # indicies fot i-th loop
                    train_index_i = x_train_i.index
                    valid_index_i = x_valid_i.index
                    test_index_i = x_test.index

                    x_test_i = copy.deepcopy(x_test)
                    y_test_i = copy.deepcopy(y_test)

                    x_train_i = x_train_i.drop(columns=by_var)
                    x_valid_i = x_valid_i.drop(columns=by_var)
                    x_test_i = x_test_i.drop(columns=by_var)

                    # information about simulation
                    simulation_info = {'simulation_name': simulation_name, 'model_name': [model_j_name],
                                       'sample_type': [sample_type], 'sample_nr': [f], 'sample_name': [i]}

                    # categorical variables encoding

                    if use_cat_enc and model_framework not in ['catboost']:
                        # UWAGA: x_test = x_test_i
                        x_train_i, x_valid_i, x_test_i = self.cat_encoding(x_train=x_train_i, y_train=y_train_i,
                                                                           x_valid=x_valid_i, x_test=x_test_i
                                                                           , cat_vars=x_var_cat,
                                                                           cat_encoding_method=cat_encoding_method)



                    # model fitting
                    if models_fit_parameters is not None:
                        model_j.fit(X=x_train_i, y=y_train_i, **models_fit_parameters)
                    else:
                        if model_framework in ['sklearn']:
                            model_j.fit(X=x_train_i, y=y_train_i)
                        elif model_framework in ['catboost']:
                            model_j.fit(X=x_train_i, y=y_train_i, plot=False, cat_features=self.cat_vars_indicies,
                                        use_best_model=True, early_stopping_rounds=75, logging_level='Silent')



                    # prediction calculation and saving
                    if by_var_save_pred:

                        pred_train = pd.DataFrame(model_j.predict(x_train_i),
                                                  columns=['y_pred'], index=x_train_i.index)
                        # print(pred_train.shape)
                        self.prediction_determining(y_true=y_train_i
                                                       , pred=pred_train, simulation_name=simulation_name,
                                                       model_name=model_j_name, sample_type=sample_type, sample_nr=f,
                                                       sample_name=str(i), set_type='train')



                        pred_valid = pd.DataFrame(model_j.predict(x_valid_i),
                                                  columns=['y_pred'], index=x_valid_i.index)
                        # print(pred_valid.shape)
                        self.prediction_determining(y_true=y_valid_i
                                                       , pred=pred_valid, simulation_name=simulation_name,
                                                       model_name=model_j_name, sample_type=sample_type, sample_nr=f,
                                                       sample_name=str(i), set_type='valid')

                        if y_test is not None:
                            pred_test = pd.DataFrame(model_j.predict(x_test_i),
                                                     columns=['y_pred'], index=x_test_i.index)
                            # print(pred_test.shape)
                            self.prediction_determining(y_true=y_test
                                                           , pred=pred_test, simulation_name=simulation_name,
                                                           model_name=model_j_name, sample_type=sample_type,
                                                           sample_nr=f, sample_name=str(i), set_type='test')



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
                simulation_info = {'simulation_name': simulation_name, 'model_name': [model_j_name],
                                   'sample_type': [sample_type],
                                   'sample_nr': [f]}  # sample_nr='0' na sztywno bo dla FULL jest tylko jedna iteracja

                # rewritting data with raw set
                x_train = copy.deepcopy(x_train_b)
                x_test = copy.deepcopy(x_test_b)
                y_train = copy.deepcopy(y_train_b)
                y_test = copy.deepcopy(y_test_b)

                # indicies
                train_index_i = x_train.index
                valid_index_i = None
                test_index_i = x_test.index

                # cat encoding (optional)
                if use_cat_enc and model_framework not in ['catboost']:
                    x_train, x_test, x_test = self.cat_encoding(x_train=x_train, y_train=y_train, x_valid=None,
                                                                x_test=x_test
                                                                , cat_vars=x_var_cat,
                                                                cat_encoding_method=cat_encoding_method)



                # model fitting
                if models_fit_parameters is not None:
                    model_j.fit(X=x_train, y=y_train, **models_fit_parameters)
                else:
                    if model_framework in ['sklearn']:
                        model_j.fit(X=x_train, y=y_train)
                    elif model_framework in ['catboost']:
                        model_j.fit(X=x_train, y=y_train, plot=False, cat_features=self.cat_vars_indicies,
                                    use_best_model=True, early_stopping_rounds=75, logging_level='Silent')



                # prediction calculation and saving
                if full_save_pred:

                    pred_train = pd.DataFrame(model_j.predict(x_train),
                                              columns=['y_pred'], index=x_train.index)
                    self.prediction_determining(y_true=y_train
                                                   , pred=pred_train, simulation_name=simulation_name,
                                                   model_name=model_j_name, sample_type=sample_type, sample_nr=f,
                                                   sample_name='_', set_type='train')

                    if y_test is not None:
                        pred_test = pd.DataFrame(model_j.predict(x_test),
                                                 columns=['y_pred'], index=x_test.index)
                        self.prediction_determining(y_true=y_test
                                                       , pred=pred_test, simulation_name=simulation_name,
                                                       model_name=model_j_name, sample_type=sample_type, sample_nr=f,
                                                       sample_name='_', set_type='test')



            # feature importance calculations
            if full_importance_n is not None:

                if loops_progress: print('feature_importance')

                # rewritting data with raw set
                x_train = copy.deepcopy(x_train_b)
                x_test = copy.deepcopy(x_test_b)
                y_train = copy.deepcopy(y_train_b)
                y_test = copy.deepcopy(y_test_b)

                # adding noise variables
                x_train['NOISE_uniform'] = np.random.uniform(-10, 10, len(x_train))
                x_train['NOISE_normal'] = np.random.normal(0, 10, len(x_train))

                if x_test is not None:
                    x_test['NOISE_uniform'] = np.random.uniform(-10, 10, len(x_test))
                    x_test['NOISE_normal'] = np.random.normal(0, 10, len(x_test))



                # CATBOOST (cat boost currently does not suppor multi-classification)
                feature_importance_cat = self.feature_importance_class_CB(iterations=full_importance_n[0],
                                                                          learning_rate=full_importance_n[2],
                                                                          depth=full_importance_n[1],
                                                                          cat_features=x_var_cat
                                                                          , x_train=x_train, y_train=y_train,
                                                                          x_test=x_test, y_test=y_test)

                feature_importance_cat = feature_importance_cat.rename(columns={'Importances': 'Importance',
                                                                                'Feature Id': 'Feature'})  # for structer compatibility with results from randomforest feature importatnce
                feature_importance_cat['method'] = 'catboost'

                # RANDOM FOREST

                # cat encoding (optional)
                if use_cat_enc:
                    x_train, x_test, x_test = self.cat_encoding(x_train=x_train, y_train=y_train, x_valid=None,
                                                                x_test=x_test, cat_vars=x_var_cat,
                                                                cat_encoding_method=cat_encoding_method)

                feature_importance_rt = self.feature_importance_class_RF(n_estimators=full_importance_n[0],
                                                                         max_depth=full_importance_n[1]
                                                                         , x_train=x_train, y_train=y_train,
                                                                         x_test=x_test, y_test=y_test)
                feature_importance_rt = feature_importance_rt.reset_index(drop=False)


                feature_importance_enet = self.feature_importance_elastic_net(l1_ratio=[0.1, 0.5, 0.9], x_train=x_train, y_train=y_train)
                feature_importance_enet['method'] = 'elastic_net'


                # gathering all results
                feature_importance = pd.concat([feature_importance_rt, feature_importance_cat], axis=0)
                feature_importance['simulation_name'] = simulation_name
                feature_importance_enet['simulation_name'] = simulation_name

                # columns reordering
                feature_importance = feature_importance[['simulation_name', 'method', 'Feature', 'Importance']]

                # saving results
                self.feature_importance = pd.concat([self.feature_importance, feature_importance])
                self.feature_importance_enet = pd.concat([self.feature_importance_enet, feature_importance_enet])

        # PART 6: FIP- feature importance permutation
        if fip_n is not None:

            sample_type = 'fip'

            # loop over samples # not implemented
            for f in range(fip_n):
                if loops_progress: print('fip ' + str(f))

                # rewritting data with raw set
                x_train = copy.deepcopy(x_train_b)
                x_test = copy.deepcopy(x_test_b)
                y_train = copy.deepcopy(y_train_b)
                y_test = copy.deepcopy(y_test_b)

                # split zbioru na testowy i uczący
                x_train_fip, x_valid_fip, y_train_fip, y_valid_fip = skl.model_selection.train_test_split(x_train,
                                                                                                          y_train,
                                                                                                          test_size=fip_sample_size)

                # indicies
                train_index_i = x_train_fip.index
                valid_index_i = x_valid_fip.index
                test_index_i = x_test.index

                # loop over feature variables  (instead of models loop !!!)
                for column in x_train.columns:
                    # model and it's features
                    model_fip_name = column
                    model_framework = self.model_framework_identify(model_fip)

                    # simulation information
                    simulation_info = {'simulation_name': simulation_name, 'model_name': [model_fip_name],
                                       'sample_type': [sample_type], 'sample_nr': [f]}


                    if fip_type == 'permutate_train':
                        # variable permutation
                        x_train_fip_perm = copy.deepcopy(x_train_fip)
                        x_valid_fip_perm = copy.deepcopy(x_valid_fip)
                        x_train_fip_perm[column] = np.random.permutation(x_train_fip_perm[column])
                        # x_valid_fip_perm[column] =  np.random.permutation(x_valid_fip_perm[column])
                        if y_test is not None:
                            x_test_perm = copy.deepcopy(x_test)
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
                        cat_vars_indicies = [x_train_fip_perm.columns.get_loc(c) for c in x_var_cat if
                                             c in x_train_fip_perm]

                    # categorical variables encoding (optional)
                    if use_cat_enc and model_framework not in ['catboost']:
                        x_train_fip_perm, x_valid_fip_perm, x_test_perm = self.cat_encoding(x_train=x_train_fip_perm,
                                                                                            y_train=y_train_fip,
                                                                                            x_valid=x_valid_fip_perm,
                                                                                            x_test=x_test_perm,
                                                                                            cat_vars=x_var_cat,
                                                                                            cat_encoding_method=cat_encoding_method)

                    # model fitting
                    if models_fit_parameters is not None:
                        model_fip.fit(X=x_train_fip_perm, y=y_train_fip, **models_fit_parameters)
                    else:
                        if model_framework in ['sklearn']:
                            model_fip.fit(X=x_train_fip_perm, y=y_train_fip)
                        elif model_framework in ['catboost']:
                            model_fip.fit(X=x_train_fip_perm, y=y_train_fip, plot=False,
                                          cat_features=self.cat_vars_indicies, use_best_model=True,
                                          early_stopping_rounds=75, logging_level='Silent')



                    # probabilities prediction calculation and saving
                    if fip_save_prob:

                        # UWAGA: aktualnie permutuje tylko x_train
                        # UWAGA przy przeklejaniu : tutaj zbiory maja w nazwie 'fip'

                        pred_train = pd.DataFrame(model_fip.predict(x_train_fip_perm),
                                                  columns=['y_pred'],
                                                  index=x_train_fip_perm.index)
                        self.prediction_determining(y_true=y_train_fip
                                                       , pred=pred_train, simulation_name=simulation_name,
                                                       model_name=model_fip_name, sample_type=sample_type, sample_nr=f,
                                                       sample_name=column, set_type='train')

                        pred_valid = pd.DataFrame(model_fip.predict(x_valid_fip_perm),
                                                  columns=['y_pred'], index=x_valid_fip.index)
                        self.prediction_determining(y_true=y_valid_fip
                                                       , pred=pred_valid, simulation_name=simulation_name,
                                                       model_name=model_fip_name, sample_type=sample_type, sample_nr=f,
                                                       sample_name=column, set_type='valid')

                        if y_test is not None:
                            pred_test = pd.DataFrame(model_fip.predict(x_test_perm),
                                                     columns=['y_pred'],
                                                     index=x_test_perm.index)
                            self.prediction_determining(y_true=y_test
                                                           , pred=pred_test, simulation_name=simulation_name,
                                                           model_name=model_fip_name, sample_type=sample_type,
                                                           sample_nr=f, sample_name=column, set_type='test')



        # PART 7: CORRELATIONS determining (target variable not included)

        if correlation:

            x_train_correlation = copy.deepcopy(x_train_b)
            y_train_correlation = copy.deepcopy(y_train_b)

            # zmiennej bez 'self' uzywam opcjonalnie jezeli nie chce sie odwolywac do globalnych ustawien
            num_vars = self.num_vars
            cat_vars = self.cat_vars

            if use_cat_enc and len(self.cat_vars) > 0:
                encode = ce.TargetEncoder(cols=self.cat_vars)
                encode.fit(X=x_train_correlation[self.cat_vars], y=y_train_correlation)  #
                x_train_correlation[self.cat_vars] = encode.transform(X=x_train_correlation[self.cat_vars],
                                                                      y=y_train_correlation)
                num_vars = num_vars + cat_vars

            # VIF
            if len(num_vars) > 0:
                vif = pd.DataFrame()
                vif['feature'] = num_vars
                vif['VIF'] = [variance_inflation_factor(x_train_correlation[num_vars].values, i) for i in
                              range(len(num_vars))]
                self.vif[simulation_name] = vif

            # correlation for numeric variables
            if self.num_vars is not None:
                correlation_numeric_pearson = self.correlation_pearson(data=x_train_correlation, vars=num_vars,
                                                                       method='pearson', round=2)
                correlation_numeric_kendall = self.correlation_pearson(data=x_train_correlation, vars=num_vars,
                                                                       method='kendall', round=2)

                self.correlation_numeric_pearson[simulation_name] = correlation_numeric_pearson
                self.correlation_numeric_kendall[simulation_name] = correlation_numeric_kendall

            # correlation for categorical variables
            if self.cat_vars is not None:
                correlation_categorical_v_cramer = self.correlation_cramer_v_matrix(data=x_train_correlation,
                                                                                    vars=cat_vars, round=2)

                self.correlation_categorical_v_cramer[simulation_name] = correlation_categorical_v_cramer

            # correlation for mix of categorical and numerical variables
            if self.cat_vars is not None and self.num_vars is not None:
                correlation_mix_ratio = self.correlation_ratio_matrix(data=x_train_correlation.reset_index(drop=True),
                                                                      vars_cat=cat_vars, vars_num=self.num_vars,
                                                                      round=2)  # tutaj zostawiam self.num_vars pierwotny

                self.correlation_mix_ratio[simulation_name] = correlation_mix_ratio

        # PART 8: METADATA about simulation
        time_end = str(datetime.datetime.now().replace(microsecond=0))
        num_vars_str = 'None' if self.num_vars is None else ', '.join(self.num_vars)  # names of numeric features
        cat_vars_str = 'None' if self.cat_vars is None else ', '.join(self.cat_vars)  # names of categorical features
        x_var_n = len(x_var)  # number of feature variables
        simulation_metainfo_new = pd.DataFrame({'simulation_name': [simulation_name]
                                                   , 'description': [description]
                                                   , 'x_n': x_var_n
                                                   , 'y_name': y_var
                                                   , 'x_num': [num_vars_str]
                                                   , 'x_cat': [cat_vars_str]
                                                   , 'cat_encoding': cat_encoding_method
                                                   , 'simulation_start': [time_start]
                                                   , 'simulation_end': [time_end], })
        # empty slots for performance metadata
        simulation_metainfo_new['performance start'] = np.nan
        simulation_metainfo_new['performance end'] = np.nan

        # final save of metadata
        self.simulation_metainfo = pd.concat([self.simulation_metainfo, simulation_metainfo_new])
        self.models[simulation_name] = models_list




    def plot_qq(self, data_new = None
                                    , x_var    = None
                                    , filter   = None
                                    , fill_var = None
                                    , facet    = None
                                    , x_lim    = [np.nan, np.nan]
                                    , alpha    = 0.5
                                    , size     = 2
                                    , frac     = 1
                                    , title    = ''
                                    , fig_w    = 15
                                    , fig_h    = 5):
        """

        """
        # plot size
        plotnine.options.figure_size = (fig_w, fig_h)

        # choosing data set for plot
        if data_new is not None:
            data = data_new
        else:
            data = self.prediction

        # filtering data
        if filter is not None:
            for i in filter.keys():
                data = data.loc[data[i].isin(filter[i]), :]

        # variable used for 'fill 'converted into string (plotnine requirement)
        if fill_var is not None:
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
            data = data.sample(n=50000)

        if len(data) == 0:
            print('LUCAS: after filtering no data to display. Suggestion is to check "set_type" if exists')
        else:
            if fill_var is not None:
                return (((ggplot(data=data) + geom_qq(aes(sample=x_var, colour=fill_var, fill=fill_var, group=fill_var )) + facet) +
                        facet + xlim(x_lim) +
                        ggtitle(title)))
            else:
                return (((ggplot(data=data) + geom_qq(aes(sample=x_var )) + facet) +
                        facet + xlim(x_lim) +
                        ggtitle(title)))


    def plot_density(self, data_new = None
                                    , x_var    = None
                                    , filter   = None
                                    , fill_var = None
                                    , facet    = None
                                    , x_lim    = [np.nan, np.nan]
                                    , alpha    = 0.5
                                    , size     = 2
                                    , frac     = 1
                                    , title    = ''
                                    , fig_w    = 15
                                    , fig_h    = 5):
        """

        """
        # plot size
        plotnine.options.figure_size = (fig_w, fig_h)

        # choosing data set for plot
        if data_new is not None:
            data = data_new
        else:
            data = self.prediction

        # filtering data
        if filter is not None:
            for i in filter.keys():
                data = data.loc[data[i].isin(filter[i]), :]

        # variable used for 'fill 'converted into string (plotnine requirement)
        if fill_var is not None:
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
            data = data.sample(n=50000)

        if len(data) == 0:
            print('LUCAS: after filtering no data to display. Suggestion is to check "set_type" if exists')
        else:
            if fill_var is not None:
                return ((ggplot(data=data, mapping=aes(x='error_normalized', fill = fill_var)) + geom_density() +
                        facet + xlim(x_lim) +
                        ggtitle(title)))
            else:
                return ((ggplot(data=data, mapping=aes(x='error_normalized')) + geom_density() +
                        facet +
                        xlim(x_lim) +
                        ggtitle(title)))






    def plot_prediction_scatterplot(  self
                                    , data_new = None
                                    , x_var    = None
                                    , y_var    = None
                                    , filter   = None
                                    , fill_var = None
                                    , facet    = None
                                    , x_lim    = [np.nan, np.nan]
                                    , alpha    = 0.5
                                    , size     = 2
                                    , frac     = 1
                                    , title    = ''
                                    , fig_w    = 15
                                    , fig_h    = 5):
        """

        """
        # plot size
        plotnine.options.figure_size = (fig_w, fig_h)

        # choosing data set for plot
        if data_new is not None:
            data = data_new
        else:
            data = self.prediction

        # filtering data
        if filter is not None:
            for i in filter.keys():
                data = data.loc[data[i].isin(filter[i]), :]

        # variable used for 'fill 'converted into string (plotnine requirement)
        if fill_var is not None:
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
            data = data.sample(n=50000)

        if len(data) == 0:
            print('LUCAS: after filtering no data to display. Suggestion is to check "set_type" if exists')
        else:
            if fill_var is not None:
                return (ggplot(data=data, mapping=aes(x=x_var, y=y_var, colour=fill_var, fill=fill_var)) +
                        geom_point(size=size, alpha=alpha) +
                        geom_smooth(method='lm', na_rm=True, inherit_aes=True, show_legend=None, raster=False,
                                    legend_fill_ratio=0.5) +
                        facet + xlim(x_lim) +
                        ggtitle(title))
            else:
                return (ggplot(data=data, mapping=aes(x=x_var, y=y_var)) +
                        geom_point(size=size, alpha=alpha) +
                        geom_smooth(method='lm', na_rm=True, inherit_aes=True, show_legend=None, raster=False,
                                    legend_fill_ratio=0.5) +
                        facet +
                        xlim(x_lim) +
                        ggtitle(title))



    def plot_scores_bar_full_fip(self
                                 , score
                                 , simulation_name
                                 , set_type='test'):
        """
        Bar plot with scores for 'full' and 'fip' sample types
        """

        scores = self.scores
        scores['score_value'] = np.round(scores['score_value'], 3)
        scores.sort_values(by=['score_value'])

        if len(scores) > 0:
            return (ggplot(data=scores.loc[
                                (scores['simulation_name'] == simulation_name) & (scores['set_type'] == set_type) & (
                                            scores['if_automatic'] == 0) & (
                                            (scores['sample_type'] == 'full') | (scores['sample_type'] == 'fip')) & (
                                            scores['score_name'] == score), :]) +
                    geom_bar(aes(x='model_name', y='score_value', fill='sample_nr'), stat='identity') +
                    geom_label(aes(x='model_name', y='score_value', label='score_value', fill='sample_nr')) +
                    ylim([0, 1]) +
                    coord_flip())


    def scores_volatility(self, simulation_name, sample_type, set_type):
        scores = self.scores

        scores_filtered = scores.loc[(scores['simulation_name'] == simulation_name) & (
                    scores['sample_type'] == sample_type) & (scores['set_type'].isin(set_type))]

        scores_group = scores_filtered.groupby(
            ['model_name', 'set_type', 'sample_type', 'score_name']).agg(
            max=pd.NamedAgg(column='score_value', aggfunc='max'),
            min=pd.NamedAgg(column='score_value', aggfunc=lambda x: min(x)),
            mean=pd.NamedAgg(column='score_value', aggfunc=lambda x: np.nanmean(x)),
            std=pd.NamedAgg(column='score_value', aggfunc=lambda x: np.nanstd(x))
        )

        scores_group = scores_group.reset_index()
        return (scores_group)

    def plot_scores(self
                    , data_new=None
                    , filter={'sample_type': ['cv'], 'set_type': ['test']}
                    , facet='score_name~simulation_name'
                    , fill_var='model_name'
                    , x_var='sample_nr'
                    , y_var='score_value'
                    , fig_w=15
                    , fig_h=10):
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
                data = data.loc[data[i].isin(filter[i]), :]

        # converting into string variable for fill (plotnine requrement)
        if fill_var != np.nan:
            data[fill_var] = data[fill_var].astype(str)

        # faceting of plot
        if facet is None:
            facet = facet_null()
        else:
            facet = facet_grid(facets=facet, scales='free')
        print(facet)
        if len(data) == 0:
            # raise ValueError('LUCAS: after filtering no data to display. Suggestion is to check "set_type" if exists')
            print('LUCAS: after filtering no data to display. Suggestion is to check "set_type" if exists')
        else:
            return ((ggplot(data=data) +
                     geom_point(aes(x=x_var, y=y_var, color=fill_var)) +
                     geom_line(aes(x=x_var, y=y_var, color=fill_var)) +
                     facet) + theme(strip_text_y = element_text(angle = 0,              # change facet text angle
                                        ha = 'left'             # change text alignment
                                       ),  strip_background_y = element_text(color = '#969dff' # change background colour of facet background
                                              , width = 0.2     # adjust width of facet background to fit facet text
                                             ) ) )




    def print_simulation_metainfo(simulation_name='s1'):

        """
        Printing metainformations
        """

        sim = self.simulation_metainfo.loc[self.simulation_metainfo['simulation_name'] == simulation_name, :]
        mod = self.models[simulation_name]
        print('SIMULATION BASIC INFO: \n')
        print(sim)
        print('\n')
        print('MODELS PARAMETERS: \n')
        for i in mod:
            print(i)
            print(mod[i])
            print('\n')

    def save_results(self, path, override=True):

        """
        currenlty only overriding is implemented
        """

        if override == True:
            import pickle
            data = {}


            data['prediction'] = self.prediction
            data['scores'] = self.scores

            data['simulation_metainfo'] = self.simulation_metainfo



            data['feature_importance'] = self.feature_importance
            data['feature_importance_enet'] = self.feature_importance_enet

            data['models'] = self.models

            data['correlation_numeric_pearson'] = self.correlation_numeric_pearson
            data['correlation_numeric_kendall'] = self.correlation_numeric_kendall
            data['correlation_mix_ratio'] = self.correlation_mix_ratio
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

        self.classification = data['classification']
        self.probabilities = data['probabilities']
        self.scores = data['scores']
        self.confusion_matrix = data['confusion_matrix']

        self.priori = data['priori']
        self.threshold = data['threshold']

        self.y_train_count = data['y_train_count']
        self.y_test_count = data['y_test_count']

        self.feature_importance = data['feature_importance']
        self.feature_importance_enet = data['feature_importance_enet']

        self.simulation_metainfo = data['simulation_metainfo']

        self.models = data['models']

        self.correlation_numeric_pearson = data['correlation_numeric_pearson']
        self.correlation_numeric_kendall = data['correlation_numeric_kendall']
        self.correlation_mix_ratio = data['correlation_mix_ratio']
        self.correlation_categorical_v_cramer = data['correlation_categorical_v_cramer']
        self.vif = data['vif']

    def stacking_between_sim(self, simulation_new_name='sim_new', simulation_params=[], method='mean'):
        """
        only for 'full'
        simulation_params = [[sim, model],[sim, model]]
        """

        # loop over simulations
        for i, sim_ in enumerate(simulation_params):

            sim = sim_[0]
            data_i = self.prediction.loc[
                     (self.prediction['sample_type'] == 'full') & (self.prediction['simulation_name'] == sim), :]
            models_i = sim_[1:]

            # loop over models in simulations
            for j, model in enumerate(models_i):
                if (i == 0) and (j == 0):  # pierwsza iteracja petli
                    data_identifiers = data_i.loc[
                        data_i['model_name'] == model, data_i.columns.difference(['y_pred'])]
                    data_identifiers = data_identifiers.reset_index(drop=True)
                    prob_array_3D = np.array(data_i.loc[data_i['model_name'] == model, ['y_pred'] ] )
                    prob_array_3D = np.expand_dims(prob_array_3D, 0)
                else:
                    prob_array_3D_i_j = data_i.loc[data_i['model_name'] == model, ['y_pred'] ]
                    prob_array_3D_i_j = np.expand_dims(prob_array_3D_i_j, 0)
                    prob_array_3D = np.append(prob_array_3D, prob_array_3D_i_j, axis=0)

        # probability average over dim=0
        prob_array_mean = np.apply_over_axes(func=lambda x, y: np.mean(x, y), a=prob_array_3D, axes=[0])

        # removing third dimension (after calculeting mean it is no more necessary)
        prob_array_mean = np.squeeze(prob_array_mean, axis=0)

        # tranform to DataFrame and add identifiers
        prob_mean = pd.DataFrame(prob_array_mean, columns=['y_pred'])
        prob_mean = pd.concat([data_identifiers, prob_mean], axis=1)

        # name of stacked model and simulation
        prob_mean['model_name'] = 'model_stack'
        prob_mean['simulation_name'] = simulation_new_name

        # saving results
        self.prediction = pd.concat(
            [self.prediction[self.prediction['simulation_name'] != simulation_new_name], prob_mean])

    def scores_agg_full(self, simulation_params=[]):

        """
        Calculating scores for aggregated result from different simulations (only 'full' sample_type).
        From each simulation only one model can be used.
        simulation_params = [['sim_1','cat'], ['sim_2','xgb']]
        """

        # getting prediction data
        data_f = self.prediction.loc[(self.prediction['sample_type'] == 'full'),:]
        data_f.drop(columns='index')  # nie potrzebujemy indeksu bo wyniki sa agregacja

        # filtering data by models and simulations
        data = pd.DataFrame()
        model_name_agg = ''
        simulation_name_agg = ''
        for params in simulation_params:
            data = pd.concat([data, data_f.loc[
                                    (data_f['simulation_name'] == params[0]) & (data_f['model_name'] == params[1]), :]], axis=0)
            simulation_name_agg = simulation_name_agg + '_' + params[0]
            model_name_agg = model_name_agg + '_' + params[1]
        # splitting data into train and test (simulations and models are mixed here as we want)
        train = data.loc[data['set_type'] == 'train', :]
        test = data.loc[data['set_type'] == 'test', :]

        # creating indicies columns for aggregated results
        group_structure = ['model_name', 'sample_nr', 'sample_name', 'sample_type', 'set_type',
                           'simulation_name']

        # model_name_agg = '_'.join(list(simulation_params.values()))
        # simulation_name_agg = '_'.join(list(simulation_params.keys()))

        # creating empty (with only indicies) data frames to collect aggregated results
        train_group = pd.DataFrame(
            [[model_name_agg, 0, '_', 'full', 'train', simulation_name_agg]],
            columns=group_structure)
        test_group = pd.DataFrame(
            [[model_name_agg, 0, '_', 'full', 'test', simulation_name_agg]],
            columns=group_structure)

        # scores
        scores_train = self.scores_determining(y_true=train['y_true'], y_pred=train['y_pred'])
        scores_test = self.scores_determining(y_true=test['y_true'], y_pred=test['y_pred'])

        # dodanie zmiennych grupujacych
        scores_train = pd.concat([train_group, scores_train], axis=1)
        scores_test = pd.concat([test_group, scores_test], axis=1)

        # wypelnienie zmiennych grupujacych
        scores_train[group_structure] = scores_train[group_structure].fillna(method='ffill')
        scores_test[group_structure] = scores_test[group_structure].fillna(method='ffill')


        # saving scores
        if len(self.scores) != 0:
            self.scores = pd.concat(
                [self.scores.loc[self.scores['simulation_name'] != simulation_name_agg, :], scores_train, scores_test])



    # FAST PLOTS

    def fast_correlation_matrices(self, simulation_name):

        """
        Printing set of tables with correlations between feature variables
        """

        return (self.sbs(
            [self.correlation_numeric_pearson.get(simulation_name, pd.DataFrame())
                , self.correlation_numeric_kendall.get(simulation_name, pd.DataFrame())
                , self.correlation_mix_ratio.get(simulation_name, pd.DataFrame())
                , self.correlation_categorical_v_cramer.get(simulation_name, pd.DataFrame())
                , self.vif.get(simulation_name, pd.DataFrame())], ['pearson', 'kendall', 'ratio', 'v_cramer', 'vif']))




    def fast_qq(self
                       , simulation_name=None
                       , x_var=None
                       , plot_density_sample_frac=0.5
                       , sample_type='full'
                       , facet='model_name~.'
                       , fig_w=15
                       , fig_h=5):
        """

        """

        # return without list !!!. If not, plot will be duplicated
        if type(sample_type) != list:
            sample_type = [sample_type]
        if sample_type[0] in list(self.prediction['sample_type'].drop_duplicates()):
            display(self.plot_qq( x_var    = x_var
                                                    , filter   = {'set_type' : ['train'], 'sample_type':sample_type, 'simulation_name':[simulation_name]}
                                                    , fill_var = None
                                                    , facet    = facet
                                                    , x_lim    = [np.nan, np.nan]
                                                    , alpha    = 0.5
                                                    , size     = 2
                                                    , frac     = plot_density_sample_frac
                                                    , title    = 'train'
                                                    , fig_w    = fig_w
                                                    , fig_h    = fig_h),
                    self.plot_qq(  x_var=x_var
                                                     , filter={'set_type' : ['test'], 'sample_type':sample_type, 'simulation_name':[simulation_name]}
                                                     , fill_var=None
                                                     , facet=facet
                                                     , x_lim=[np.nan, np.nan]
                                                     , alpha=0.5
                                                     , size=2
                                                     , frac=plot_density_sample_frac
                                                     , title='test'
                                                     , fig_w=fig_w
                                                     , fig_h=fig_h)
                    )




    def fast_density(self
                   , simulation_name=None
                   , x_var=None
                   , plot_density_sample_frac=0.5
                   , sample_type='full'
                   , facet='model_name~.'
                   , fig_w=15
                   , fig_h=5):
        """

        """

        # return without list !!!. If not, plot will be duplicated
        if type(sample_type) != list:
            sample_type = [sample_type]
        if sample_type[0] in list(self.prediction['sample_type'].drop_duplicates()):
            display(self.plot_density( x_var    = x_var
                                                    , filter   = {'set_type' : ['train'], 'sample_type':sample_type, 'simulation_name':[simulation_name]}
                                                    , fill_var = None
                                                    , facet    = facet
                                                    , x_lim    = [np.nan, np.nan]
                                                    , alpha    = 0.5
                                                    , size     = 2
                                                    , frac     = plot_density_sample_frac
                                                    , title    = 'train'
                                                    , fig_w    = fig_w
                                                    , fig_h    = fig_h),
                    self.plot_density(  x_var=x_var
                                                     , filter={'set_type' : ['test'], 'sample_type':sample_type, 'simulation_name':[simulation_name]}
                                                     , fill_var=None
                                                     , facet=facet
                                                     , x_lim=[np.nan, np.nan]
                                                     , alpha=0.5
                                                     , size=2
                                                     , frac=plot_density_sample_frac
                                                     , title='test'
                                                     , fig_w=fig_w
                                                     , fig_h=fig_h)
                    )



    def fast_scatterplot(self
                       , simulation_name=None
                       , x_var=None
                       , y_var=None
                       , plot_density_sample_frac=0.5
                       , sample_type='full'
                       , facet='model_name~.'
                       , fig_w=15
                       , fig_h=5):
        """

        """

        # return without list !!!. If not, plot will be duplicated
        if type(sample_type) != list:
            sample_type = [sample_type]
        if sample_type[0] in list(self.prediction['sample_type'].drop_duplicates()):
            display(self.plot_prediction_scatterplot( x_var    = x_var
                                                    , y_var    = y_var
                                                    , filter   = {'set_type' : ['train'], 'sample_type':sample_type, 'simulation_name':[simulation_name]}
                                                    , fill_var = None
                                                    , facet    = facet
                                                    , x_lim    = [np.nan, np.nan]
                                                    , alpha    = 0.5
                                                    , size     = 2
                                                    , frac     = plot_density_sample_frac
                                                    , title    = 'train'
                                                    , fig_w    = fig_w
                                                    , fig_h    = fig_h),
                    self.plot_prediction_scatterplot(  x_var=x_var
                                                     , y_var=y_var
                                                     , filter={'set_type' : ['test'], 'sample_type':sample_type, 'simulation_name':[simulation_name]}
                                                     , fill_var=None
                                                     , facet=facet
                                                     , x_lim=[np.nan, np.nan]
                                                     , alpha=0.5
                                                     , size=2
                                                     , frac=plot_density_sample_frac
                                                     , title='test'
                                                     , fig_w=fig_w
                                                     , fig_h=fig_h)
                    )







    def fast_scores_plot(self
                         , simulation_name=None
                         , score='balanced_accuracy'
                         , sample_type='cv'):

        """
        Plot line plot with scores for train and test test (sample_type can be selected). This is wrapper for method 'plot_scores'
        """

        display(
            self.plot_scores(filter={'sample_type': [sample_type], 'set_type': ['test'],
                                     'simulation_name': [simulation_name]}, x_var='sample_nr', y_var='score_value',
                             fill_var='model_name', facet='score_name~.'))

    def fast_conf_matrix_scores_flat_version(self, simulation_name, fig_w=10, fig_h=2.5):
        scores_matrix = self.scores[
            (self.scores['simulation_name'] == simulation_name) & (self.scores['sample_type'] == 'full') ].drop(
            columns={'simulation_name', 'sample_name', 'sample_nr', 'sample_type'})
        scores_matrix = scores_matrix.rename(columns={'score_value': '_'})

        scores_matrix = scores_matrix.set_index(['model_name', 'set_type', 'score_name'])
        scores_matrix = scores_matrix.unstack(2)

        # usuwanie indeksów wierszowych
        scores_matrix = scores_matrix.reset_index(drop=False)

        # laczenie indeksow kolumnowych
        scores_matrix.columns = [''.join(col).strip() for col in scores_matrix.columns.values]

        # poprawka nazw indeksów kolumnowych
        scores_matrix = scores_matrix.rename(columns={'model_name_': 'model_name', 'set_type_': 'set_type'})

        return (scores_matrix)

    def fast_scores(self, simulation_name):

        """
        Printing confusion matrix, scores for sample_type = 'full' but for all models and all cut_off/priori. Additionaly information about and threshold are printted.
        """


        # (1) feature_importance
        feature_importance = self.feature_importance.loc[self.feature_importance['simulation_name'] == simulation_name,:]
        feature_importance = feature_importance.reset_index(drop=True)  # index reset so function 'sbs' works properly

        feature_importance_rf = feature_importance.loc[
            feature_importance['method'] == 'Random_Forest_Importance', ['Feature', 'Importance']]
        feature_importance_rf_per = feature_importance.loc[
            feature_importance['method'] == 'RandomForest_permutation', ['Feature', 'Importance']]
        feature_importance_cat = feature_importance.loc[
            feature_importance['method'] == 'catboost', ['Feature', 'Importance']]

        feature_importance_enet = self.feature_importance_enet.loc[
                self.feature_importance_enet['simulation_name'] == simulation_name, ['Feature', 'Importance',
                                                                                     'Importance_abs', 'l1']]

        # (2) scores (scores na razie sa wyswietlana gdzie indziej wiec tutaj jest zakomentowane)
        # scores_names_list = self.scores.loc[
        #     self.scores['simulation_name'] == simulation_name, 'score_name'].drop_duplicates()
        #
        # scores_list = []
        #
        # for i in scores_names_list:
        #     scores_i = self.scores.loc[
        #         (self.scores['if_automatic'] == 0) & (self.scores['simulation_name'] == simulation_name) & (
        #                     self.scores['sample_type'] == 'full') & (self.scores['score_name'] == i), ['model_name',
        #                                                                                                'set_type',
        #                                                                                                'threshold_priori_id',
        #                                                                                                'score_value']]
        #
        #     scores_i = scores_i.reset_index(drop=True)  # index reset so function 'sbs' works properly
        #
        #     scores_list = scores_list + [scores_i]



        # displaying all elements (1-5)

        return ([self.sbs(
                [feature_importance_enet, feature_importance_rf, feature_importance_rf_per, feature_importance_cat],
                ['Feature Importance e-net', 'Feature Importance RF', 'Feature Permutation Importance RF',
                 'Feature Importance CATBOOST'])
            ,
                 'recall = TP / TP + FN;   precision = TP / (TP + FP);    f1 = ( 2*TP ) / (2*TP + FP + FN );   balanced_accuracy = (TP/(TP+FN) + TN/(TN+FP)) / 2 '])



    def summary_big(self, simulation_name, cv_n, hold_n, by_var, fip_n):
        display(self.h('meta informations about simulation'))
        display(self.simulation_metainfo.loc[self.simulation_metainfo['simulation_name'] == simulation_name])
        display(self.h('correlation between features'))
        display(self.fast_correlation_matrices(simulation_name=simulation_name))
        display(self.h('scores and feature importance'))
        display(self.fast_scores(simulation_name=simulation_name))
        display(self.h('scores'))
        display(self.fast_conf_matrix_scores_flat_version(simulation_name=simulation_name))
        display(self.h('scatterplots plots full'))
        self.fast_scatterplot(simulation_name=simulation_name, x_var='y_pred', y_var='y_true', plot_density_sample_frac=0.5, sample_type='full', facet='.~model_name', fig_w=15, fig_h=5)
        self.fast_scatterplot(simulation_name=simulation_name, x_var='y_pred', y_var='error',plot_density_sample_frac=0.5, sample_type='full', facet='.~model_name', fig_w=15, fig_h=5)
        if cv_n is not None:
            display(self.h('scatterplots plots cv'))
            self.fast_scatterplot(simulation_name=simulation_name, x_var='y_pred', y_var='error', plot_density_sample_frac=0.5, sample_type='cv', facet='sample_nr~model_name', fig_w=15, fig_h=5)
            self.fast_scatterplot(simulation_name=simulation_name, x_var='y_pred', y_var='y_true',
                                  plot_density_sample_frac=0.5, sample_type='cv', facet='sample_nr~model_name', fig_w=15,
                                  fig_h=5)
        if hold_n:
            display(self.h('scatterplots plots holdout'))
            self.fast_scatterplot(simulation_name=simulation_name, x_var='y_pred', y_var='error', plot_density_sample_frac=0.5, sample_type='hold', facet='sample_nr~model_name', fig_w=15, fig_h=10)
            self.fast_scatterplot(simulation_name=simulation_name, x_var='y_pred', y_var='y_true',
                                  plot_density_sample_frac=0.5, sample_type='hold', facet='sample_nr~model_name', fig_w=15,
                                  fig_h=5)

        if by_var:
            display(self.h('scatterplots plots by_var'))
            self.fast_scatterplot(simulation_name=simulation_name, x_var='y_pred', y_var='error', plot_density_sample_frac=0.5, sample_type='by_var', facet='sample_name~model_name', fig_w=15, fig_h=10)
            self.fast_scatterplot(simulation_name=simulation_name, x_var='y_pred', y_var='y_true',
                                  plot_density_sample_frac=0.5, sample_type='by_var', facet='sample_name~model_name', fig_w=15,
                                  fig_h=5)

        if fip_n is not None:
            display(self.h('scatterplots plots fip_n'))
            self.fast_scatterplot(simulation_name=simulation_name, x_var='y_pred', y_var='error', plot_density_sample_frac=0.5, sample_type=['full', 'fip'], facet='sample_nr~model_name', fig_w=15, fig_h=10)
            self.fast_scatterplot(simulation_name=simulation_name, x_var='y_pred', y_var='y_true',
                                  plot_density_sample_frac=0.5, sample_type='fip', facet='sample_nr~model_name', fig_w=15,
                                  fig_h=5)

        display(self.h('error density full'))
        self.fast_density(simulation_name=simulation_name, x_var='error_normalized',  plot_density_sample_frac=0.5, sample_type=['full'], facet='.~model_name', fig_w=15, fig_h=5)

        if cv_n is not None:
            display(self.h('error density cv'))
            self.fast_density(simulation_name=simulation_name, x_var='error_normalized', plot_density_sample_frac=0.5,
                          sample_type=['cv'], facet='sample_nr~model_name', fig_w=15, fig_h=10)
        if hold_n is not None:
            display(self.h('error density hold'))
            self.fast_density(simulation_name=simulation_name, x_var='error_normalized', plot_density_sample_frac=0.5,
                          sample_type=['hold'], facet='sample_nr~model_name', fig_w=15, fig_h=10)
        if by_var is not None:
            display(self.h('error density by_var'))
            self.fast_density(simulation_name=simulation_name, x_var='error_normalized', plot_density_sample_frac=0.5,
                          sample_type=['by_var'], facet='sample_name~model_name', fig_w=15, fig_h=10)
        if fip_n is not None:
            display(self.h('error density fip_n'))
            self.fast_density(simulation_name=simulation_name, x_var='error_normalized', plot_density_sample_frac=0.5,
                              sample_type=['fip'], facet='sample_nr~model_name', fig_w=15, fig_h=10)

        display(self.h('error qq full'))
        self.fast_qq(simulation_name=simulation_name, x_var='error_normalized', plot_density_sample_frac=0.5,
                              sample_type=['full'], facet='.~model_name', fig_w=15, fig_h=5)

        if cv_n is not None:
            display(self.h('error qq cv_n'))
            self.fast_qq(simulation_name=simulation_name, x_var='error_normalized', plot_density_sample_frac=0.5,
                         sample_type=['cv'], facet='sample_nr~model_name', fig_w=15, fig_h=10)
        if hold_n is not None:
            display(self.h('error qq hold'))
            self.fast_qq(simulation_name=simulation_name, x_var='error_normalized', plot_density_sample_frac=0.5,
                         sample_type=['hold'], facet='sample_nr~model_name', fig_w=15, fig_h=10)
        if by_var is not None:
            display(self.h('error qq by_var'))
            self.fast_qq(simulation_name=simulation_name, x_var='error_normalized', plot_density_sample_frac=0.5,
                         sample_type=['by_var'], facet='sample_name~model_name', fig_w=15, fig_h=10)
        if fip_n is not None:
            display(self.h('error qq fip_n'))
            self.fast_qq(simulation_name=simulation_name, x_var='error_normalized', plot_density_sample_frac=0.5,
                         sample_type=['fip'], facet='sample_nr~model_name', fig_w=15, fig_h=10)


        if cv_n is not None:
            display(self.h('scores by cross validation'))
            display(self.fast_scores_plot(simulation_name=simulation_name, sample_type='cv'))
            display(self.scores_volatility(simulation_name=simulation_name, sample_type='cv', set_type=['test']))
        if hold_n is not None:
            display(self.h('scores by holdout'))
            display(self.fast_scores_plot(simulation_name=simulation_name, sample_type='hold'))
            display(self.scores_volatility(simulation_name=simulation_name, sample_type='hold', set_type=['test']))
        if by_var is not None:
            display(self.h('scores by by_var'))
            display(self.fast_scores_plot(simulation_name=simulation_name, sample_type='by_var'))
            display(self.scores_volatility(simulation_name=simulation_name, sample_type='by_var', set_type=['test']))


    def h(self, text, size=3, bold=True, color='blue'):
        if not bold:
            return (HTML('<font size = "' + str(size) + '" color= "' + color + '" >' + text + '</font>'))
        else:
            return (HTML('<font size = " ' + str(size) + '" color= "' + color + '" ><b>' + text + '</b></font>'))

    def sbs(self, dfs: list, captions: list):

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

















