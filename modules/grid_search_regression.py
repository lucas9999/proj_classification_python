import optuna
import pandas as pd
import numpy as np

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from plotnine.themes.themeable import axis_line
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB
# from xgboost                   import XGBRegressor
from lightgbm import LGBMRegressor
# from sklearn.svm import SVC
import sklearn.metrics as skm

import sklearn as skl
from plotnine import *
import plotnine

import optuna
from optuna import trial


class Objective(object):
    def __init__(self
                 , method = 'RF'
                 , opt_function = 'aps'
                 , calibration_method = None
                 , pos_label = None
                 , params    = None
                 , threshold = None
                 , priori    = None
                 , x_train   = None
                 , y_train   = None
                 , x_test    = None
                 , y_test    = None):
        # Hold this implementation specific arguments as the fields of the class.
        self.method = method
        self.opt_function = opt_function
        self.calibration_method = calibration_method
        self.params = params
        self.threshold = threshold
        self.priori = priori
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.scores = pd.DataFrame()
        self.y_labels = None
        self.pos_label = pos_label
        # self.x_var_num_range = x_var_num_range

    def __call__(self, trial):

        # optional variables sampling ( to implement in the future)

        # if x_var_num_range is not None:
        #   number_of_var = list(range(self.x_var_num_range[0], self.x_var_num_range[1]+1))
        #   x_var = np.random.choice(list(x_var.columns), number_of_var)
        # else:
        #   x_var = x_train.columns

        if self.params is not None:
            params = exec(self.params)
        else:
            params = None

        if self.method == 'SVR':  # support vectors machine

            SVR_params = {
                 'C': trial.suggest_loguniform('C', 0.000001, 100)
                ,'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'])
                ,'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']) }
            if params is not None:
                SVR_params.update(params)
            model = sklearn.svm.SVR(**SVR_params)



        elif self.method == 'RF':  # random forest

            # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.feature_importances_
            RF_params = {
                'n_estimators': int(trial.suggest_uniform('n_estimators', 20, 300))
                , 'max_depth': int(trial.suggest_uniform('max_depth', 2, 50))
                , 'criterion': trial.suggest_categorical('criterion', ['mse', 'mae'])
                , 'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
            }
            if params is not None:
                RF_params.update(params)
            model = RandomForestRegressor(**RF_params, n_jobs=15)

        elif self.method == 'AB':  # ada boost

            AB_params = {
                'n_estimators': int(trial.suggest_uniform('n_estimators', 20, 300))
                , 'learning_rate': trial.suggest_uniform('learning_rate', 0.05, 0.3)
                , 'loss': trial.suggest_categorical('loss', ['linear', 'square', 'exponential'])
            }
            if params is not None:
                AB_params.update(params)
            model = AdaBoostRegressor(**AB_params)

        elif self.method == 'GB':  # gradient boost

            GB_params = {
                'n_estimators': int(trial.suggest_loguniform('n_estimators', 20, 300))
                , 'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.5)
                , 'max_depth': int(trial.suggest_loguniform('max_depth', 2, 50))
                , 'loss': trial.suggest_categorical('loss', ['ls', 'lad', 'huber', 'quantile'])
            }
            if params is not None:
                GB_params.update(params)
            model = GradientBoostingRegressor(**GB_params)

        # elif self.method == 'NB':  # naive bayes
        #
        #     NB_params = {
        #         'var_smoothing': trial.suggest_loguniform('var_smoothing', 1e-10, 1e-05)
        #     }
        #     if params is not None:
        #         NB_params.update(params)
        #     model = skl.naive_bayes.GaussianNB(**NB_params)

        elif self.method == 'KNN':  # k-nearest neighbours

            KNN_params = {
                'n_neighbors': int(trial.suggest_loguniform('n_neighbors', 3, 7))
            }
            if params is not None:
                KNN_params.update(params)
            model = skl.neighbors.KNeighborsRegressor(**KNN_params)

        # elif self.method == 'LR':  # logistic retression
        #     LR_params = {
        #         'penalty': trial.suggest_categorical('penalty', ['l1', 'l2'])
        #         , 'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False])
        #     }
        #     if params is not None:
        #         LR_params.update(params)
        #     model = skl.linear_model.LogisticRegression(**LR_params)


        elif self.method == 'LGBM':  # logistic retression
            LGBM_params = {
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2'])
                , 'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False])
            }
            if params is not None:
                LGBM_params.update(params)
            model = LGBMRegressor(**LGBM_params)


        elif self.method == 'XGB':  # XGBoost

            # https://xgboost.readthedocs.io/en/latest/parameter.html
            XGB_params = {
                'n_estimators': int(trial.suggest_loguniform('n_estimators', 5, 300))
                , 'booster': trial.suggest_categorical('booster', ['dart', 'gbtree'])
                , 'eta': trial.suggest_loguniform('eta', 0.01, 0.5)
                , 'max_depth': int(trial.suggest_loguniform('max_depth', 3, 30))
                # ,'reg_lambda':       trial.suggest_loguniform( 'reg_lambda', 0, 1)
                # ,'reg_alpha' :       trial.suggest_loguniform( 'reg_alpha', 0, 1)
            }
            if params is not None:
                XGB_params.update(params)
            model = XGBRegressor(**XGB_params, nthread=15)

        elif self.method == 'CAT':  # catboost

            # https://catboost.ai/docs/concepts/python-reference_parameters-list.html#python-reference_parameters-list
            CAT_params = {
                'iterations': int(trial.suggest_loguniform('iterations', 20, 300))
                , 'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.5)
                , 'depth': int(trial.suggest_loguniform('depth', 2, 16))
            }
            if params is not None:
                CAT_params.update(params)
            model = CatBoostRegressor(**CAT_params, early_stopping_rounds=75, logging_level='Silent')



        # model fit
        model.fit(self.x_train, self.y_train)


        # model prediction and classification
        prob_train = pd.DataFrame(model.predict(self.x_train), columns=['y_pred'])
        prob_train = prob_train.reset_index(drop=True)

        prob_test = pd.DataFrame(model.predict(self.x_test), columns=['y_pred'])
        prob_test = prob_test.reset_index(drop=True)


        # selection and determining score function to optimize


        try:
            mean_squared_er_train = skm.mean_squared_error(y_true=self.y_train, y_pred=prob_train)
            mean_squared_er_test = skm.mean_squared_error(y_true=self.y_test, y_pred=prob_test)
        except:
            mean_squared_er_train = np.nan
            mean_squared_er_test  = np.nan

        try:
            mean_absolute_er_train = skm.median_absolute_error(y_true=self.y_train, y_pred=prob_train)
            mean_absolute_er_test = skm.median_absolute_error(y_true=self.y_test, y_pred=prob_test)
        except:
            mean_absolute_er_train = np.nan
            mean_absolute_er_test  = np.nan


        scores = pd.DataFrame([[mean_squared_er_train, mean_squared_er_test ,mean_absolute_er_train, mean_absolute_er_test]],
                       columns=['mean_squared_er_train', 'mean_squared_er_test', 'mean_absolute_er_train', 'mean_absolute_er_test'])

        self.scores = pd.concat([self.scores, scores])

        # zeracanie scoru którego 'optuna' używa do optymalizacji grida (możemy chyba tylko jedną wartość zwrócić.
        if self.opt_function == 'mean_squared_er':
            return (mean_squared_er_test)
        elif self.opt_function == 'mean_absolute_er':
            return (mean_absolute_er_test)


# y_train = y_train
# y_test = y_test
#
# ob = Objective(method = 'RF',  x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, threshold={'0':0.5, '1':0.5}, priori=None, pos_label=1, opt_function='aps')
# #
# study = optuna.create_study()
# #
# opt = study.optimize(ob, n_trials=3)


##################


def grid_optuna_(  methods_list=None
                 , direction="maximize"
                 , opt_function='aps'
                 , n_trials=10
                 , params_for_methods=None
                 , x_train=None
                 , y_train=None
                 , x_test=None
                 , y_test=None):
    trial_scores = pd.DataFrame()
    trial_hyperparameters = pd.DataFrame()

    # loop over models
    for i in range(len(methods_list)):
        # i = 0
        # loop over thresholds
        method = methods_list[i]
        if params_for_methods is not None:
            params = params_for_methods.get(method, None)
        else:
            params = None

        # grid
        ob = Objective(method=method, opt_function=opt_function, x_train=x_train, y_train=y_train, x_test=x_test,
                       y_test=y_test, params=params)
        study = optuna.create_study(direction=direction)
        opt = study.optimize(ob, n_trials=n_trials, gc_after_trial=True)

        # getting scores from trials of grid
        trial_scores_i = ob.scores

        # preparing scores to collect
        trial_scores_i['method'] = method
        trial_scores_i = trial_scores_i.reset_index(drop=True)
        trial_scores_i = trial_scores_i.reset_index(drop=False)


        # doliczenie proporcji train do test dla scorow
        trial_scores_i['mean_squared_er_overfit']    = trial_scores_i['mean_squared_er_test']/trial_scores_i['mean_squared_er_train']
        trial_scores_i['mean_squared_er_overfit']    = trial_scores_i['mean_squared_er_overfit'].replace(np.inf, -1)
        trial_scores_i['mean_absolute_er_overfit'] = trial_scores_i['mean_absolute_er_test']/trial_scores_i['mean_absolute_er_train']
        trial_scores_i['mean_absolute_er_overfit'] = trial_scores_i['mean_absolute_er_overfit'].replace(np.inf, -1)


        trial_scores = pd.concat([trial_scores, trial_scores_i], ignore_index=True)
        trial_scores = trial_scores.rename({'index': 'number'})





        # getting hyperparameters from trials of grid
        trial_hiperparameters_i = study.trials_dataframe()

        # preparing hyperparameters to collect
        trial_hiperparameters_i = trial_hiperparameters_i.drop(['datetime_start', 'datetime_complete', 'state'],
                                                                 axis=1)  # 'system_attrs__number'

        trial_hiperparameters_i = pd.concat([trial_hiperparameters_i, trial_scores_i[['mean_squared_er_test', 'mean_absolute_er_test']]], axis=1, ignore_index=False)


        trial_hiperparameters_i = trial_hiperparameters_i.melt(id_vars=['number'])

        trial_hiperparameters_i['method'] = method

        trial_hyperparameters = pd.concat([trial_hyperparameters, trial_hiperparameters_i], ignore_index=True)
    return ([trial_scores, trial_hyperparameters])


# data = r.diamonds
# data['target'] = pd.Series(np.random.randint(0,2, len(data) ))
# data['target'] = data['target'].astype(str)
#
# y_var = 'target'
# x_var = ['price', 'x', 'y']
#
# # podstawowy zbior danych
# x_train, x_test, y_train, y_test = skl.model_selection.train_test_split(data[x_var], data[y_var], test_size=0.33)
# x_train    = x_train.reset_index(drop=True)
# y_train    = y_train.reset_index(drop=True)
#
#
# methods_list = ['RF', 'AB']
#
# n_trials = 5
#
#
# grid
# grid_res = grid_optuna_( methods_list = methods_list
#             ,n_trials = 3
#             ,direction    = "minimize"
#             ,opt_function = 'mse'
#             ,x_train  = x_train
#             ,y_train  = y_train
#             ,x_test   = x_test
#             ,y_test   = y_test)
#
# # wyniki grida
# grid_res[0]
# grid_res[1]


def grid_optuna_scores_plot(data, fig_w=10, fig_h=10):
    """
    Line plot of scores from grid search
    """
    plotnine.options.figure_size = (fig_w, fig_h)
    data = pd.melt(frame=data, id_vars=['index', 'method'])
    print(ggplot(data=data) + geom_line(aes(x='index', y='value',colour='method')) + facet_grid('variable~.', scales='free_y'))


# TEST CODE:
# grid_optuna_scores_plot(grid_res[0])


def grid_optuna_hyperparameters_plot(data=None, method='RF', params_to_print=None, fig_w=10, fig_h=5):
    """
    Line plot of hyperparameters from grid search
    """
    plotnine.options.figure_size = (fig_w, fig_h)
    params_to_print = ['params_' + x for x in params_to_print]
    data = data.loc[(data['variable'].isin(params_to_print)) & (data['method'].isin([method])), :]
    data['value'] = data['value'].astype(float)

    # print(ggplot(data = trials) + geom_line(aes(x = 'number', y = 'value'))  +  facet_grid('.~threshold') + ggtitle('score'))
    print(ggplot(data=data) + geom_line(aes(x='number', y='value')) + facet_grid('variable~.', scales='free_y') + ggtitle(method))


# TEST CODE:
# grid_optuna_hyperparameters_plot(data = grid_res[1], params_to_print = ['n_estimators', 'max_depth'], method = 'RF')


def grid_optuna_hyperparameters_tables(trial_hyperparameters):
    scores = trial_hyperparameters[['method', 'variable','value']]
    sc_gr = scores.groupby(['method'])

    for name_1, group_1 in sc_gr:
        group_1 = group_1.reset_index(drop=True)
        group_1_gr = group_1.groupby(['variable'])
        scores_n = pd.DataFrame()
        for name_2, group_2 in group_1_gr:
            group_2 = group_2[['variable', 'value']]
            group_2 = group_2.reset_index(drop=True)
            scores_n = pd.concat([scores_n, group_2['value']], axis=1)
        print(name_1)

        scores_n.columns = list(group_1_gr.groups.keys())
        scores_n['value'] = scores_n['value'].astype('float64')
        display(scores_n.sort_values('value', ascending=False))

# hiplot.Experiment.from_iterable(data_df.reset_index(drop=True).to_dict(orient='Index').values()).display()