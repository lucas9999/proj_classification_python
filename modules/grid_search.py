import optuna
import pandas as pd
import numpy as np

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from plotnine.themes.themeable import axis_line
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
# from xgboost                   import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
import sklearn.metrics as skm
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
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

        if self.method == 'SVC':  # support vectors machine

            SVM_params = {
                'svc_c': trial.suggest_loguniform('svc_c', 1e-10, 1e10)}
            if params is not None:
                SVM_params.update(params)
            model = sklearn.svm.SVC(**SVM_params)

        elif self.method == 'RF':  # random forest

            # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.feature_importances_
            RF_params = {
                'n_estimators': int(trial.suggest_uniform('n_estimators', 20, 300))
                , 'max_depth': int(trial.suggest_uniform('max_depth', 2, 50))
                , 'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
                , 'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
                , 'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
            }
            if params is not None:
                RF_params.update(params)
            model = RandomForestClassifier(**RF_params, n_jobs=15)

        elif self.method == 'AB':  # ada boost

            AB_params = {
                'n_estimators': int(trial.suggest_uniform('n_estimators', 20, 300))
                , 'learning_rate': trial.suggest_uniform('learning_rate', 0.05, 0.3)
                , 'algorithm': trial.suggest_categorical('algorithm', ['SAMME.R', 'SAMME'])
            }
            if params is not None:
                AB_params.update(params)
            model = AdaBoostClassifier(**AB_params)

        elif self.method == 'GB':  # gradient boost

            GB_params = {
                'n_estimators': int(trial.suggest_loguniform('n_estimators', 20, 300))
                , 'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.5)
                , 'max_depth': int(trial.suggest_loguniform('max_depth', 2, 50))
                , 'loss': trial.suggest_categorical('loss', ['deviance', 'exponential'])
            }
            if params is not None:
                GB_params.update(params)
            model = GradientBoostingClassifier(**GB_params)

        elif self.method == 'NB':  # naive bayes

            NB_params = {
                'var_smoothing': trial.suggest_loguniform('var_smoothing', 1e-10, 1e-05)
            }
            if params is not None:
                NB_params.update(params)
            model = skl.naive_bayes.GaussianNB(**NB_params)

        elif self.method == 'KNN':  # k-nearest neighbours

            KNN_params = {
                'n_neighbors': int(trial.suggest_loguniform('n_neighbors', 3, 7))
            }
            if params is not None:
                KNN_params.update(params)
            model = skl.neighbors.KNeighborsClassifier(**KNN_params)

        elif self.method == 'LR':  # logistic retression
            LR_params = {
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2'])
                , 'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False])
            }
            if params is not None:
                LR_params.update(params)
            model = skl.linear_model.LogisticRegression(**LR_params)


        elif self.method == 'LGBM':  # logistic retression
            LGBM_params = {
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2'])
                , 'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False])
            }
            if params is not None:
                LGBM_params.update(params)
            model = LGBMClassifier(**LGBM_params)


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
            model = XGBClassifier(**XGB_params, nthread=15)

        elif self.method == 'CAT':  # catboost

            # https://catboost.ai/docs/concepts/python-reference_parameters-list.html#python-reference_parameters-list
            CAT_params = {
                'iterations': int(trial.suggest_loguniform('iterations', 20, 300))
                , 'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.5)
                , 'depth': int(trial.suggest_loguniform('depth', 2, 16))
            }
            if params is not None:
                CAT_params.update(params)
            model = CatBoostClassifier(**CAT_params, early_stopping_rounds=75, logging_level='Silent')

        # calibration (optional)
        if self.calibration_method is not None:
            model = CalibratedClassifierCV(model, cv=3, method=self.calibration_method)

        # model fit
        model.fit(self.x_train, self.y_train)

        # y labels
        y_labels = list(self.y_train.drop_duplicates())
        y_labels.sort()
        self.y_labels = y_labels

        # model prediction and classification
        prob_train = pd.DataFrame(model.predict_proba(self.x_train), columns=[str(x) for x in model.classes_])
        prob_train = prob_train.reset_index(drop=True)

        prob_test = pd.DataFrame(model.predict_proba(self.x_test), columns=[str(x) for x in model.classes_])
        prob_test = prob_test.reset_index(drop=True)

        if self.priori is not None:

            for p in range(prob_test.shape[1]):
                var = prob_test.columns[p]
                priori_p = self.priori[var]
                prob_test.iloc[:, p] = prob_test.iloc[:, p] * priori_p
                prob_train.iloc[:, p] = prob_train.iloc[:, p] * priori_p
                prob_train_sum = prob_train.apply(lambda x: 1 / sum(x), axis=1)
                prob_test_sum = prob_test.apply(lambda x: 1 / sum(x), axis=1)

                for p in self.y_labels:
                    prob_train.loc[:, p] = prob_train.loc[:, p] * prob_train_sum
                    prob_test.loc[:, p] = prob_test.loc[:, p] * prob_test_sum

        if prob_test.shape[1] == 2 and self.threshold is not None:
            var_1 = list(self.threshold.keys())[0]
            var_2 = list(self.threshold.keys())[1]
            threshold_var_1 = list(self.threshold.values())[0]
            threshold_var_2 = list(self.threshold.values())[1]
            classification_train = pd.DataFrame([var_1 if x >= threshold_var_1 else var_2 for x in prob_train[str(var_1)]])
            classification_test = pd.DataFrame([var_1 if x >= threshold_var_1 else var_2 for x in prob_test[str(var_1)]])

        else:
            classification_train = pd.DataFrame(prob_train.idxmax(axis=1))  # zapis jako DataFrame by dalej zrobić 'concat'
            classification_test = pd.DataFrame(prob_test.idxmax(axis=1))  # zapis jako DataFrame by dalej zrobić 'concat'

        # selection and determining score function to optimize

        balanced_accuracy = skm.balanced_accuracy_score(y_true=self.y_test, y_pred=classification_test)
        accuracy = skm.accuracy_score(y_true=self.y_test, y_pred=classification_test)

        # ta funkcja nie jest napisane pod katem modeli multilabel
        aps = average_precision_score(y_true=self.y_test, y_score=prob_test[str(self.pos_label)], pos_label=str(self.pos_label))

        skm.recall_score(y_true=self.y_test.astype(int), y_pred=classification_test.astype(int), average='binary', pos_label=self.pos_label)

        # recall
        try:
            if len(self.y_labels) < 3:
                recall_train = skm.recall_score(y_true=self.y_train.astype(int), y_pred=classification_train.astype(int), average='binary', pos_label=self.pos_label)
                recall_test  = skm.recall_score(y_true=self.y_test.astype(int),  y_pred=classification_test.astype(int),  average='binary', pos_label=self.pos_label)
            else:
                recall_train = skm.recall_score(y_true=self.y_train.astype(str), y_pred=classification_train.astype(str), average='weighted')
                recall_test  = skm.recall_score(y_true=self.y_test.astype(str), y_pred=classification_test.astype(str), average='weighted')
        except:
            recall_train = np.nan
            recall_test  = np.nan

        # precision
        try:
            if len(self.y_labels) < 3:
                precision_train = skm.precision_score(y_true=self.y_train.astype(int), y_pred=classification_train.astype(int), average='binary', pos_label=self.pos_label)
                precision_test  = skm.precision_score(y_true=self.y_test.astype(int), y_pred=classification_test.astype(int), average='binary',   pos_label=self.pos_label)
            else:
                precision_train = skm.precision_score(y_true=self.y_train.astype(str), y_pred=classification_train.astype(str), average='weighted')
                precision_test  = skm.precision_score(y_true=self.y_test.astype(str), y_pred=classification_test.astype(str),   average='weighted')
        except:
            precision_train = np.nan
            precision_test  = np.nan

        # f1
        try:
            if len(self.y_labels) < 3:
                f1_train = skm.f1_score(y_true=self.y_train, y_pred=classification_train, average='binary', pos_label=str(self.pos_label))
                f1_test  = skm.f1_score(y_true=self.y_test, y_pred=classification_test, average='binary', pos_label=str(self.pos_label))
            else:
                f1_train = skm.f1_score(y_true=self.y_train, y_pred=classification_train, average='weighted')
                f1_test  = skm.f1_score(y_true=self.y_test, y_pred=classification_test, average='weighted')
        except:
            f1_train = np.nan
            f1_test  = np.nan

        scores = pd.DataFrame([[balanced_accuracy, accuracy, recall_train, recall_test, precision_train, precision_test, f1_train, f1_test, aps]],
                       columns=['balanced_accuracy', 'accuracy', 'recall_train', 'recall_test', 'precision_train', 'precision_test', 'f1_train', 'f1_test','aps'])

        self.scores = pd.concat([self.scores, scores])

        # zeracanie scoru którego 'optuna' używa do optymalizacji grida (możemy chyba tylko jedną wartość zwrócić.
        if self.opt_function == 'balanced_accuracy':
            return (balanced_accuracy)
        elif self.opt_function == 'aps':
            return (aps)
        elif self.opt_function == 'accuracy':
            return (accuracy)
        elif self.opt_function == 'recall':
            return (recall_test)
        elif self.opt_function == 'precision':
            return (precision_test)


# y_train = y_train.astype(str)
# y_test = y_test.astype(str)
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
                 , thresholds_list=None
                 , pos_label=None
                 , priori=None
                 , n_trials=10
                 , params_for_methods=None
                 , x_train=None
                 , y_train=None
                 , x_test=None
                 , y_test=None):
    trial_scores = pd.DataFrame()
    trial_hyperparameters = pd.DataFrame()
    threshold_columns = list(thresholds_list.columns)

    # loop over models
    for i in range(len(methods_list)):
        # i = 0
        # loop over thresholds
        method = methods_list[i]
        if params_for_methods is not None:
            params = params_for_methods.get(method, None)
        else:
            params = None

        for j in range(len(thresholds_list)):
            # j = 0
            print([method, j])

            threshold = {threshold_columns[0]: thresholds_list.iloc[j, 0],
                         threshold_columns[1]: thresholds_list.iloc[j, 1]}

            # grid
            ob = Objective(method=method, opt_function=opt_function, x_train=x_train, y_train=y_train, x_test=x_test,
                           y_test=y_test, threshold=threshold, priori=priori, pos_label=pos_label, params=params)
            study = optuna.create_study(direction=direction)
            opt = study.optimize(ob, n_trials=n_trials, gc_after_trial=True)

            # getting scores from trials of grid
            trial_scores_ij = ob.scores

            # preparing scores to collect
            trial_scores_ij['method'] = method
            trial_scores_ij['threshold'] = thresholds_list.iloc[j, 0]
            trial_scores_ij = trial_scores_ij.reset_index(drop=True)
            trial_scores_ij = trial_scores_ij.reset_index(drop=False)


            # doliczenie proporcji train do test dla scorow
            trial_scores_ij['recall_overfit']    = trial_scores_ij['recall_test']/trial_scores_ij['recall_train']
            trial_scores_ij['recall_overfit']    = trial_scores_ij['recall_overfit'].replace(np.inf, -1)
            trial_scores_ij['precision_overfit'] = trial_scores_ij['precision_test']/trial_scores_ij['precision_train']
            trial_scores_ij['precision_overfit'] = trial_scores_ij['precision_overfit'].replace(np.inf, -1)
            trial_scores_ij['f1_overfit']        = trial_scores_ij['f1_test']/trial_scores_ij['f1_train']
            trial_scores_ij['f1_overfit']        = trial_scores_ij['f1_overfit'].replace(np.inf, -1)


            trial_scores = pd.concat([trial_scores, trial_scores_ij], ignore_index=True)
            trial_scores = trial_scores.rename({'index': 'number'})





            # getting hyperparameters from trials of grid
            trial_hiperparameters_ij = study.trials_dataframe()

            # preparing hyperparameters to collect
            trial_hiperparameters_ij = trial_hiperparameters_ij.drop(['datetime_start', 'datetime_complete', 'state'],
                                                                     axis=1)  # 'system_attrs__number'

            trial_hiperparameters_ij = pd.concat([trial_hiperparameters_ij, trial_scores_ij[['recall_overfit', 'precision_overfit','f1_overfit']]], axis=1, ignore_index=False)


            trial_hiperparameters_ij = trial_hiperparameters_ij.melt(id_vars=['number'])

            trial_hiperparameters_ij['method'] = method
            trial_hiperparameters_ij['threshold'] = thresholds_list.iloc[j, 0]

            trial_hyperparameters = pd.concat([trial_hyperparameters, trial_hiperparameters_ij], ignore_index=True)
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
#
# thresholds_list = pd.DataFrame({'0':[0.2, 0.4, 0.8]})
# thresholds_list['1'] = 1 - thresholds_list['0']
#
# methods_list = ['RF', 'AB']
#
# priori = None
# n_trials = 5
#
#
# grid
# grid_res = grid_optuna_( methods_list = methods_list
#             ,thresholds_list = thresholds_list
#             ,priori   = None
#             ,n_trials = 3
#             ,direction    = "maximize"
#             ,opt_function = 'recall'
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
    data = pd.melt(frame=data, id_vars=['index', 'method', 'threshold'])
    data['threshold'] = data['threshold'].astype(str)
    print(ggplot(data=data) + geom_line(aes(x='index', y='value', color='threshold')) + facet_grid('variable~method',
                                                                                                   scales='free_y'))


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
    data['threshold'] = data['threshold'].astype(str)

    # print(ggplot(data = trials) + geom_line(aes(x = 'number', y = 'value'))  +  facet_grid('.~threshold') + ggtitle('score'))
    print(ggplot(data=data) + geom_line(aes(x='number', y='value', color='threshold')) + facet_grid('variable~method', scales='free_y') + ggtitle(method))


# TEST CODE:
# grid_optuna_hyperparameters_plot(data = grid_res[1], params_to_print = ['n_estimators', 'max_depth'], method = 'RF')


def grid_optuna_hyperparameters_tables(trial_hyperparameters):
    scores = trial_hyperparameters[['method', 'threshold', 'variable','value']]
    sc_gr = scores.groupby(['method', 'threshold'])

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