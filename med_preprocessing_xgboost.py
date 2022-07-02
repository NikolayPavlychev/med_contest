#/med_contest/preprocessing.py created by: Nikolay Pavlychev pavlychev.n.se@gmail.com
#-----------------------------------------------------------------------------------------------------------------------
print('Import libs...')

import os
import sys
import random
import time
import joblib

import json

import numpy as np
import scipy
import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression

import itertools
from itertools import product

import OhePreprocessing
import importlib
importlib.reload(OhePreprocessing)
from OhePreprocessing import OhePreprocessing

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print(torch.cuda.get_device_name(0))

print('Successfully!')

#-----------------------------------------------------------------------------------------------------------------------
print('Import data...')

ROOT_DIR = os.path.abspath(os.curdir)

cols = ['ID','Пол','Семья','Этнос','Национальность','Религия','Образование','Профессия','Вы_работаете','Выход_на_пенсию','Прекращение_работы_по_болезни','Сахарный_диабет','Гепатит','Онкология','Хроническое_заболевание_легких','Бронжиальная_астма','Туберкулез_легких_','ВИЧ/СПИД','Регулярный_прим_лекарственных_средств','Травмы_за_год','Переломы','Статус_Курения','Возраст_курения','Сигарет_в_день','Пассивное_курение','Частота_пасс_кур','Алкоголь','Возраст_алког','Время_засыпания','Время_пробуждения','Сон_после_обеда','Спорт_клубы','Религия_клубы','ID_y','Артериальная_гипертензия','ОНМК','Стенокардия_ИБС_инфаркт_миокарда','Сердечная_недостаточность','Прочие_заболевания_сердца']

dataset = pd.read_csv(ROOT_DIR + '/train_dataset_train/' + 'train.csv', dtype=object,sep=',', header=1, names=cols)

categorical_cols = ['Пол','Семья','Этнос','Национальность','Религия','Образование','Профессия',
                    'Статус_Курения','Частота_пасс_кур','Алкоголь']
binary_cols = ['Вы_работаете','Выход_на_пенсию','Прекращение_работы_по_болезни','Сахарный_диабет','Гепатит','Онкология','Хроническое_заболевание_легких','Бронжиальная_астма','Туберкулез_легких_','ВИЧ/СПИД','Регулярный_прим_лекарственных_средств','Травмы_за_год','Переломы',
            'Пассивное_курение','Сон_после_обеда','Спорт_клубы','Религия_клубы']
id_cols = ['ID', 'ID_y']
float_cols =['Возраст_курения','Сигарет_в_день', 'Возраст_алког']
time_cols = ['Время_засыпания','Время_пробуждения']
targets = ['Артериальная_гипертензия','ОНМК','Стенокардия_ИБС_инфаркт_миокарда','Сердечная_недостаточность','Прочие_заболевания_сердца']

print("Check cols count:")
print(len(list(dataset.columns))/(len(categorical_cols)+len(binary_cols)+len(id_cols)+len(float_cols)+len(time_cols)+len(targets)))

dataset['Время_засыпания'] = pd.to_datetime(dataset['Время_засыпания']).dt.hour
dataset['Время_пробуждения'] = pd.to_datetime(dataset['Время_пробуждения']).dt.hour
ind = dataset[dataset['Время_засыпания']==0].index
dataset.loc[ind,'Время_засыпания'] = 24
ind = dataset[dataset['Время_засыпания']>=18].index
dataset['AM_before'] = 0
dataset.loc[ind,'AM_before'] = 1
dataset['sleep_duration'] = 0
dataset.loc[ind, 'sleep_duration'] = 24-dataset['Время_засыпания']+dataset['Время_пробуждения']
ind = dataset[dataset['Время_засыпания']<18].index
dataset.loc[ind, 'sleep_duration'] = dataset['Время_пробуждения']-dataset['Время_засыпания']

dataset_view = dataset[['Время_засыпания', 'Время_пробуждения', 'AM_before', 'sleep_duration']]

# training_dataset, test_dataset = train_test_split(dataset, train_size=0.75)

for col in categorical_cols:
    max_value = dataset[col].dropna().max()
    dataset[col] = dataset[col].fillna(max_value)

training_dataset_ohe, train_cols_order, ohe_cols_order = OhePreprocessing(dataset=dataset, train_bool=True,feature_list=categorical_cols)
test_dataset_ohe  = OhePreprocessing(dataset=dataset, train_bool=False,feature_list=categorical_cols,train_cols_order=train_cols_order)

for col in float_cols:
    training_dataset_ohe[col] = training_dataset_ohe[col] .astype(float)
    # test_dataset_ohe[col] = test_dataset_ohe[col].astype(float)
    training_dataset_ohe[col] = training_dataset_ohe[col].fillna(training_dataset_ohe[col].mean())
    # test_dataset_ohe[col] = test_dataset_ohe[col].fillna(test_dataset_ohe[col].mean())
    min_value = training_dataset_ohe[col].min()
    max_value = training_dataset_ohe[col].max()
    training_dataset_ohe[col] = training_dataset_ohe[col].apply(lambda x: (x-min_value)/(max_value-min_value))
    # min_value = test_dataset_ohe[col].min()
    # max_value = test_dataset_ohe[col].max()
    # test_dataset_ohe[col] = test_dataset_ohe[col].apply(lambda x: (x-min_value)/(max_value-min_value))

for col in binary_cols:
    training_dataset_ohe[col] = training_dataset_ohe[col] .astype(int)
    # test_dataset_ohe[col] = test_dataset_ohe[col].astype(int)

for col in targets:
    training_dataset_ohe[col] = training_dataset_ohe[col] .astype(int)
    # test_dataset_ohe[col] = test_dataset_ohe[col].astype(int)

cols_order = ['ID']+float_cols+time_cols+binary_cols+ohe_cols_order+targets
cols_order.remove('Время_засыпания')
cols_order.remove('Время_пробуждения')

training_dataset_ohe = training_dataset_ohe[cols_order]
# test_dataset_ohe = test_dataset_ohe[cols_order]



params_dict_list = {'C': [0.005, 0.1, 1, 10], 'penalty': ['l2'],
                    'class_weight':['balanced'], 'n_jobs': [-1], 'random_state':[42], 'max_iter':[100,200], 'class_weight':['balanced']}
keys, values = zip(*params_dict_list.items())
permutations_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

for k, item in enumerate(permutations_params_dict_list):
    f1_score_training_mean = []
    f1_score_test_mean = []
    weights_train = []
    weights_test = []

    print('Experiment '+str(k)+' was started...')
    for target in targets:
        X_train, X_test, y_train, y_test  = train_test_split(training_dataset_ohe.drop(['ID'], axis=1).drop(targets, axis=1), training_dataset_ohe[target],test_size=0.3,random_state=42)
        weights_train.append(y_train.value_counts()[1])
        weights_test.append(y_test.value_counts()[1])
        model = LogisticRegression(class_weight= item['class_weight'], C= item['C'], penalty= item['penalty'], max_iter= item['max_iter'], n_jobs= item['n_jobs'], random_state= item['random_state'])
        model.fit(X_train, y_train)
        preds_train = model.predict(X_train)
        f1 = f1_score(y_train, preds_train)
        print('training f1 score', 'class = ',target,' ', f1)
        f1_score_training_mean.append(f1)

        preds_test = model.predict(X_test)
        f1 = f1_score(y_test, preds_test)
        print('test f1 score', 'class = ', target, ' ', f1)
        f1_score_test_mean.append(f1)

    weights_train =weights_train/np.sum(weights_train)
    weights_test =weights_test/np.sum(weights_test)

    print('f1_score_train_mean = ',np.mean(f1_score_training_mean))
    print('f1_score_test_mean = ',np.mean(f1_score_test_mean))

    weight_f1_score_training_mean = []
    for num1, num2 in zip(f1_score_training_mean, weights_train):
        weight_f1_score_training_mean.append(num1*num2)
    weight_f1_score_test_mean = []
    for num1, num2 in zip(f1_score_test_mean, weights_test):
        weight_f1_score_test_mean.append(num1*num2)

    print('weight_f1_score_train_mean = ',np.sum(weight_f1_score_training_mean))
    print('weight_f1_score_test_mean = ',np.sum(weight_f1_score_test_mean))

    metrics ={}
    metrics.update({'f1_score_train_mean':f1_score_training_mean})
    metrics.update({'f1_score_test_mean':f1_score_test_mean})
    metrics.update({'weight_f1_score_train_mean':weight_f1_score_training_mean})
    metrics.update({'weight_f1_score_test_mean':weight_f1_score_test_mean})

    with open(ROOT_DIR+'/experiments/'+'metrics_exp_'+str(k)+'  '+str(item)+'.json', 'w') as outfile:
        json.dump(metrics, outfile)

    print('Successfully!')

#logs analysis and select the best opt model

# joblib.dump(model, './MedContest_LogReg_best.pickle')


f1_score_training_mean = []
f1_score_test_mean = []
weights_train = []
weights_test = []

from sklearn.svm import SVC
import xgboost as xgb

for target in targets:
    import xgboost as xgb

    X_train, X_test, y_train, y_test = train_test_split(training_dataset_ohe.drop(['ID'], axis=1).drop(targets, axis=1),
                                                        training_dataset_ohe[target], test_size=0.3, random_state=42)

    # clf = xgb.XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=10, random_state=0,tree_method='gpu_hist', reg_lambda=10)
    clf = SVC(C=0.1, kernel='poly', degree=2,gamma='scale')
    clf.fit(X_train,y_train)
    preds_train =  clf.predict(X_train)
    preds_test =  clf.predict(X_test)

    f1 = f1_score(y_train, preds_train)
    print('training f1 score', 'class = ', target, ' ', f1)
    f1_score_training_mean.append(f1)

    preds_test = clf.predict(X_test)
    f1 = f1_score(y_test, preds_test)
    print('test f1 score', 'class = ', target, ' ', f1)
    f1_score_test_mean.append(f1)

print('f1_score_train_mean = ',np.mean(f1_score_training_mean))
print('f1_score_test_mean = ',np.mean(f1_score_test_mean))



