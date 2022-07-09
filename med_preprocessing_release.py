#/med_contest/preprocessing.py created by: Nikolay Pavlychev pavlychev.n.se@gmail.com
#-----------------------------------------------------------------------------------------------------------------------
print('Import libs...')

import os
import sys
import random
import time
import joblib

import json

import warnings
warnings.filterwarnings("ignore")


import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import recall_score, precision_score,recall_score, classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression

import itertools
from itertools import product

import OhePreprocessing
import importlib
importlib.reload(OhePreprocessing)
from OhePreprocessing import OhePreprocessing

print('Successfully!')

#-----------------------------------------------------------------------------------------------------------------------
print('Import data...')

ROOT_DIR = os.path.abspath(os.curdir)

cols = ['ID','Пол','Семья','Этнос','Национальность','Религия','Образование','Профессия','Вы_работаете','Выход_на_пенсию','Прекращение_работы_по_болезни','Сахарный_диабет','Гепатит','Онкология','Хроническое_заболевание_легких','Бронжиальная_астма','Туберкулез_легких_','ВИЧ/СПИД','Регулярный_прим_лекарственных_средств','Травмы_за_год','Переломы','Статус_Курения','Возраст_курения','Сигарет_в_день','Пассивное_курение','Частота_пасс_кур','Алкоголь','Возраст_алког','Время_засыпания','Время_пробуждения','Сон_после_обеда','Спорт_клубы','Религия_клубы','ID_y','Артериальная_гипертензия','ОНМК','Стенокардия_ИБС_инфаркт_миокарда','Сердечная_недостаточность','Прочие_заболевания_сердца']

dataset = pd.read_csv(ROOT_DIR + '/train_dataset_train/' + 'train.csv', dtype=object,sep=',', header=0, names=cols)
dataset = dataset.drop(['ID_y'],axis=1)
categorical_cols = ['Пол','Семья','Этнос','Национальность','Религия','Образование','Профессия',
                    'Статус_Курения','Частота_пасс_кур','Алкоголь']
binary_cols = ['Вы_работаете','Выход_на_пенсию','Прекращение_работы_по_болезни','Сахарный_диабет','Гепатит','Онкология','Хроническое_заболевание_легких','Бронжиальная_астма','Туберкулез_легких_','ВИЧ/СПИД','Регулярный_прим_лекарственных_средств','Травмы_за_год','Переломы',
            'Пассивное_курение','Сон_после_обеда','Спорт_клубы','Религия_клубы']
id_cols = ['ID' ]
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

for col in categorical_cols:
    max_value = dataset[col].dropna().max()
    dataset[col] = dataset[col].fillna(max_value)

training_dataset_ohe, train_cols_order, ohe_cols_order = OhePreprocessing(dataset=dataset, train_bool=True,feature_list=categorical_cols)

for col in float_cols:
    training_dataset_ohe[col] = training_dataset_ohe[col] .astype(float)
    training_dataset_ohe[col] = training_dataset_ohe[col].fillna(training_dataset_ohe[col].mean())
    min_value = training_dataset_ohe[col].min()
    max_value = training_dataset_ohe[col].max()
    training_dataset_ohe[col] = training_dataset_ohe[col].apply(lambda x: (x-min_value)/(max_value-min_value))

for col in binary_cols:
    training_dataset_ohe[col] = training_dataset_ohe[col] .astype(int)

for col in targets:
    training_dataset_ohe[col] = training_dataset_ohe[col] .astype(int)

cols_order = ['ID']+float_cols+time_cols+binary_cols+ohe_cols_order+targets
cols_order.remove('Время_засыпания')
cols_order.remove('Время_пробуждения')

training_dataset_ohe = training_dataset_ohe[cols_order]

for target in targets:
    print(training_dataset_ohe[target].value_counts())

#-----------------------------------------------------------------------------------------------------------------------
#Preprocessing val samples

submit_cols = ['ID', 'Артериальная гипертензия','ОНМК','"Стенокардия, ИБС, инфаркт миокарда"','Сердечная недостаточность','Прочие заболевания сердца']


cols = ['ID','Пол','Семья','Этнос','Национальность','Религия','Образование','Профессия','Вы_работаете','Выход_на_пенсию','Прекращение_работы_по_болезни','Сахарный_диабет','Гепатит','Онкология','Хроническое_заболевание_легких','Бронжиальная_астма','Туберкулез_легких_','ВИЧ/СПИД','Регулярный_прим_лекарственных_средств','Травмы_за_год','Переломы','Статус_Курения','Возраст_курения','Сигарет_в_день','Пассивное_курение','Частота_пасс_кур','Алкоголь','Возраст_алког','Время_засыпания','Время_пробуждения','Сон_после_обеда','Спорт_клубы','Религия_клубы']

dataset = pd.read_csv(ROOT_DIR + '/' + 'test_dataset_test.csv', dtype=object,sep=',', header=0, names=cols)

categorical_cols = ['Пол','Семья','Этнос','Национальность','Религия','Образование','Профессия',
                    'Статус_Курения','Частота_пасс_кур','Алкоголь']
binary_cols = ['Вы_работаете','Выход_на_пенсию','Прекращение_работы_по_болезни','Сахарный_диабет','Гепатит','Онкология','Хроническое_заболевание_легких','Бронжиальная_астма','Туберкулез_легких_','ВИЧ/СПИД','Регулярный_прим_лекарственных_средств','Травмы_за_год','Переломы',
            'Пассивное_курение','Сон_после_обеда','Спорт_клубы','Религия_клубы']
id_cols = ['ID']
float_cols =['Возраст_курения','Сигарет_в_день', 'Возраст_алког']
time_cols = ['Время_засыпания','Время_пробуждения']
targets = ['Артериальная_гипертензия','ОНМК','Стенокардия_ИБС_инфаркт_миокарда','Сердечная_недостаточность','Прочие_заболевания_сердца']

print("Check cols count:")
print(len(list(dataset.columns))/(len(categorical_cols)+len(binary_cols)+len(id_cols)+len(float_cols)+len(time_cols)))

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

for col in categorical_cols:
    max_value = dataset[col].dropna().max()
    dataset[col] = dataset[col].fillna(max_value)

test_dataset_ohe  = OhePreprocessing(dataset=dataset, train_bool=False,feature_list=categorical_cols,train_cols_order=train_cols_order)

for col in float_cols:

    test_dataset_ohe[col] = test_dataset_ohe[col].astype(float)

    test_dataset_ohe[col] = test_dataset_ohe[col].fillna(test_dataset_ohe[col].mean())

    min_value = test_dataset_ohe[col].min()
    max_value = test_dataset_ohe[col].max()
    test_dataset_ohe[col] = test_dataset_ohe[col].apply(lambda x: (x-min_value)/(max_value-min_value))

for col in binary_cols:
    test_dataset_ohe[col] = test_dataset_ohe[col] .astype(int)
    # test_dataset_ohe[col] = test_dataset_ohe[col].astype(int)


cols_order = ['ID']+float_cols+time_cols+binary_cols+ohe_cols_order
cols_order.remove('Время_засыпания')
cols_order.remove('Время_пробуждения')

test_dataset_ohe = test_dataset_ohe[cols_order]

#-----------------------------------------------------------------------------------------------------------------------
#Experiments

params_dict_list = {'C': [0.001, 0.001, 0.01, 0.1, 1, 10], 'penalty': ['l1','l2','elasticnet'], 'solver':['saga'],
                    'l1_ratio':[0.3,0.4,0.5,0.6],'class_weight':['balanced'], 'n_jobs': [-1], 'random_state':[42],
                    'max_iter':[50,100,200,300], 'test_size':[0.1]}
keys, values = zip(*params_dict_list.items())
permutations_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

recall_models_test_max = []
recall_models_training_max = []

index_test_max = []
index_train_max = []

#-----------------------------------------------------------------------------------------------------------------------
#Training models

for target in targets:
    print('Experiment ' + str(target) + ' was started...')
    recall_score_training_max = []
    recall_score_test_max = []
    for k, item in enumerate(permutations_params_dict_list):
        item = permutations_params_dict_list[71]
        X_train, X_test, y_train, y_test  = train_test_split(training_dataset_ohe.drop(['ID'], axis=1).drop(targets, axis=1), training_dataset_ohe[target],test_size=item['test_size'],random_state=item['random_state'])
        model = LogisticRegression(class_weight= item['class_weight'], C= item['C'], penalty= item['penalty'], solver=item['solver'], l1_ratio=item['l1_ratio'],max_iter= item['max_iter'], n_jobs= item['n_jobs'], random_state= item['random_state'])
        model.fit(X_train, y_train)
        preds_train = model.predict(X_train)
        recall = recall_score(y_train, preds_train)
        recall_score_training_max.append(recall)

        preds_test = model.predict(X_test)
        recall = recall_score(y_test, preds_test)
        recall_score_test_max.append(recall)

        break


    indices_train_filter = np.argwhere(np.array(recall_score_training_max) == 1)
    indices_train_filter = np.squeeze(indices_train_filter).tolist()
    if isinstance(indices_train_filter, int):
        recall_score_training_max[indices_train_filter] = 0.5
    else:
        for ind in indices_train_filter:
            recall_score_training_max[ind] = 0.5

    indices_train_filter = np.argwhere(np.array(recall_score_test_max) == 1)
    indices_train_filter = np.squeeze(indices_train_filter).tolist()
    if isinstance(indices_train_filter, int):
        recall_score_test_max[indices_train_filter] = 0.5
    else:
        for ind in indices_train_filter:
            recall_score_test_max[ind] = 0.5

    tmp = max(recall_score_training_max)
    index = recall_score_training_max.index(tmp)

    recall_test_max = recall_score_test_max[index]
    recall_training_max = recall_score_training_max[index]

    recall_models_test_max.append(recall_test_max)
    index_test_max.append(index)
    recall_models_training_max.append(recall_training_max)
    index_train_max.append(index)


print('MODEL: ',target)
print('recall_models_test_opt = ',np.mean(recall_models_test_max))
print('recall_models_training_opt = ',np.mean(recall_models_training_max))
print('Successfully!')

#-----------------------------------------------------------------------------------------------------------------------
#Inference opt models
dataset_preds_val = pd.DataFrame(test_dataset_ohe['ID'])

for k, target in enumerate(targets):
    print('Inference best model ' + str(target) + ' was started...')
    recall_score_training_max = []
    recall_score_test_max = []
    for item in permutations_params_dict_list:
        # item = permutations_params_dict_list[index_test_max[k]]
        item = permutations_params_dict_list[71]
        X_train, X_test, y_train, y_test  = train_test_split(training_dataset_ohe.drop(['ID'], axis=1).drop(targets, axis=1), training_dataset_ohe[target],test_size=0.001,random_state=item['random_state'])
        model = LogisticRegression(class_weight= item['class_weight'], C= item['C'], penalty= item['penalty'], solver=item['solver'], l1_ratio=item['l1_ratio'],max_iter= item['max_iter'], n_jobs= item['n_jobs'], random_state= item['random_state'])
        model.fit(X_train, y_train)

        break

    preds_val = model.predict(test_dataset_ohe.drop(['ID'], axis=1))

    dataset_preds_val[target] = preds_val

dataset_preds_val.columns = submit_cols

dataset_val = dataset[['ID']].merge(dataset_preds_val,on='ID',how='inner')

dataset_val.to_csv(ROOT_DIR + '/' + 'LogReg_submit.csv',sep=',',index=False)
#-----------------------------------------------------------------------------------------------------------------------
