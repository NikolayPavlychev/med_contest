#/itmo_contest/preprocessing.py created by: Nikolay Pavlychev pavlychev.n.se@gmail.com
#-----------------------------------------------------------------------------------------------------------------------
print('Import libs...')
import os
import sys
import random
import time
import joblib

import numpy as np
import scipy
import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder

print('Successfully!')
#-----------------------------------------------------------------------------------------------------------------------
print('Import data...')
ROOT_DIR = os.path.abspath(os.curdir)

cols_comp_students = ['ISU', 'KURS', 'DATE_START', 'DATE_END', 'PRIZNAK', 'MAIN_PLAN']
comp_students = pd.read_csv(ROOT_DIR + '/data/' + 'comp_students.csv', dtype=object,sep=',', header=1, names=cols_comp_students)

cols_comp_portrait = ['ISU', 'GENDER', 'CITIZENSHIP', 'EXAM_TYPE', 'EXAM_SUBJECT_1', 'EXAM_SUBJECT_2', 'EXAM_SUBJECT_3',
                      'ADMITTED_EXAM_1', 'ADMITTED_EXAM_2', 'ADMITTED_EXAM_3', 'ADMITTED_SUBJECT_PRIZE_LEVEL',
                      'REGION_ID']
comp_portrait = pd.read_csv(ROOT_DIR + '/data/' + 'comp_portrait.csv', dtype=object,sep=',', header=1, names=cols_comp_portrait)

cols_comp_marks = ['ISU', 'ST_YEAR', 'SEMESTER', 'TYPE_NAME', 'MARK', 'MAIN_PLAN', 'DISC_ID', 'PRED_ID']
comp_marks = pd.read_csv(ROOT_DIR + '/data/' + 'comp_marks.csv', dtype=object,sep=',', header=1, names=cols_comp_marks)

cols_comp_disc = ['PLAN_ID', 'DISC_ID', 'CHOICE', 'SEMESTER', 'DISC_NAME', 'DISC_DEP', 'KEYWORD_NAMES']
comp_disc = pd.read_csv(ROOT_DIR + '/data/' + 'comp_disc.csv', dtype=object,sep=',', header=1, names=cols_comp_disc)

cols_comp_teachers = ['ISU', 'GENDER', 'DATE_BIRTH', 'ST_YEAR', 'SEMESTER', 'DISC_ID', 'MAIN_PLAN', 'TYPE_NAME', 'MARK']
comp_teachers = pd.read_csv(ROOT_DIR + '/data/' + 'comp_teachers.csv', dtype=object,sep=',', header=1, names=cols_comp_teachers)

cols_train = ['ISU', 'ST_YEAR','SEMESTER', 'DISC_ID', 'TYPE_NAME',  'DEBT']
train = pd.read_csv(ROOT_DIR + '/samples/' + 'train.csv', dtype=object,sep=',', header=1, names=cols_train)

cols_test = ['ISU', 'ST_YEAR', 'SEMESTER', 'DISC_ID', 'TYPE_NAME']
test = pd.read_csv(ROOT_DIR + '/samples/' + 'test.csv', dtype=object,sep=',', header=1, names=cols_test)

print('Successfully!')
#-----------------------------------------------------------------------------------------------------------------------
print("Start preprocessing...")
print("Converting data type and print info...")
comp_students['DATE_START'] = pd.to_datetime(comp_students['DATE_START'],yearfirst=True).dt.normalize()
comp_students['DATE_END'] = pd.to_datetime(comp_students['DATE_END'],yearfirst=True).dt.normalize()

print(comp_students.info())

comp_portrait['ADMITTED_EXAM_1'] = comp_portrait['ADMITTED_EXAM_1'].astype(float)
comp_portrait['ADMITTED_EXAM_2'] = comp_portrait['ADMITTED_EXAM_2'].astype(float)
comp_portrait['ADMITTED_EXAM_3'] = comp_portrait['ADMITTED_EXAM_3'].astype(float)

print(comp_portrait.info())

print(comp_marks.info())

print(comp_disc.info())

comp_teachers['MARK']= comp_teachers['MARK'].astype(float)

print(comp_teachers.info())

train['DEBT'] = train['DEBT'].astype(int)

print(train.info())

print(test.info())

print('Successfully!')
#-----------------------------------------------------------------------------------------------------------------------
print('Merging data...')
print('comp_students table:')
print(' Full students ISU:',comp_students.shape[0],'\n',
      'Unique students ISU=',comp_students['ISU'].unique().shape[0])
print('Check duplicates:')
print(comp_students.shape[0]/comp_students.drop_duplicates().shape[0])

print('comp_portrait table:')
print(' Full students ISU:',comp_portrait.shape[0],'\n',
      'Unique students ISU=',comp_portrait['ISU'].unique().shape[0])
print('Check duplicates:')
print(comp_portrait.shape[0]/comp_portrait.drop_duplicates().shape[0])

# comp_portrait_students = comp_portrait.merge(comp_students,on=['ISU'],how='outer')
# print('comp_portrait_students shape = ',comp_portrait_students.shape)

print('comp_marks table:')
print(' Full students ISU:',comp_marks.shape[0],'\n',
      'Unique students ISU=',comp_marks['ISU'].unique().shape[0])
print('Check duplicates:')
print(comp_marks.shape[0]/comp_marks.drop_duplicates().shape[0])
comp_marks = comp_marks.drop_duplicates()
print('Check duplicates:')
print(comp_marks.shape[0]/comp_marks.drop_duplicates().shape[0])

print('comp_disc table:')
print('Check duplicates:')
print(comp_disc.shape[0]/comp_disc.drop_duplicates().shape[0])
comp_disc = comp_disc.drop_duplicates()
print('Check duplicates:')
print(comp_disc.shape[0]/comp_disc.drop_duplicates().shape[0])

print('comp_teachers table:')
print('Check duplicates:')
print(comp_teachers.shape[0]/comp_teachers.drop_duplicates().shape[0])
comp_teachers = comp_teachers.drop_duplicates()
print('Check duplicates:')
print(comp_teachers.shape[0]/comp_teachers.drop_duplicates().shape[0])

comp_disc = comp_disc.rename(columns={'PLAN_ID':'MAIN_PLAN'})
comp_disc_teachers = comp_disc.merge(comp_teachers,on=['MAIN_PLAN', 'DISC_ID', 'SEMESTER'],how='outer')
print('comp_disc_teachers shape = ',comp_disc_teachers.shape)

print('Successfully!')
#-----------------------------------------------------------------------------------------------------------------------
print('Preparing target...')

train_dataset = train[train['ST_YEAR'].isin(['2018','2019','2020'])][['ISU', 'ST_YEAR', 'SEMESTER', 'DISC_ID', 'TYPE_NAME', 'DEBT']].drop_duplicates()
# test_dataset = train[train['ST_YEAR'].isin(['2019','2020'])][['ISU', 'ST_YEAR', 'SEMESTER','DISC_ID', 'TYPE_NAME', 'DEBT']].drop_duplicates()

isu_mark_history_train = []

for st_year in train_dataset['ST_YEAR'].unique():
    for semester in train_dataset['SEMESTER'].unique():
        data_temp = (
            train_dataset
                .drop('DISC_ID', axis=1)
            [(train_dataset['ST_YEAR'] < st_year) & (train_dataset['SEMESTER'] < semester)]
                .groupby(['ISU', 'TYPE_NAME'], as_index=False)
                .agg(DEBT_MEAN=('DEBT', 'mean'), DEBT_SUM=('DEBT', 'sum'), DEBT_COUNT=('DEBT', 'count')
                     )
        )
        data_temp['ST_YEAR'] = st_year
        data_temp['SEMESTER'] = semester

        isu_mark_history_train.append(data_temp)

isu_mark_history_train = pd.concat(isu_mark_history_train)

students_mark_history_train = []

for st_year in train_dataset['ST_YEAR'].unique():
    for semester in train_dataset['SEMESTER'].unique():
        data_temp = (
            train_dataset
                .drop('ISU', axis=1)
            [(train_dataset['ST_YEAR'] < st_year) & (train_dataset['SEMESTER'] < semester)]
                .groupby(['DISC_ID', 'TYPE_NAME'], as_index=False)
                .agg(DISC_DEBT_MEAN=('DEBT', 'mean'), DISC_DEBT_SUM=('DEBT', 'sum'), DISC_DEBT_COUNT=('DEBT', 'count')
                     )
        )
        data_temp['ST_YEAR'] = st_year
        data_temp['SEMESTER'] = semester

        students_mark_history_train.append(data_temp)

students_mark_history_train = pd.concat(students_mark_history_train)

# isu_mark_history_train = train_dataset[train_dataset['ST_YEAR'].isin(['2018'])].groupby(by=['ISU', 'TYPE_NAME'])['DEBT'].agg({'median'})
# students_mark_history_train = train_dataset[train_dataset['ST_YEAR'].isin(['2018'])].groupby(by=['DISC_ID'])['DEBT'].agg({'mean','std'})
# isu_mark_history_train = isu_mark_history_train.reset_index().rename({'median':'debt_hist'},axis=1)
# students_mark_history_train = students_mark_history_train.reset_index().rename({'std':'students_debt_hist_std','mean':'students_debt_hist_mean'},axis=1)

train_dataset = train_dataset.merge(isu_mark_history_train,on=['ISU', 'ST_YEAR', 'SEMESTER', 'TYPE_NAME'],how='left')
train_dataset = train_dataset.merge(students_mark_history_train,on=['DISC_ID', 'ST_YEAR', 'SEMESTER', 'TYPE_NAME'],how='left')

train_dataset['ST_YEAR'].value_counts()

print('train target shape = ', train_dataset.shape)
print(train_dataset.info())

test_dataset = train_dataset[train_dataset['ST_YEAR'] == '2020']
print(test_dataset['ST_YEAR'].value_counts())

train_dataset = train_dataset[train_dataset['ST_YEAR'].isin(['2018','2019'])]

# isu_mark_history_test = train[train['ST_YEAR'].isin(['2019'])].groupby(by=['ISU', 'TYPE_NAME'])['DEBT'].agg({'median'})
# students_mark_history_test = train[train['ST_YEAR'].isin(['2019'])].groupby(by=['DISC_ID'])['DEBT'].agg({'mean','std'})
# isu_mark_history_test = isu_mark_history_test.reset_index().rename({'median':'debt_hist'},axis=1)
# students_mark_history_test = students_mark_history_test.reset_index().rename({'std':'students_debt_hist_std','mean':'students_debt_hist_mean'},axis=1)
# test_dataset = test_dataset.merge(students_mark_history_test,on=['DISC_ID'],how='left')
# test_dataset = test_dataset.merge(isu_mark_history_test,on=['ISU', 'TYPE_NAME'],how='left')
#
# print('test target shape = ', test_dataset.shape)
# print(test_dataset.info())
#
# train_dataset['debt_hist'] = train_dataset['debt_hist'].fillna(value=train_dataset['debt_hist'].median())
# train_dataset['students_debt_hist_std'] = train_dataset['students_debt_hist_std'].fillna(value=train_dataset['students_debt_hist_std'].median())
# train_dataset['students_debt_hist_mean'] = train_dataset['students_debt_hist_mean'].fillna(value=train_dataset['students_debt_hist_mean'].median())
#
# test_dataset['debt_hist'] = test_dataset['debt_hist'].fillna(value=test_dataset['debt_hist'].median())
# test_dataset['students_debt_hist_std'] = test_dataset['students_debt_hist_std'].fillna(value=test_dataset['students_debt_hist_std'].median())
# test_dataset['students_debt_hist_mean'] = test_dataset['students_debt_hist_mean'].fillna(value=test_dataset['students_debt_hist_mean'].median())

cols_agg = ['DEBT_MEAN', 'DEBT_SUM', 'DEBT_COUNT', 'DISC_DEBT_MEAN',
       'DISC_DEBT_SUM', 'DISC_DEBT_COUNT']

for col in cols_agg:
    train_dataset[col] = train_dataset[col].fillna(value=train_dataset[col].median())
    test_dataset[col] = test_dataset[col].fillna(value=test_dataset[col].median())

print('comp_disc_teachers table:')
print('Check duplicates by DISC_ID:')
print(comp_disc_teachers['DISC_ID'].unique().shape[0]/comp_disc_teachers.drop_duplicates().shape[0])

comp_disc_teachers_552619236026332123 = comp_disc_teachers[comp_disc_teachers['DISC_ID']=='552619236026332123']

comp_disc_teachers = comp_disc_teachers[['DISC_ID', 'ST_YEAR', 'SEMESTER','CHOICE', 'DISC_NAME',
                                           'KEYWORD_NAMES', 'GENDER', 'DATE_BIRTH',
                                           'TYPE_NAME', 'MARK']]
comp_disc_teachers = comp_disc_teachers.dropna()
print(comp_disc_teachers.dtypes)

comp_disc_popularity = comp_disc_teachers.groupby(by=
                                                  ['DISC_ID', 'TYPE_NAME', 'ST_YEAR', 'SEMESTER'])[['CHOICE', 'DISC_NAME',
                                                                             'KEYWORD_NAMES', 'GENDER', 'DATE_BIRTH',
                                                                              'MARK']].agg({'CHOICE':'count',
                                                                                            'DISC_NAME':'max',
                                                                                            'KEYWORD_NAMES':'max',
                                                                                            'GENDER':'max',
                                                                                            'DATE_BIRTH':'max',
                                                                                            'MARK':['std','mean']})

comp_disc_popularity = comp_disc_popularity.reset_index()
comp_disc_popularity.columns = comp_disc_popularity.columns.droplevel(1)
comp_disc_popularity.columns.values[10] = "MARK_MEAN"
comp_disc_popularity.columns.values[9] = "MARK_STD"

comp_disc_popularity['MARK_STD'] = comp_disc_popularity['MARK_STD'].fillna(comp_disc_popularity['MARK_STD'].dropna().median())
comp_disc_popularity['AGE'] = 2022-comp_disc_popularity['DATE_BIRTH'].astype(int)
comp_disc_popularity = comp_disc_popularity.drop(['DATE_BIRTH'],axis=1)

comp_disc_popularity_train = comp_disc_popularity[comp_disc_popularity['ST_YEAR']=='2018/2019'].drop(['ST_YEAR'],axis=1)
comp_disc_popularity_test = comp_disc_popularity[comp_disc_popularity['ST_YEAR']=='2019/2020'].drop(['ST_YEAR'],axis=1)
comp_disc_popularity_val = comp_disc_popularity[comp_disc_popularity['ST_YEAR']=='2020/2021'].drop(['ST_YEAR'],axis=1)

train_dataset = train_dataset.merge(comp_disc_popularity_train,on=['DISC_ID', 'TYPE_NAME'],how='left')
test_dataset = test_dataset.merge(comp_disc_popularity_test,on=['DISC_ID', 'TYPE_NAME'],how='left')

print('Successfully!')
#-----------------------------------------------------------------------------------------------------------------------
print('preprocessing validation dataset...')

all_st_df_test = []

for st_year in train['ST_YEAR'].unique():
    for semester in train['SEMESTER'].unique():
        data_temp = (
            train
                .drop('DISC_ID', axis=1)
            [(train['ST_YEAR'] <= st_year) & (train['SEMESTER'] <= semester)]
                .groupby(['ISU', 'TYPE_NAME'], as_index=False)
                .agg(DEBT_MEAN=('DEBT', 'mean'), DEBT_SUM=('DEBT', 'sum'), DEBT_COUNT=('DEBT', 'count')
                     )
        )
        data_temp['ST_YEAR'] = str(int(st_year) + 1)
        data_temp['SEMESTER'] = str(int(semester) + 1)

        all_st_df_test.append(data_temp)

all_disc_df_test = []

for st_year in train['ST_YEAR'].unique():
    for semester in train['SEMESTER'].unique():
        data_temp = (
            train
                .drop('ISU', axis=1)
            [(train['ST_YEAR'] <= st_year) & (train['SEMESTER'] <= semester)]
                .groupby(['DISC_ID', 'TYPE_NAME'], as_index=False)
                .agg(DISC_DEBT_MEAN=('DEBT', 'mean'), DISC_DEBT_SUM=('DEBT', 'sum'), DISC_DEBT_COUNT=('DEBT', 'count')
                     )
        )
        data_temp['ST_YEAR'] = str(int(st_year) + 1)
        data_temp['SEMESTER'] = str(int(semester) + 1)

        all_disc_df_test.append(data_temp)

all_st_df_test = pd.concat(all_st_df_test)
all_disc_df_test = pd.concat(all_disc_df_test)

val_dataset = test.merge(all_st_df_test, on=['ISU', 'SEMESTER', 'ST_YEAR', 'TYPE_NAME'], how='left')
val_dataset = val_dataset.merge(all_disc_df_test, on=['DISC_ID', 'SEMESTER', 'ST_YEAR', 'TYPE_NAME'], how='left')

cols_agg = ['DEBT_MEAN', 'DEBT_SUM', 'DEBT_COUNT', 'DISC_DEBT_MEAN',
       'DISC_DEBT_SUM', 'DISC_DEBT_COUNT']

for col in cols_agg:
    val_dataset[col] = val_dataset[col].fillna(value=val_dataset[col].median())
    val_dataset[col] = val_dataset[col].fillna(value=val_dataset[col].median())

val_dataset = val_dataset.merge(comp_disc_popularity_val,on=['DISC_ID', 'TYPE_NAME'],how='left')

val_dataset = val_dataset.drop_duplicates()

print('val_dataset  shape = ', val_dataset.shape)
print(val_dataset.info())

#
# isu_mark_history_val = train[train['ST_YEAR'].isin(['2020'])].groupby(by=['ISU', 'TYPE_NAME'])['DEBT'].agg({'median'})
# students_mark_history_val = train[train['ST_YEAR'].isin(['2020'])].groupby(by=['DISC_ID'])['DEBT'].agg({'mean','std'})
# isu_mark_history_val = isu_mark_history_val.reset_index().rename({'median':'debt_hist'},axis=1)
# students_mark_history_val = students_mark_history_val.reset_index().rename({'std':'students_debt_hist_std','mean':'students_debt_hist_mean'},axis=1)
# val_dataset = test[['ISU', 'ST_YEAR', 'DISC_ID', 'TYPE_NAME']].drop_duplicates()
# val_dataset = val_dataset.merge(students_mark_history_val,on=['DISC_ID'],how='left')
# val_dataset = val_dataset.merge(isu_mark_history_val,on=['ISU', 'TYPE_NAME'],how='left')
# val_dataset = val_dataset.merge(comp_disc_popularity_val,on=['DISC_ID', 'TYPE_NAME'],how='left')



comp_portrait = comp_portrait.rename({'GENDER':'STUDENT_GENDER'},axis=1)

print(train_dataset.shape,train_dataset.shape,val_dataset.shape)
train_dataset = train_dataset.merge(comp_portrait,on='ISU',how='left')
test_dataset = test_dataset.merge(comp_portrait,on='ISU',how='left')
val_dataset = val_dataset.merge(comp_portrait,on='ISU',how='left')
print(train_dataset.shape,test_dataset.shape,val_dataset.shape)

train_dataset = train_dataset.drop(['SEMESTER_y'],axis=1).rename(columns={'SEMESTER_x':'SEMESTER'})
test_dataset = test_dataset.drop(['SEMESTER_y'],axis=1).rename(columns={'SEMESTER_x':'SEMESTER'})
val_dataset = val_dataset.drop(['SEMESTER_y'],axis=1).rename(columns={'SEMESTER_x':'SEMESTER'})

features_cols = ['ISU', 'ST_YEAR', 'SEMESTER', 'DISC_ID', 'TYPE_NAME', 'DEBT',
       'DEBT_MEAN', 'DEBT_SUM', 'DEBT_COUNT', 'DISC_DEBT_MEAN',
       'DISC_DEBT_SUM', 'DISC_DEBT_COUNT', 'CHOICE', 'DISC_NAME',
       'KEYWORD_NAMES', 'GENDER', 'MARK_STD', 'MARK_MEAN', 'AGE',
       'STUDENT_GENDER', 'CITIZENSHIP', 'EXAM_TYPE', 'EXAM_SUBJECT_1',
       'EXAM_SUBJECT_2', 'EXAM_SUBJECT_3', 'ADMITTED_EXAM_1',
       'ADMITTED_EXAM_2', 'ADMITTED_EXAM_3', 'ADMITTED_SUBJECT_PRIZE_LEVEL',
       'REGION_ID']

features_cols_val = ['ISU', 'ST_YEAR', 'SEMESTER', 'DISC_ID', 'TYPE_NAME',
       'DEBT_MEAN', 'DEBT_SUM', 'DEBT_COUNT', 'DISC_DEBT_MEAN',
       'DISC_DEBT_SUM', 'DISC_DEBT_COUNT', 'CHOICE', 'DISC_NAME',
       'KEYWORD_NAMES', 'GENDER', 'MARK_STD', 'MARK_MEAN', 'AGE',
       'STUDENT_GENDER', 'CITIZENSHIP', 'EXAM_TYPE', 'EXAM_SUBJECT_1',
       'EXAM_SUBJECT_2', 'EXAM_SUBJECT_3', 'ADMITTED_EXAM_1',
       'ADMITTED_EXAM_2', 'ADMITTED_EXAM_3', 'ADMITTED_SUBJECT_PRIZE_LEVEL',
       'REGION_ID']

train_dataset = train_dataset[features_cols]
test_dataset = test_dataset[features_cols]
val_dataset = val_dataset[features_cols_val]

joblib.dump(train_dataset, ROOT_DIR + '/samples/' + 'train.pickle')
joblib.dump(test_dataset, ROOT_DIR + '/samples/' + 'test.pickle')
joblib.dump(val_dataset, ROOT_DIR + '/samples/' + 'val.pickle')

print('Successfully!')
#-----------------------------------------------------------------------------------------------------------------------
print('OneHotEncode preprocessing of categorical features...')

train_dataset = joblib.load(ROOT_DIR + '/samples/' + 'train.pickle')
test_dataset = joblib.load(ROOT_DIR + '/samples/' + 'test.pickle')
val_dataset = joblib.load(ROOT_DIR + '/samples/' + 'val.pickle')

train_dataset = train_dataset.drop(['KEYWORD_NAMES','REGION_ID'],axis=1)
test_dataset = test_dataset.drop(['KEYWORD_NAMES', 'REGION_ID'],axis=1)
val_dataset = val_dataset.drop(['KEYWORD_NAMES', 'REGION_ID'],axis=1)

from OhePreprocessing import OhePreprocessing

train_dataset_ohe_form, cat_dummies, train_cols_order = OhePreprocessing(dataset=train_dataset,target=True, train_bool=True,
                                                                         cat_dummies = None, train_cols_order = None)
test_dataset_ohe_form = OhePreprocessing(dataset=test_dataset,target=True, train_bool=False,
                                                                         cat_dummies = cat_dummies, train_cols_order = train_cols_order)
val_dataset_ohe_form = OhePreprocessing(dataset=val_dataset,target=False, train_bool=False,
                                                                         cat_dummies = cat_dummies, train_cols_order = train_cols_order)


cols_agg = ['MARK_STD', 'MARK_MEAN', 'AGE', 'CHOICE', 'ADMITTED_EXAM_1', 'ADMITTED_EXAM_2', 'ADMITTED_EXAM_3']

for col in cols_agg:
    train_dataset_ohe_form[col] = train_dataset_ohe_form[col].fillna(value=train_dataset_ohe_form[col].median())
    test_dataset_ohe_form[col] = test_dataset_ohe_form[col].fillna(value=test_dataset_ohe_form[col].median())
    val_dataset_ohe_form[col] = val_dataset_ohe_form[col].fillna(value=val_dataset_ohe_form[col].median())


joblib.dump(train_dataset_ohe_form, ROOT_DIR + '/samples/' + 'train_ohe.pickle')
joblib.dump(test_dataset_ohe_form, ROOT_DIR + '/samples/' + 'test_ohe.pickle')
joblib.dump(val_dataset_ohe_form, ROOT_DIR + '/samples/' + 'val_ohe.pickle')

print('Successfully!')
#-----------------------------------------------------------------------------------------------------------------------

