#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
import collections

import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

#missing value computation
def cal_missing_val(df):
    data_dict = {}
    for col in df.columns:
        data_dict[col] = (df[col].isnull().sum()/df.shape[0])*100
    return pd.DataFrame.from_dict(data_dict, orient='index', columns=['MissingValueInPercentage'])

def get_most_frequent(source_type, data, match_col):
    filled_val = data[data['icu_type'] ==source_type['icu_type']][match_col].value_counts().idxmax()
    return (filled_val)

def get_arf(arf_val):
    if arf_val>2.5:
        return 1
    return 0

def category_clenaing(data):
    admit_source = {'Other ICU': 'ICU', 'ICU to SDU':'Step-Down Unit (SDU)', 'Other Hospital':'Other', 'Recovery Room':'Observation', 
                        'Acute Care/Floor' :'Acute Care'}
    data['hospital_admit_source'].replace(admit_source, inplace=True)
    icu_types = {'Med-Surg ICU':'MSICU', 'Neuro ICU':'NSICU', 'Cardiac ICU':'CCU', 'CCU-CTICU':'CTICU'}
    data['icu_type'].replace(icu_types,inplace=True)
    
    return (data)

def imputation_data(data):
    #demographic
    data['age'].fillna(np.nanmedian(data['age']), inplace=True)
    data['weight'].fillna(np.nanmedian(data['weight']), inplace=True)
    data['ethnicity'].fillna('Other/Unknown', inplace=True)
    data['gender'].fillna(random.choice(['M', 'F']), inplace=True)
    data['bmi'].fillna(np.nanmedian(data['bmi']), inplace=True)
    data['icu_admit_source']     = data[['icu_admit_source', 'icu_type']].apply(lambda x: get_most_frequent(x, data, 'icu_admit_source') if pd.isnull(x['icu_admit_source']) else x['icu_admit_source'], axis=1)
    #convert weight and age to categories
    bins_wt  = [35, 50, 65, 80, 95, np.inf]
    name_wt  = ['35-50', '50-65', '65-80', '80-95', '95+']
    data['weight_cat'] = pd.cut(data['weight'], bins_wt, labels=name_wt)
    bins_age = [15, 30, 45, 55, 65, 75, 85, np.inf]
    name_age = ['15-30', '30-45', '45-55', '55-65', '65-75','75-85','85+']
    data['age_cat'] = pd.cut(data['age'], bins_age, labels=name_age)
    
    #apache
    data['apache_3j_diagnosis']  = data[['apache_3j_diagnosis', 'icu_type']].apply(lambda x: get_most_frequent(x, data,'apache_3j_diagnosis') if pd.isnull(x['apache_3j_diagnosis']) else x['apache_3j_diagnosis'], axis=1)
    data['apache_3j_bodysystem'] = data[['apache_3j_bodysystem', 'icu_type']].apply(lambda x: get_most_frequent(x,data, 'apache_3j_bodysystem') if pd.isnull(x['apache_3j_bodysystem']) else x['apache_3j_bodysystem'], axis=1)
    data['arf_apache']           = data[['d1_creatinine_max', 'arf_apache']].apply(lambda x: get_arf(x['d1_creatinine_max']) if pd.isnull(x['arf_apache']) else x['arf_apache'], axis=1)
    data['intubated_apache'].fillna(np.random.choice([1, 0], p=[0.15, 0.85]), inplace=True)
    data['ventilated_apache'].fillna(np.random.choice([1, 0], p=[0.35, 0.65]), inplace=True)
    
    #comorbidity
    data['aids'].fillna(np.random.choice([1, 0], p=[0.01, 0.99]), inplace=True)
    data['cirrhosis'].fillna(np.random.choice([1, 0], p=[0.01, 0.99]), inplace=True)
    data['diabetes_mellitus'].fillna(np.random.choice([1, 0], p=[0.22, 0.78]), inplace=True)
    data['hepatic_failure'].fillna(np.random.choice([1, 0], p=[0.01, 0.99]), inplace=True)
    data['immunosuppression'].fillna(np.random.choice([1, 0], p=[0.02, 0.98]), inplace=True)
    data['leukemia'].fillna(np.random.choice([1, 0], p=[0.01, 0.99]), inplace=True)
    data['lymphoma'].fillna(np.random.choice([1, 0], p=[0.01, 0.99]), inplace=True)
    data['solid_tumor_with_metastasis'].fillna(np.random.choice([1, 0], p=[0.02, 0.98]), inplace=True)
    
    #group_cat
    group_cat = ['age_cat', 'weight_cat','gender', 'icu_type','apache_3j_bodysystem']
    
    #gcs imputation
    gcs_map_cols = ['gcs_eyes_apache','gcs_motor_apache','gcs_verbal_apache', 'map_apache']
    
    for gcs in gcs_map_cols:
        #1st get missing values filled from similar patients by groupby
        data[gcs] = data.groupby(group_cat)[gcs].transform(lambda x: x.fillna(np.nanmedian(x)))
        
        #2nd get missing values filled by median
        data[gcs].fillna(np.nanmedian(data[gcs]), inplace=True)
    
    data['gcs_total'] = data[['gcs_eyes_apache','gcs_motor_apache','gcs_verbal_apache']].sum(axis=1)
    
    #lab columns with/without apache
    lab_apache_cols = ['albumin', 'creatinine', 'bun', 'glucose', 'heartrate', 'hematocrit', 'arterial_po2', 'spo2', 'sodium', 'calcium',
                      'temp','wbc', 'platelets','potassium','hemaglobin','hco3', 'inr']
    
    for lab_a in lab_apache_cols:
        #1st get missing values filled from similar columns
        data.loc[data['d1_'+lab_a+'_max'].isnull(),'d1_'+lab_a+'_max'] = data['h1_'+lab_a+'_max']
        data.loc[data['d1_'+lab_a+'_min'].isnull(),'d1_'+lab_a+'_min'] = data['h1_'+lab_a+'_min']
        
        if (lab_a+'_apache' in data.columns):
            data.loc[data['d1_'+lab_a+'_max'].isnull(),'d1_'+lab_a+'_max'] = data[lab_a+'_apache']
            data.loc[data['d1_'+lab_a+'_min'].isnull(),'d1_'+lab_a+'_min'] = data[lab_a+'_apache']
        
        data[lab_a] = (data['d1_'+lab_a+'_max'] + data['d1_'+lab_a+'_min'])/2
    
        #2nd get missing values filled from similar patients by groupby
        data[lab_a] = data.groupby(group_cat)[lab_a].transform(lambda x: x.fillna(np.nanmedian(x)))
        #3rd get missing values filled by median
        data[lab_a].fillna(np.nanmedian(data[lab_a]), inplace=True)
        
    #vitals
    vitals_cols = ['diasbp', 'mbp', 'sysbp']
    for vit_c in vitals_cols:
        #1st get missing values filled from similar columns - invasive/noninvasive min and max
        data.loc[data['d1_'+vit_c+'_max'].isnull(),'d1_'+vit_c+'_max'] = data['d1_'+vit_c+'_invasive_max']
        data.loc[data['d1_'+vit_c+'_max'].isnull(),'d1_'+vit_c+'_max'] = data['d1_'+vit_c+'_noninvasive_max']
        
        data.loc[data['d1_'+vit_c+'_min'].isnull(),'d1_'+vit_c+'_min'] = data['d1_'+vit_c+'_invasive_min']
        data.loc[data['d1_'+vit_c+'_min'].isnull(),'d1_'+vit_c+'_min'] = data['d1_'+vit_c+'_noninvasive_min']
        
        data.loc[data['h1_'+vit_c+'_max'].isnull(),'h1_'+vit_c+'_max'] = data['h1_'+vit_c+'_invasive_max']
        data.loc[data['h1_'+vit_c+'_max'].isnull(),'h1_'+vit_c+'_max'] = data['h1_'+vit_c+'_noninvasive_max']
        
        data.loc[data['h1_'+vit_c+'_min'].isnull(),'h1_'+vit_c+'_min'] = data['h1_'+vit_c+'_invasive_min']
        data.loc[data['h1_'+vit_c+'_min'].isnull(),'h1_'+vit_c+'_min'] = data['h1_'+vit_c+'_noninvasive_min']
    
        #2nd get missing values filled from similar patients by groupby
        sub_vitals = ['d1_'+vit_c+'_max','d1_'+vit_c+'_min', 'h1_'+vit_c+'_max','h1_'+vit_c+'_min']
        for sub_vit_c in sub_vitals:
            data[sub_vit_c] = data.groupby(group_cat)[sub_vit_c].transform(lambda x: x.fillna(np.nanmedian(x)))
            #3rd get missing values filled by median
            data[sub_vit_c].fillna(np.nanmedian(data[sub_vit_c]), inplace=True)
      
    data['d1_resprate_max'] = data['d1_resprate_max'].fillna(np.nanmedian(data['d1_resprate_max']))
    data['d1_resprate_min'] = data['d1_resprate_min'].fillna(np.nanmedian(data['d1_resprate_min']))
    
    # death prob values
    data['apache_4a_hospital_death_prob'] = data.groupby(group_cat)['apache_4a_hospital_death_prob'].transform(lambda x: x.fillna(np.nanmean(x)))
    data['apache_4a_icu_death_prob']      = data.groupby(group_cat)['apache_4a_icu_death_prob'].transform(lambda x: x.fillna(np.nanmean(x)))
    data['apache_4a_hospital_death_prob'].fillna(np.nanmean(data['apache_4a_hospital_death_prob']), inplace=True)
    data['apache_4a_icu_death_prob'].fillna(np.nanmean(data['apache_4a_icu_death_prob']), inplace=True)
    
    return (data)

def get_ordinal_cat(cat_columns, df):
    for cat_col in cat_columns:
        cat_dict = dict(df[cat_col].value_counts())
        map_dict ={}
        sorted_dict = collections.OrderedDict(cat_dict)
        for idx, val in enumerate(sorted_dict):
            map_dict[val] = idx
        df[cat_col] = df[cat_col].map(map_dict)
    return df

def enc_data(data):
    label_encoding   = ['gender']
    ordinal_encoding = ['icu_admit_source', 'icu_stay_type', 'icu_type','apache_3j_diagnosis','apache_3j_bodysystem', 'ethnicity']
    # creating instance of labelencoder
    labelencoder    = LabelEncoder()
    data['gender']  = labelencoder.fit_transform(data['gender'])
    data = get_ordinal_cat(ordinal_encoding, data)
    data = pd.get_dummies(data, columns=['calc_hospital_death_prob_cls','calc_hospital_median_cls', 'calc_icu_death_prob_cls','calc_icu_median_cls'])
    return (data)

def engineered_data(data):
    data['Change_diagnosis'] = 0
    body_2 = {'Neurologic': 'Neurological','Renal/Genitourinary': 'Genitourinary', 'Haematologic':'Hematological','Undefined Diagnoses':
         'Undefined diagnoses'}
    data['apache_2_bodysystem'].replace(body_2, inplace=True)
    data.loc[data['apache_2_bodysystem'] != data['apache_3j_bodysystem'], 'Change_diagnosis'] = 1 
    
    return (data)

def get_gr_prob(val):
    arr = np.array(val)
    death_prob = sum((arr>0.5).astype(int))/len(arr)
    return (death_prob)

def get_gr_median(val):
    arr = np.array(val)
    med_ = np.nanmedian(arr)
    return (med_)

def hospital_features(data):
    class_labels = ['T1','T2','T3','T4','T5']
    data['calc_hospital_death_prob'] = data.groupby('hospital_id')['apache_4a_hospital_death_prob'].transform(lambda x: get_gr_prob(x))
    data['calc_hospital_median']     = data.groupby('hospital_id')['apache_4a_hospital_death_prob'].transform(lambda x: get_gr_median(x))
    data['calc_icu_death_prob']      = data.groupby('hospital_id')['apache_4a_icu_death_prob'].transform(lambda x: get_gr_prob(x))
    data['calc_icu_median']          = data.groupby('hospital_id')['apache_4a_icu_death_prob'].transform(lambda x: get_gr_median(x))
    data['calc_hospital_death_prob_cls']= pd.cut(np.array(data['calc_hospital_death_prob']), 5, labels=class_labels)
    data['calc_hospital_median_cls']    = pd.cut(np.array(data['calc_hospital_median']), 5, labels=class_labels)
    data['calc_icu_death_prob_cls']     = pd.cut(np.array(data['calc_icu_death_prob']), 5, labels=class_labels)
    data['calc_icu_median_cls']         = pd.cut(np.array(data['calc_icu_median']), 5, labels=class_labels)
    return (data)

def run_lgb(train_X, train_y, val_X, val_y, test_X, train_all, cat_features):
    params = {
        "objective" : "binary",
        "metric" : "binary_logloss", #auc
        "num_leaves" : 70,
        "learning_rate" : 0.04, #0.004
        "bagging_fraction" : 0.6,
        "feature_fraction" : 0.6,
        "bagging_frequency" : 6,
        "bagging_seed" : 42,
        "verbosity" : -1,
        "seed": 101,
        'boosting_type' : 'gbdt', #'dart'
        'scoring': 'roc_auc',
        'is_unbalance': True
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y, categorical_feature = cat_features)
    lgval = lgb.Dataset(val_X, label=val_y, categorical_feature = cat_features, reference=lgtrain)
    evals_result = {}
    model = lgb.train(params, lgtrain, 5000, 
                      valid_sets=[lgtrain, lgval], 
                      early_stopping_rounds=1000, 
                      verbose_eval=150, 
                      evals_result=evals_result)
    
    pred_test_y  = model.predict(test_X, num_iteration=model.best_iteration)
    pred_train_y = model.predict(train_all, num_iteration=model.best_iteration)
    return pred_test_y, pred_train_y, model, evals_result
