import os
import xport

import numpy as np
import pandas as pd

def merge_xpt(fname):
    if type(fname) == list:
        df = []
        for f in fname:
            with open(f, 'rb') as file:
                df.append(xport.to_dataframe(file))
        all_files = np.array(df)  # store all datasets in a np.array

        # merge data.frames
        for i in range(len(fname)-1):
            df[0] = df[0].merge(df[i+1], on=['SEQN'])
        return all_files, df[0]

    with open(fname, 'rb') as file:
        df = xport.to_dataframe(file)
    return df


def get_training_data(data_root):
    data_dir = os.path.join(data_root, "2013-2014")
    fnames = [
        os.path.join(data_dir, "questionnaire", "DIQ_H.XPT"),
        os.path.join(data_dir, "examination", "BMX_H.XPT"),
        os.path.join(data_dir, "demographics", "DEMO_H.XPT"),
        os.path.join(data_dir, "laboratory", "TCHOL_H.XPT"),
        os.path.join(data_dir, "dietary", "DR1TOT_H.XPT"),
        os.path.join(data_dir, "dietary", "DR2TOT_H.XPT"),
        os.path.join(data_dir, "questionnaire", "ALQ_H.XPT"),
        os.path.join(data_dir, "questionnaire", "SLQ_H.XPT"),
        os.path.join(data_dir, "questionnaire", "SMQ_H.XPT"),
        os.path.join(data_dir, "laboratory", "INS_H.XPT"),
        os.path.join(data_dir, "laboratory", "OGTT_H.XPT")
    ]

    files, merged = merge_xpt(fnames)
    df = merged[[
        'DIQ010', 'RIDAGEYR', 'LBDINSI', 'RIDRETH3',
        'LBXTC', 'DR1TSUGR', 'DR1TCARB', 'DR2TSUGR',
        'DR2TCARB', 'ALQ120Q', 'SLD010H'
    ]].rename(index=str, columns={'DIQ010': 'Diabetes',
                                  'RIDAGEYR': 'Age',
                                  'RIDRETH3': 'Race',
                                  'LBXTC': 'Cholesterol',
                                  'DR1TSUGR': 'D1Sugar',
                                  'DR1TCARB': 'D1Carb',
                                  'DR2TSUGR': 'D2Sugar',
                                  'DR2TCARB': 'D2Carb',
                                  'ALQ120Q': 'Alcohol',
                                  'SLD010H': 'SleepHours',
                                  'SMQ040': 'SmokeNow',
                                  'LBXGLT': 'Glucose2H',
                                  'LBDINSI': 'Insulin'})
    return df


def get_testing_data(data_root):
    # NOTE TO TA: Feel free to split this dataset in half to give half to
    # students and keep half as the grading dataset (if you want live
    # validation for them)
    data_dir = os.path.join(data_root, "2015-2016")

    fnames = [
        os.path.join(data_dir, "questionnaire", "DIQ_I.XPT"),
        os.path.join(data_dir, "examination", "BMX_I.XPT"),
        os.path.join(data_dir, "demographics", "DEMO_I.XPT"),
        os.path.join(data_dir, "laboratory", "TCHOL_I.XPT"),
        os.path.join(data_dir, "dietary", "DR1TOT_I.XPT"),
        os.path.join(data_dir, "dietary", "DR2TOT_I.XPT"),
        os.path.join(data_dir, "questionnaire", "ALQ_I.XPT"),
        os.path.join(data_dir, "questionnaire", "SLQ_I.XPT"),
        os.path.join(data_dir, "questionnaire", "SMQ_I.XPT"),
        os.path.join(data_dir, "laboratory", "INS_I.XPT"),
        os.path.join(data_dir, "laboratory", "OGTT_I.XPT")
        ]

    files, merged = merge_xpt(fnames)
    df = merged[['DIQ010',
                 'RIDAGEYR',
                 'LBDINSI',
                 'RIDRETH3',
                 'LBXTC',
                 'DR1TSUGR',
                 'DR1TCARB',
                 'DR2TSUGR',
                 'DR2TCARB',
                 'ALQ120Q',
                 'SLD012']].rename(index=str,
                                   columns={'DIQ010':'Diabetes',
                                             'RIDAGEYR':'Age',
                                             'RIDRETH3':'Race',
                                             'LBXTC':'Cholesterol',
                                             'DR1TSUGR':'D1Sugar',
                                             'DR1TCARB':'D1Carb',
                                             'DR2TSUGR':'D2Sugar',
                                             'DR2TCARB':'D2Carb',
                                             'ALQ120Q':'Alcohol',
                                             'SLD012':'SleepHours',
                                             'SMQ040':'SmokeNow',
                                             'LBXGLT':'Glucose2H',
                                             'LBDINSI':'Insulin'})
    #df['Race'] = df['Race'].astype('category')
    df = df.dropna()  # drop the rows with NaN
    df = df[df.Diabetes < 3]  #
    n_yes = sum(df["Diabetes"] == 1.0)
    n_no = sum(df["Diabetes"] == 2.0)

    df_c2 = df[df["Diabetes"] == 2.0].iloc[np.random.permutation(n_no)[0:n_yes]]
    df_c1 = df[df["Diabetes"] == 1.0]
    df_balanced_test = pd.concat([df_c1, df_c2], axis=0)

    return df_balanced_test.drop("Diabetes", axis=1), \
           df_balanced_test["Diabetes"].values

    # y_test = pd.get_dummies(df_balanced_test["Diabetes"]).values
