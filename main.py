import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LogisticRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, f1_score
from sklearn import datasets, metrics
#import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
#import seaborn as sns
import pickle
import os
from sklearn.preprocessing import label_binarize
import lg
lgg = lg.logg()
#%matplotlib inline

#location is save in "D:\iNeuron\Homework-MLproject\Logistic_regression_home_work"
x = pd.DataFrame()
x_scale = pd.DataFrame()
y = pd.DataFrame()
df = pd.DataFrame()
df_new = pd.DataFrame()
vif_matrix = pd.DataFrame()

class Logis:
    def __init__(self,dir):
        global nwdf,df
        try:
            lgg.info("Extraction of file started")
            os.chdir(r"D:\iNeuron\Homework-MLproject\Logistic_regression_home_work")
            if not("out.csv" in os.listdir()):
                lgg.info("new file is create to store data")
                self.curr = dir
                listofdir=['bending1', 'bending2', 'cycling', 'lying',
                           'sitting', 'standing', 'walking']
                global df
                for i in listofdir:
                    os.chdir(self.curr+"\\"+i)
                    filename=os.listdir()
                    for j in filename:
                        nwdf = pd.read_csv(self.curr+"\\"+i+"\\"+j,header=4,error_bad_lines=False)
                        nwdf['label']=i
                        nwdf.drop('# Columns: time',axis=1,inplace=True)
                        df = pd.concat([df,nwdf],ignore_index=True)
                os.chdir(self.curr)
                os.chdir(r"D:\iNeuron\Homework-MLproject\Logistic_regression_home_work")
                df.to_csv('out.csv')
            else:
                lgg.info("Old file exist, therefore taking data from it")
                df = pd.read_csv("out.csv")
                df.drop(columns=['Unnamed: 0'], inplace=True)
        except Exception as e:
            print(f'Exception occur at {i} , {j} file, the exception is {e}')
            lgg.excpt(e)
        self.fillingempty()

    def fillingempty(self):
        try:
            lgg.info('filling empty with mean')
            global df
            for i in df.columns:
                if (df[i].isnull().sum()) != 0:
                    df[i].fillna(df[i].mean(), inplace=True)
        except Exception as e:
            print(f'Exception {e} occured while filling empty feature with mean of that feature')
            lgg.info(e)
        self.extract_x_y()

    def extract_x_y(self):
        try:
            lgg.info("extract x and y from dataframe")
            global x,y
            y = df['label']
            x = df.drop(columns=['label'])
        except Exception as e:
            print(f"Exception {e} occur during extraction ")
        self.EDA_outlier()

    def EDA_outlier(self):
        global df,df_new
        try:
            lgg.info('EDA for outlier is performed')
            q = df['var_rss12'].quantile(.7)
            df_new = df[df['var_rss12'] < q]

            q = df_new['avg_rss23'].quantile(.7)
            df_new = df_new[df_new['avg_rss23'] < q]

            q = df_new['var_rss23'].quantile(.7)
            df_new = df_new[df_new['var_rss23'] < q]

            q = df_new['var_rss13'].quantile(.7)
            df_new = df_new[df_new['var_rss13'] < q]
        except Exception as e:
            print(f'Exception {e} occur during EDA part')
        self.EDA_scalar()

    def EDA_scalar(self):
        try:
            lgg.info('EDA scaling of x is done')
            global x,y,x_scale
            y = df_new["label"]
            x = df_new.drop(columns=['label'])
            scalar = StandardScaler()
            scalar.fit(x)
            x_scale = (scalar.transform(x))
        except Exception as e:
            print(f'Exception {e} occur during scaling x')
        self.vif_score()

    def vif_score(self):
        try:
            global vif_matrix
            lgg.info('vif_score was started')
            global x
            scaler = StandardScaler()
            arr = scaler.fit_transform(x)
            vif_matrix = pd.DataFrame([[x.columns[i],
                                  variance_inflation_factor(arr, i)] for i in range(arr.shape[1])],
                                columns=["FEATURE", "VIF_SCORE"])
        except Exception as e:
            print(f'Exception e occur during vif_score i.e., {e}')
        self.test_train_splt()


    def test_train_splt(self):
        try:
            lgg.info('test train split is started')
            global x_scale
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_scale,
                                                                                    y, test_size=0.33,
                                                                                    random_state=42)
            self.y_nw = label_binarize(self.y_test, classes=['bending1', 'bending2', 'cycling', 'lying',
                                                   'sitting', 'standing', 'walking'])
            self.n_classes = self.y_nw.shape[1]
            lgg.info('test train split is completed')
        except Exception as e:
            print(f"exception occur during train_test_splt {e}")
        self.fit_evaluate_models()

    def fit_evaluate_models(self):
        try:
            lgg.info('fit_evaluation of model is started')
            auc_macro_dict = {}
            accuracy = {}
            f1_s = {}
            solver = ["lbfgs", "sag", "saga", "newton-cg"]
            for i in range(len(solver)):
                lr = LogisticRegression(multi_class='multinomial', solver=solver[i])
                lr.fit(self.x_train, self.y_train)
                self.y_pred = lr.predict(self.x_test)
                self.y_pred_prob = lr.predict_proba(self.x_test)
                val = roc_auc_score(self.y_nw, self.y_pred_prob, average='macro')
                auc_macro_dict[solver[i]] = val
                accuracy[solver[i]] = accuracy_score(self.y_test, self.y_pred)
                f1_s[solver[i]] = f1_score(self.y_test, self.y_pred, average='macro')
            for i in range(len(solver)):
                print(f"For {solver[i]} followings are the matrics ")
                print(f'AUC values after macro average = {round(auc_macro_dict[solver[i]], 3)}')
                print(f'accuracy = {round(accuracy[solver[i]], 3)}')
                print(f'f1 score is {round(f1_s[solver[i]], 3)}')
                print("*******************************")
            lgg.info('fit_evaluation of model is completed')
        except Exception as e:
            print(f"exception {e} occur during fit_evalation of different model")

lgr = Logis("D:\iNeuron\Homework-MLproject\Logistic_regression_home_work")
print("All model preformed equally, so any model will work fine")