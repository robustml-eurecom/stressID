import numpy as np
import scipy.stats as stats

import pandas as pd
import random
import time 

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, LeaveOneOut, GroupKFold
from sklearn.svm import LinearSVC, SVR
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.metrics import balanced_accuracy_score, f1_score

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

from sklearn.utils import shuffle




# Compare several classification models over K repetition, using K group splits, grouped by subjects
def make_nclassif(X, y, n_splits=10, feature_selector=None, list_classifiers=None, impute=True, scale=True, verbose=True):
    # Dictionnary to store f1-score and accuracy
    df_res= pd.DataFrame({'n':[],'f1-score':[],'accuracy':[], 'classifier':[]})
    conf_matrices = []
    
    if impute:
        imputer = IterativeImputer(random_state=0)
    else:
        imputer = None
        
    if scale:
        scaler = StandardScaler()
    else:
        scaler = None
    
    
    # Defaut classifiers tested: Logistic regression, Random Forests, Adaboost
    if not list_classifiers :
        list_classifiers = [LogisticRegression(max_iter=2000),
                            RandomForestClassifier(max_depth=5),
                            AdaBoostClassifier(n_estimators=100)]
                       
            
    # Make n-group random splits grouped by subjects
    groups = [l.split('_')[0] for l in list(y.index)]
    
    #rstate = random.randint(0,100)
    
    X_shuffled, y_shuffled, groups_shuffled = shuffle(X, y, groups)

    group_kfold = GroupKFold(n_splits=n_splits)
    group_kfold.get_n_splits(X_shuffled, y_shuffled, groups_shuffled)

    # SPLITS
    for s, (train_index, test_index) in enumerate(group_kfold.split(X, y, groups)):
        x_train = X.iloc[train_index]
        x_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        
        if verbose:
        	print('Split {0:2d}/{1:2d}'.format(s+1, n_splits))
        
        # Fit each model
        for model in list_classifiers:

            if feature_selector == 'l1':  
                clf = Pipeline([
                    ('impute',imputer), 
                    ('scale', scaler), 
                    ('feature_selection', SelectFromModel(LogisticRegression(max_iter=5000,
                                                                             C=0.1, 
                                                                             penalty="l1", 
                                                                             dual=False, 
                                                                             solver='saga'))),
                    ('classification', model)
                ])
            elif feature_selector == 'RFECV':
                clf = Pipeline([
                    ('impute',imputer), 
                    ('scale', scaler), 
                    ('feature_selection', RFECV(RandomForestClassifier(max_depth=5),
                                                step=2, 
                                                cv=2)),
                    ('classification', model)
                ])
            elif feature_selector == 'RFE':
                    clf = Pipeline([
                        ('impute',imputer), 
                        ('scale', scaler), 
                        ('feature_selection', RFE(RandomForestClassifier(max_depth=5),
                                                    n_features_to_select=20, step=2)),
                        ('classification', model)
                    ])
            else:
                clf = Pipeline([
                    ('impute',imputer), 
                    ('scale', scaler), 
                    ('classification', model)
                ])
                
            tic = time.perf_counter()
            clf.fit(x_train, y_train)
            toc = time.perf_counter()
            
            # Retrieve accuracy and F1-score
            y_pred = clf.predict(x_test)
            conf_matrices.append(confusion_matrix(y_test, y_pred))
            df_res = df_res.append({'n':int(s),'f1-score':f1_score(y_test, y_pred, average='weighted'),
                                    'accuracy':balanced_accuracy_score(y_test, y_pred), 
                                    'classifier':model.__class__.__name__, 'time':toc-tic},ignore_index=True)
    
    return df_res, conf_matrices
    
   
##########################################################################################  


def avg_res(res):
    return res.groupby(['classifier']).mean()[['f1-score', 'accuracy', 'time']]
    
    


##########################################################################################    
       
def make_nclassif_random_splits(X, y, n_splits=10, feature_selector=None, list_classifiers=None, impute=True, scale=True, verbose=True):
    # Dictionnary to store f1-score and accuracy
    df_res= pd.DataFrame({'n':[],'f1-score':[],'accuracy':[], 'classifier':[]})
    
    if impute:
        imputer = IterativeImputer()
    else:
        imputer = None
        
    if scale:
        scaler = StandardScaler()
    else:
        scaler = None
    
    
    # Defaut classifiers tested: Logistic regression, Random Forests, Adaboost
    if not list_classifiers :
        list_classifiers = [LogisticRegression(max_iter=2000),
                            RandomForestClassifier(max_depth=5),
                            AdaBoostClassifier(n_estimators=100)]
                        
    # Make n random splits 
    for s in range(n_splits):
        rstate = random.randint(0,100)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=rstate)
        
        if verbose:
            print('Split {0:2d}/{1:2d}'.format(s+1, n_splits))
        
        # Fit each model
        for model in list_classifiers:

            if feature_selector == 'l1':  
                clf = Pipeline([
                    ('impute',imputer), 
                    ('scale', scaler), 
                    ('feature_selection', SelectFromModel(LogisticRegression(max_iter=5000,
                                                                             C=0.1, 
                                                                             penalty="l1", 
                                                                             dual=False, 
                                                                             solver='saga'))),
                    ('classification', model)
                ])
            elif feature_selector == 'RFE':
                clf = Pipeline([
                    ('impute',imputer), 
                    ('scale', scaler), 
                    ('feature_selection', RFECV(RandomForestClassifier(max_depth=5),
                                                step=2, 
                                                cv=2)),
                    ('classification', model)
                ])
            else:
                clf = Pipeline([
                    ('impute',imputer), 
                    ('scale', scaler), 
                    ('classification', model)
                ])
                
            tic = time.perf_counter()
            clf.fit(x_train, y_train)
            toc = time.perf_counter()
            
            # Retrieve accuracy and F1-score
            y_pred = clf.predict(x_test)
            df_res = df_res.append({'n':int(s),'f1-score':f1_score(y_test, y_pred, average='weighted'),
                                    'accuracy':balanced_accuracy_score(y_test, y_pred), 
                                    'classifier':model.__class__.__name__, 'time':toc-tic},ignore_index=True)
    
    return df_res

