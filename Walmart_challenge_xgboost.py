import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from datetime import date
from collections import Counter

from sklearn.base import BaseEstimator
from sklearn.datasets import load_svmlight_file, dump_svmlight_file


from sklearn import cross_validation

import sys
sys.path.append('C:/Users/ze/Box Sync/kaggle/xgboost')
import xgboost as xgb
import numpy as np


def xgboost_pred(train,labels,test,test_labels,final_test):
    params = {}
    params["objective"] = "multi:softprob"
    params["eval_metric"]="mlogloss"
    params["eta"] = 0.05 #0.02 
    params["min_child_weight"] = 6
    params["subsample"] = 0.9 #although somehow values between 0.25 to 0.75 is recommended by Hastie
    params["colsample_bytree"] = 0.7
    params["scale_pos_weight"] = 1
    params["silent"] = 1
    params["max_depth"] = 8
    params["num_class"]=38
    
    plst = list(params.items())

    #Using 5000 rows for early stopping. 
    #offset = len(labels)/6
    #print offset
    num_rounds = 20000
    xgtest = xgb.DMatrix(final_test)

    xgtrain = xgb.DMatrix(train, label=labels)
    xgval = xgb.DMatrix(test, label=test_labels)
 
    watchlist = [(xgtrain, 'train'),(xgval, 'val')]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=30)
    #create a train and validation dmatrices 

    #reverse train and labels and use different 5k for early stopping. 
    # this adds very little to the score but it is an option if you are concerned about using all the data. 
#     train = train[::-1,:]
#     labels = np.log(labels[::-1])
# 
    print 'ready to generate test data'

    #combine predictions
    #since the metric only cares about relative rank we don't need to average
    return model.predict(xgtest,ntree_limit=model.best_iteration)




train=pd.read_csv('C:/Users/ze/Box Sync/kaggle/Walmart_recruiting/train/train.csv')
test=pd.read_csv('C:/Users/ze/Box Sync/kaggle/Walmart_recruiting/test/test.csv')
train_ori=pd.read_csv('C:/Users/ze/Box Sync/kaggle/Walmart_recruiting/train/train.csv')
test_ori=pd.read_csv('C:/Users/ze/Box Sync/kaggle/Walmart_recruiting/test/test.csv')
"""Drop everything except weekday,scan count, and department description """

train_departments_labels=list(Counter(train['TripType']))
"""need to convert the trip_type label to integers starting from 0 to n-1"""
label_lookup={label:train_departments_labels.index(label) for label in train_departments_labels}
label_reverse={train_departments_labels.index(label):label for label in train_departments_labels}
print label_reverse


indexed_label=train['TripType'].apply(lambda x:label_lookup[x])
indexed_label.name="Indexed_Labels"
"""used an inline function to convert the TripType to """
train=pd.concat([train,indexed_label],axis=1)

test_plchldr=test['FinelineNumber'].copy()
test_plchldr.name='Indexed_Labels'
test.insert(0,'TripType',test_plchldr)
test=pd.concat([test,test_plchldr],axis=1)
unified_data=train.append(test)
"""rather than getting rid UPC and Fineline number, let's try averaging them first
    Similarly, let's try to find a good way to incorporate scan count into 
 """
print unified_data.columns.values
NaN_cols=unified_data.isnull().any()
NaN_cols=NaN_cols.index[NaN_cols]
for nancol in NaN_cols:
    print nancol
    '''
    Need to consider not using mode, but use -2, so we can drop it later. 
    '''
    print unified_data[nancol].mode() 
    #unified_data[nancol].fillna(train[nancol].mode().ix[0],inplace=True)
    unified_data[nancol].fillna(-2,inplace=True)


unified_data_list=unified_data['FinelineNumber'].tolist()
fine_dict=dict(Counter(unified_data_list))
fine_dict={key:fine_dict[key] for key in fine_dict.keys() if fine_dict[key]>1000}
print len(fine_dict)
print unified_data['FinelineNumber'].isin(fine_dict.keys())
unified_data['FinelineNumber'][~(unified_data['FinelineNumber'].isin(fine_dict.keys()))]=-1
unified_data=pd.concat([unified_data,pd.get_dummies(unified_data['Weekday']),pd.get_dummies(unified_data['DepartmentDescription'],sparse=False),pd.get_dummies(unified_data['FinelineNumber'],sparse=False)],axis=1)
print unified_data.columns.values
unified_data=unified_data.drop(['Weekday','DepartmentDescription',-1.0,-2.0],axis=1)


grouped_data=unified_data.groupby(['VisitNumber'],sort=False)
grouped_train=train_ori.groupby(['VisitNumber'],sort=False)
"""here we apply two separate functions to the reamining features after grouping by VisitNumber
    first function is sum. This is applied to features that are part of the Department Description.
    Namely, the description is turned into a 1-of-k coding vector. So summing them produces a good representation 
    of the total departments visited on this.
    If we try the same thing for UPC number or fineline, we might be wrong, but could try the average.
    We will consider this next 
"""
func_dict={feature: np.sum for feature in unified_data.columns.values}
for feature in func_dict.keys():
    print feature
    if ((feature not in test_ori['DepartmentDescription'].tolist()) and (feature not in fine_dict.keys()) ):
        print str(feature)+' is using mode as the aggregation method'
        #func_dict[feature]=lambda x:np.argmax(np.bincount(x))
        func_dict[feature]=np.mean
    elif feature=='Upc':
        func_dict[feature]=lambda x:np.argmax(np.bincount(x))

"""The agg function of grouped pandas feature allows application of a list of functions to the data that are grouped by a specific value"""
grouped_data=grouped_data.agg(func_dict)

final_train=grouped_data.iloc[:len(grouped_train)-1]
final_test=grouped_data.iloc[len(grouped_train):]
print final_train.columns.values
print final_test.columns.values
print final_train.columns==final_test.columns


#X_train, X_test = cross_validation.train_test_split(grouped_train, test_size=0.1)
print final_train['Indexed_Labels']
ssf=cross_validation.StratifiedShuffleSplit(final_train['Indexed_Labels'],n_iter=1,test_size=0.1)
for train_idx,test_idx in ssf:
    X_train,X_test=final_train.iloc[train_idx],final_train.iloc[test_idx]
train_label=np.array(X_train['Indexed_Labels'])
test_label=np.array(X_test['Indexed_Labels'])
X_train=X_train.drop(['Indexed_Labels','TripType','VisitNumber'],axis=1)
test_np=np.array(X_test.drop(['Indexed_Labels','TripType','VisitNumber'],axis=1))
train_np=np.array(X_train)
final_test_rdy=final_test.drop(['Indexed_Labels','TripType','VisitNumber'],axis=1)
print X_train.columns==final_test_rdy.columns
for idx,col in np.ndenumerate(X_train.columns.values):
    print (col,final_test_rdy.columns.values[idx])
    
final_test_np=np.array(final_test_rdy)


pred=xgboost_pred(train_np, train_label, test_np, test_label, final_test_np)
preds_final = pd.DataFrame(pred,index=final_test['VisitNumber'])
preds_final.to_csv('xgboost_walmart_trial_unified_top_250fineline_with_fineline.csv', index=False)
