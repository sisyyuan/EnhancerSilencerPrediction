#!/usr/bin/python

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, plot_roc_curve, roc_auc_score, auc, classification_report, roc_curve, RocCurveDisplay
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.svm import SVC,LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from pathlib import Path
import joblib

def predict_test(model, X_test):
    yhat = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    return yhat, y_prob

def report(y_test, yhat):
    report = classification_report(y_test, yhat, target_names=['1', '0'], output_dict=True, labels=np.unique(yhat))
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    accuracy = report['accuracy']
    return precision, recall, f1_score, accuracy

def train_and_evaluate(kf, X, y, feature_list, classifer, name, cell):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 1000)
    df_report = pd.DataFrame(columns=['Classifer', 'Features', 'Fold', 'Precision', 'Recall', 'F1_score', 'Accuracy', 'AUC'])
    fig, ax = plt.subplots(1,1, figsize=(8,8), dpi=1800)
    i = 0
    for train_index, test_index in kf.split(X):
        i += 1
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = classifer.fit(X_train, y_train)
        model.fit(X_train, y_train)
        viz = RocCurveDisplay.from_estimator(
            model,
            X_test,
            y_test,
#            name=f"ROC fold {i}",
            alpha=0.5,
            lw=1,
            color="grey",
            ax=ax,
        )
        yhat, y_prob = predict_test(model, X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob[:,1])
        roc_auc = auc(fpr, tpr)
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
        precision, recall, f1_score, accuracy = report(y_test, yhat)
        reportNew = {"Classifer":name, 'Features':"-".join(feature_list), "Fold":i,'Precision':precision, 'Recall':recall, 'F1_score':f1_score, 'Accuracy':accuracy, 'AUC':roc_auc}
        df_reportNew = pd.DataFrame(reportNew, index=[0])
        df_report = pd.concat([df_report,df_reportNew],ignore_index=True)
        df_report.reset_index()
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
#    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Random', alpha=.8)

    mean_tpr = np.median(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    ax.plot(mean_fpr, mean_tpr, color='r',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize = 35)
    ax.set_ylabel("True Positive Rate", fontsize = 35)
    ax.tick_params(axis='both', which='both', labelsize=28)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
    #ax.set_title("ROC Curve with Markers {} on Classifier {} in Cell {}".format(feature_list, name, cell),loc='center', wrap=True)
    #ax.legend(loc="lower right")
    legend = ax.get_legend()
    if legend:
        legend.remove()
    feature_name = "-".join(feature_list)

    figure_name = "{}-{}.island.primary.ROC.tf.starr.png".format(name.replace(" ", "_"),feature_name)
    report_name = "{}-{}.island.primary.report.tf.starr.csv".format(name.replace(" ", "_"), feature_name)

    plt.savefig(figure_name,dpi=1800,bbox_inches='tight')
    df_report.to_csv(report_name)
    return model

names = ["LogisticRegression","LinearSVM",
         "DecisionTree", "RandomForest", "NeuralNet", "AdaBoost",
         "NaiveBayes"]

classifiers = [
    LogisticRegression(fit_intercept=True),
    #LinearSVC(random_state=np.random.RandomState(0)),
    LinearSVC(random_state=np.random.RandomState(0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    ]

###################sys.argv[]: Training Set, Predicting File Folder, Classifier Index, file save folder
def main():
    classifer_index = int(sys.argv[3])
    Classifier=classifiers[classifer_index]
    Classifier_Name=names[classifer_index]    
    df=pd.read_csv(sys.argv[1],index_col=0)
    df_features_minmax=df[sys.argv[6:]]
    df_label=df['TFs'].clip(0,1)
    kf = KFold(n_splits=10, shuffle=True, random_state=np.random.RandomState(123456))
    features=sys.argv[6:]
    Model=train_and_evaluate(kf, df_features_minmax, df_label, features, Classifier, Classifier_Name, sys.argv[5])
    feature_name = "-".join(features)
    joblib.dump(Model,'{}-{}.joblib'.format(Classifier_Name,feature_name))
    if Classifier_Name == "LogisticRegression":
        print(sys.argv[6:],Model.coef_,Model.intercept_)
    #####################Predict Using the Trained Model
    '''
    Model=joblib.load('{}/{}.joblib'.format(sys.argv[4],Classifier_Name))
    files = Path(sys.argv[2]).glob('*.minmax')
    for file in files:
        if os.path.getsize(file) != 0:
            df_predict=pd.read_csv(file,index_col=0)
            yhat, y_prob=predict_test(Model, df_predict)
            df_predict["Predict_Label"]=yhat
            df_predict["Predict_Prob"]=y_prob[:,1]
            df_predict["Cell"]=str(file).split('/')[-1].replace('.minmax','')
            df_predict.to_csv(str(file)+Classifier_Name+".predict")
    '''
if __name__=="__main__":
    main()
