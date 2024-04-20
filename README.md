# Simultaneous Prediction of Functional States and Types of cis-regulatory Modules Reveals Their Prevalent Dual Uses as Enhancers and Silencers
## The training datasets for predicting funcitonal states of enhancers and silencers
   - All-Enhancer-Features-Part1.zip (We split the training dataset into 3 parts for upload.)
   - All-Enhancer-Features-Part2.zip
   - All-Enhancer-Features-Part3.zip
   - All-Silencer-Features.zip

## The code and models for training, evaluation, and testing:
   - Enhancers:
     - Train_evaluate_model_enhancer.py
     - Model trained with logistic regression: LogisticRegression-CA-H3K27ac-H3K4me1.joblib
   - Silencers
     - Train_evaluate_model_silencer.py
     - Model trained with logistic regression: LogisticRegression-CA-H3K9me3-H3K27me3.joblib

## The code for predicting active enhancers and silencers in the whole genome:
   - Predict_Enhancer.py
   - Predict_Silencer.py

## Predicted active enhancers and silencers examples in K562 cells
   - human-Enhancers-K562.CRM.minmaxLogisticRegression.predict.zip
   - human-Silencers-K562.CRM.minmaxLogisticRegression.predict.zip
