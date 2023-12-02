# Simultaneous Prediction of Functional States and Types of cis-regulatory Modules
## The code and models for training, evaluation, and testing:
   - Enhancers:
	   - Train_evaluate_model_enhancer.py
     - Model trained with logistic regression: LogisticRegression-CA-H3K27ac-H3K4me1.joblib
   - Silencers
     - Train_evaluate_model_silencer.py
     - Model trained with logistic regression: LogisticRegression-CA-H3K9me3-H3K27me3.joblib
## The code for predicting active enhancers and silencers in the whole genome:
   - Predict_enhancer.py
   - Predict_silencer.py
## The training datasets for enhancers and silencers
   - All-Enhancer-Features-0.3.zip
   - All-Silencer-Features.zip
## Predicted active enhancers and silencers in K562 cells
   - human-enhancers-K562.CRM.minmaxLogisticRegression.predict.zip
   - human-silencers-K562.CRM.minmaxLogisticRegression.predict.zip
