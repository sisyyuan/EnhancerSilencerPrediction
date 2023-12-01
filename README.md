Simultaneous Prediction of Functional States and Types of cis-regulatory Modules

The code for training and predicting active CRMs in the whole genome in K562

The code for training, evaluation, testing.

Train_evaluate_model_island_features_importance_signal.py
Example: Train_evaluate_model_island_features_importance_signal.py K562:Erythroblast:Bone+Marrow classifer_indicator (for example 0 for "LogesticRegression")
Classifer indicators: 0: "LogesticRegression",1:"Linear SVM",2:"Decision Tree", 3:"Random Forest", 4:"Neural Net", 5:"AdaBoost", 6:"Naive Bayes"
The code for predicting active CRMs in the whole genome.

Train_evaluate_model_island_features_importance_signal_predict_whole.py
Example: python Train_evaluate_model_island_features_importance_signal_predict_whole.py K562:Erythroblast:Bone+Marrow classifer_indicator (for example 0 for "LogesticRegression")
Classifer indicators: 0: "LogesticRegression",1:"Linear SVM",2:"Decision Tree", 3:"Random Forest", 4:"Neural Net", 5:"AdaBoost", 6:"Naive Bayes"
The model trained in K562

LogesticRegression+CA-H3K27ac-H3K4me1-H3K4me3.0.5.signal.whole_model.jobib
The training dataset in K562

K562:Erythroblast:Bone+Marrow.data.tf.info.signal.0.5.clean.csv
The features of a subset CRMs in K562

K562:Erythroblast:Bone+Marrow.data.tf.info.signal.0.5.features.csv
predicted active CRMs in human and mouse.

human_active_crms.tar.gz
mouse_active_crms.tar.gz
