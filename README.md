# diabetes_prediction
The objective of this project is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset.


Data Description:-
Dataset is taken from kaggle competetion(PIMA Indians Diabetes) but originally generated in National Institute of Diabetes and Digestive and Kidney Diseases.
The datasets consists of several medical predictor variables and one target variable, Outcome.
Predictor variables/features:

    Pregnancies: Number of times pregnant
    Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
    BloodPressure: Diastolic blood pressure (mm Hg)
    SkinThickness: Triceps skin fold thickness (mm)
    Insulin: 2-Hour serum insulin (mu U/ml)
    BMI: Body mass index (weight in kg/(height in m)^2)
    DiabetesPedigreeFunction: Diabetes pedigree function
    Age: Age (years)
    Outcome: Class variable (0 or 1)

Here in the solution I compared different machine learning algorithms(ensemble method, gaussian classifier, logistic regression, naive bayes , nearest neighbours , Decision trees , etc.) and a deep learning model . 


After training these different models I used AUC-ROC value as a comparison parameter and choose one machine learning algorithm for further fine tuning of hyperparameters  of choosen algorithm.



Here I achieved around 82% accuracy score.
