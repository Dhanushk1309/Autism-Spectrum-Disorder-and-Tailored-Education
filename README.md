# 🧠 Autism Detection Using Random Forest Classifier

This project builds a Machine Learning model to detect whether a person is likely to have Autism Spectrum Disorder (ASD) based on behavioral and clinical indicators such as Age, Sex, ADHD, Anxiety, Epilepsy, Jaundice, Family History, and multiple screening test scores (CARS, SRS, WISC, etc.).

The model uses a Random Forest Classifier and is saved as a .pkl file for deployment (Flask/Streamlit).

📌 Project Overview

The goal of this project is to predict ASD Result (0 = No Autism, 1 = Autism) using patient data.
The workflow includes:

Loading and cleaning the dataset

Feature selection (Age, Sex, ADHD, Anxiety, etc.)

Handling missing values

Splitting the dataset into training/testing sets

Training a Random Forest Classifier

Hyperparameter tuning (550–600 estimators)

Saving the best model using pickle

📂 Dataset Description

The dataset includes the following key features:

Feature	Description
Age	Age of the person
Sex	Gender (0/1)
ADHD	ADHD symptoms
anxiety	Anxiety symptoms
epilepsy	Epilepsy symptoms
CARS	Childhood Autism Rating Scale score
SRS	Social Responsiveness Scale score
WISC	Intelligence score
ABA	Applied Behavior Analysis
Asq	Autism Screening Questionnaire
Jaundice	Yes/No
Family_ASD	Family history of autism
Result	Target variable (0 = No Autism, 1 = Autism)
⚙️ Technologies Used

Python

Pandas

NumPy

Scikit-learn

Random Forest Classifier

Pickle

Jupyter Notebook

🧹 Data Preprocessing

Selected relevant diagnostic and behavioral features

Mapped Result values to binary format

Removed missing values

Converted dataframe to NumPy arrays

Split into train (80%) and test (20%) sets

🧠 Model Training
✔ Random Forest Classifier

The model tested estimators in the range 550 to 600 to find the best accuracy.

for i in range(550, 600):
    rfc = RandomForestClassifier(n_estimators=i)
    rfc.fit(X_train, y_train)
    pred_i = rfc.predict(X_test)
    error.append(np.mean(pred_i != y_test))


The best estimator was found to be:

optimal_estimators = 571
rfc = RandomForestClassifier(n_estimators=571)
rfc.fit(X_train, y_train)

💾 Saving the Model

The trained model is saved as:

pickle.dump(rfc, open('autism.pkl', 'wb'))


This is used for Flask web app deployment.

📈 Model Performance

Hyperparameter tuning improves accuracy & lowers error

Random Forest gives stable predictions for binary classification

Pickle model loads instantly for API/web usage

🚀 How to Run the Project

Install dependencies:

pip install pandas numpy scikit-learn


Run the Python script:

python autism_model.py


The model will be saved as:

autism.pkl


You can load it in Flask or Streamlit:

model = pickle.load(open('autism.pkl', 'rb'))

📌 Future Enhancements

Deploy using Flask/Streamlit

Add confusion matrix and evaluation metrics

Include feature importance visualization

Add UI for user input

👨‍💻 Author

Dhanush
Python & SQL Developer
Machine Learning Project Enthusiast
