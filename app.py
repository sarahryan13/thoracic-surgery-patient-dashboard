import base64
import io
import pickle

from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from markupsafe import Markup
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the Random Forest Classifier pickle
f = open('prediction_model.pkl', 'rb')
pred_model = pickle.load(f)
f.close()


@app.before_first_request
def startup():
    # Loads cleaned dataset when page is first loaded
    get_cleaned_data()


# Launch the login screen first
@app.route('/', methods=['GET', 'POST'])
def app_login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = "You must be an employee to access this dashboard!\n" \
                    "Please try again."
        else:
            return redirect(url_for('generate_dashboard'))
    return render_template('login.html', error=error)


@app.route('/main', methods=['POST', 'GET'])
def generate_dashboard():
    # Data Visualization: Patient Age Histogram
    age_plot = ''
    if len(patient_df) > 0:
        age_plot_url = get_encoded_hist(patient_df)
        age_plot = Markup('<img src="data:image/png;base64,{}" width: 360px, height: 288px>'.format(age_plot_url))

    # Data Visualization: Tumor Size Affect on Survival KDE Plot
    tnm_plot = ''
    if len(patient_df) > 0:
        tnm_plot_url = get_encoded_kde(patient_df)
        tnm_plot = Markup('<img src="data:image/png;base64,{}" width: 360px, height: 288px>'.format(tnm_plot_url))

    # Data Visualization: Correlation Heatmap
    heatmap = ''
    if len(patient_df) > 0:
        heatmap_url = get_encoded_heatmap(patient_df)
        heatmap = Markup('<img src="data:image/png;base64,{}" width: 360px, height: 288px>'.format(heatmap_url))

    # Displays the index page with the data visualizations when website is loaded
    if request.method == 'GET':
        return render_template('index.html',
                               age_plot=age_plot,
                               tnm_plot=tnm_plot,
                               heatmap=heatmap)

    # Collects user input data, makes prediction and returns result of prediction
    if request.method == 'POST':
        diagnosis = request.form['Diagnosis']
        fvc = request.form['FVC']
        pain = request.form['Pain']
        hae = request.form['Hae']
        dys = request.form['Dys']
        weak = request.form['Weak']
        tnm = request.form['TNM']
        t2diab = request.form['T2Diab']
        pad = request.form['PAD']
        smoker = request.form['Smoker']

        # Convert user input into DataFrame
        input_vals = pd.DataFrame([[diagnosis, fvc, pain, hae, dys, weak, tnm, t2diab, pad, smoker]],
                                  columns=['Diagnosis', 'FVC', 'Pain', 'Hae', 'Dys', 'Weak', 'TNM', 'T2Diab', 'PAD',
                                           'Smoker'])

        # Use machine learning model to predict patient outcome
        prediction = pred_model.predict_proba(input_vals)[:, 0]

        # Formatting result to display as % with 2 decimal places
        pred_percent = str(prediction).lstrip('[').rstrip(']')
        float_pred = float(pred_percent) * 100
        formatted_prediction = round(float_pred, 2)

        # Classify patient outcome [0-lives, 1-dies]
        classification = pred_model.predict(input_vals)
        if classification == 0:
            classification = "Prediction: Patient will be alive 1 year post surgery."
        elif classification == 1:
            classification = "Prediction: Patient will not be alive 1 year post surgery."

        # Generate F1 Score and Confusion Matrix for the prediction model
        ml_df = patient_df.drop(columns=['FEV', 'PerfStat', 'Cough', 'MI', 'Asthma', 'Age'])
        X = ml_df.drop("Target", axis=1)
        y = ml_df["Target"]
        X_scaled = StandardScaler().fit_transform(X)
        np.random.seed(42)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled,
                                                            y,
                                                            stratify=y,
                                                            test_size=0.3)

        pred_model.fit(X_train, y_train)
        y_pred = pred_model.predict(X_test)

        # Calculate F1 Score
        accuracy = f1_score(y_test, y_pred, average="micro")
        avg_accuracy = round(accuracy * 100, 2)
        print("F1 Score:", avg_accuracy, "%")

        return render_template('index.html',
                               scroll='prediction',
                               original_input={'Diagnosis': diagnosis,
                                               'FVC': fvc,
                                               'Pain': pain,
                                               'Hae': hae,
                                               'Dys': dys,
                                               'Weak': weak,
                                               'TNM': tnm,
                                               'T2Diab': t2diab,
                                               'PAD': pad,
                                               'Smoker': smoker},
                               age_plot=age_plot,
                               tnm_plot=tnm_plot,
                               heatmap=heatmap,
                               result=formatted_prediction,
                               classification=classification,
                               avg_accuracy=avg_accuracy)


@app.route('/')
def get_cleaned_data():
    # Loads the clean dataset and drops the first column
    global patient_df
    surgery_df = pd.read_csv('data/clean-surgery-data.csv')
    patient_df = surgery_df.drop(columns=['Unnamed: 0'])


def get_encoded_hist(df):
    # Take in loaded DataFrame and create a histogram
    fig, ax = plt.subplots()
    ax.hist(df["Age"], bins=25, histtype='stepfilled', color="skyblue", ec="w")
    plt.title('Age of Patients')
    plt.xlabel('Ages')
    plt.ylabel('Patients')
    plt.grid()

    # Encode the histogram
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    age_plot_url = base64.b64encode(img.getvalue()).decode()
    return age_plot_url


def get_encoded_kde(df):
    # Take in loaded DataFrame and create a KDE plot
    # Split df into survivors & non-survivors
    survivors = df[df["Target"] == 0]
    non_survivors = df[df["Target"] == 1]

    y = survivors["TNM"]
    z = non_survivors["TNM"]

    np.random.seed(42)
    sns.kdeplot(y, shade=True, color='#fdb147', label='Patient Lives')
    sns.kdeplot(z, shade=True, color='#ff6f52', label='Patient Dies')

    plt.title('Patient Outcome by Tumor Size')
    plt.xlabel('Tumor Size')
    plt.xlim([0.1, 4.9])
    plt.ylim([0.0, 1.2])

    # Encode the KDE plot
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    tnm_plot_url = base64.b64encode(img.getvalue()).decode()
    return tnm_plot_url


def get_encoded_heatmap(df):
    # Take in loaded DataFrame and create a histogram
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax = sns.heatmap(corr_matrix,
                     annot=True,
                     linewidth=0.1,
                     fmt=".2f",
                     cmap="Spectral_r")
    ax.figure.tight_layout()
    plt.title("Correlation Heatmap")

    # Encode the heatmap
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    heatmap_url = base64.b64encode(img.getvalue()).decode()
    return heatmap_url


def get_encoded_confusion_matrix(df):
    # Take in loaded DataFrame and create a histogram
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax = sns.heatmap(corr_matrix,
                     annot=True,
                     linewidth=0.1,
                     fmt=".2f",
                     cmap="Spectral_r")
    ax.figure.tight_layout()
    plt.title("Confusion Matrix")

    # Encode the heatmap
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    confusion_url = base64.b64encode(img.getvalue()).decode()
    return confusion_url


if __name__ == '__main__':
    app.run()
