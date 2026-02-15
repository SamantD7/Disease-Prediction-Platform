from django.shortcuts import render
import numpy as np
from joblib import load

model = load('./model/diabetes_model.pkl')
model_lr = load('./model/predict2.pkl')
model_rf = load('./model/heart_random_forest.pkl')
model_rf_diabetes = load('./model/predict1.pkl')

model_rf_parkinson = load('./model/parkinson_rf.pkl')
model_lr_parkinson = load('./model/parkinson_lr.pkl')
parkinson_scaler = load('./model/parkinson_scalar.pkl')

def homepage(request):
    return render(request, 'index.html')

def classify_risk(probability):
    if probability >= 0.8:
        return "High Risk"
    elif probability >= 0.5:
        return "Moderate Risk"
    else:
        return "Low Risk"

def heart_disease(request):
    if request.method == 'POST':
        try:
            age = int(request.POST['age'])
            sex = int(request.POST['sex'])
            cp = int(request.POST['cp'])
            trestbps = int(request.POST['trestbps'])
            chol = int(request.POST['chol'])
            fbs = int(request.POST['fbs'])
            restecg = int(request.POST['restecg'])
            thalach = int(request.POST['thalach'])
            exang = int(request.POST['exang'])
            oldpeak = float(request.POST['oldpeak'])
            slope = int(request.POST['slope'])
            ca = int(request.POST['ca'])
            thal = int(request.POST['thal'])

            input_data = [age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]
            
            input_array = np.asarray(input_data).reshape(1, -1)
            pred_rf = model_rf.predict(input_array)[0]
            prob_rf = model_rf.predict_proba(input_array)[0][1]
            risk_rf = round(prob_rf * 100, 2)

            pred_lr = model_lr.predict(input_array)[0]
            prob_lr = model_lr.predict_proba(input_array)[0][1]
            risk_lr = round(prob_lr * 100, 2)

            context = {
                'rf_result': "⚠️ At Risk" if pred_rf == 1 else "✅ Not at Risk",
                'lr_result': "⚠️ At Risk" if pred_lr == 1 else "✅ Not at Risk",
                'rf_risk': f"{risk_rf}%",
                'lr_risk': f"{risk_lr}%",
                'rf_label': classify_risk(prob_rf),
                'lr_label': classify_risk(prob_lr),
            }
            return render(request, 'heart_result.html', context)

        except Exception as e:
            context = {'error': f"Error occurred: {e}"}
            return render(request, 'heart_result.html', context)

    return render(request, 'heart_assessment.html')

def diabetes_disease(request):
    if request.method == 'POST':
        try:
            age = int(request.POST['age'])
            bmi = float(request.POST['bmi'])
            blood_pressure_systolic = float(request.POST['blood_pressure_systolic'])
            glucose = float(request.POST['glucose'])
            hba1c = float(request.POST['hba1c'])
            blood_pressure_diastolic = float(request.POST['blood_pressure_diastolic'])
            cholesterol_total = float(request.POST['cholesterol_total'])
            cholesterol_hdl = float(request.POST['cholesterol_hdl'])
            cholesterol_ldl = float(request.POST['cholesterol_ldl'])
            ggt = float(request.POST['ggt'])
            serum_urate = float(request.POST['serum_urate'])
            dietary_calories = float(request.POST['dietary_calories'])
            family_history = int(request.POST['family_history'])
            gestational_diabetes = int(request.POST['gestational_diabetes'])

            input_data = np.array([[age, bmi, blood_pressure_systolic, glucose, hba1c,
                                    blood_pressure_diastolic, cholesterol_total, cholesterol_hdl,
                                    cholesterol_ldl, ggt, serum_urate, dietary_calories,
                                    family_history, gestational_diabetes]])

            prediction_lr = model.predict(input_data)[0]
            probability_lr = model.predict_proba(input_data)[0][1]
            risk_percent_lr = round(probability_lr * 100, 2)

            prediction_rf = model_rf_diabetes.predict(input_data)[0]
            probability_rf = model_rf_diabetes.predict_proba(input_data)[0][1]
            risk_percent_rf = round(probability_rf * 100, 2)

            context = {
                'rf_result1': "⚠️ At Risk" if prediction_rf == 1 else "✅ Not at Risk",
                'lr_result1': "⚠️ At Risk" if prediction_lr == 1 else "✅ Not at Risk",
                'rf_risk1': f"{risk_percent_rf}%",
                'lr_risk1': f"{risk_percent_lr}%",
                'risk_level_rf1': classify_risk(probability_rf),
                'risk_level_lr1': classify_risk(probability_lr),
            }

            return render(request, 'diabetes_result.html', context)

        except Exception as e:
            context = {'error': f"Error occurred: {e}"}
            return render(request, 'diabetes_result.html', context)

    return render(request, 'diabetes_assessment.html')

def Parkinson_disease(request):
    if request.method == 'POST':
        try:
            fo = float(request.POST['mdvp_fo'])
            fhi = float(request.POST['mdvp_fhi'])
            flo = float(request.POST['mdvp_flo'])
            jitter = float(request.POST['mdvp_jitter'])
            shimmer = float(request.POST['mdvp_shimmer'])
            nhr = float(request.POST['nhr'])
            hnr = float(request.POST['hnr'])
            rpde = float(request.POST['rpde'])
            dfa = float(request.POST['dfa'])
            spread1 = float(request.POST['spread1'])
            spread2 = float(request.POST['spread2'])
            d2 = float(request.POST['d2'])
            ppe = float(request.POST['ppe'])

            input_data = np.array([[fo, fhi, flo, jitter, shimmer, nhr, hnr,
                                    rpde, dfa, spread1, spread2, d2, ppe]])

            scaled_input = parkinson_scaler.transform(input_data)

            pred_rf = model_rf_parkinson.predict(scaled_input)[0]
            prob_rf = model_rf_parkinson.predict_proba(scaled_input)[0][1]
            risk_rf = round(prob_rf * 100, 2)

            pred_lr = model_lr_parkinson.predict(scaled_input)[0]
            prob_lr = model_lr_parkinson.predict_proba(scaled_input)[0][1]
            risk_lr = round(prob_lr * 100, 2)

            context = {
                'rf_result': "⚠️ At Risk" if pred_rf == 1 else "✅ Not at Risk",
                'lr_result': "⚠️ At Risk" if pred_lr == 1 else "✅ Not at Risk",
                'rf_risk': f"{risk_rf}%",
                'lr_risk': f"{risk_lr}%",
                'rf_label': classify_risk(prob_rf),
                'lr_label': classify_risk(prob_lr),
            }
            return render(request, 'parkinson_result.html', context)

        except Exception as e:
            return render(request, 'parkinson_result.html', {'error': f"Error occurred: {e}"})

    return render(request, 'parkinson_assessment.html')
