import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

#Loading the saved models

diabetes_model = pickle.load(open('D:/final year project/Diabetes Disease Prediction/diabetes_model.sav','rb'))
heart_disease_model = pickle.load(open('D:/final year project/Heart Disease Prediction/heart_disease_model.sav','rb'))
parkinsons_model = pickle.load(open('D:/final year project/Parkinsons Disease Prediction/parkinsons_model.sav','rb'))


# best_model = load_model.best_estimator_

# # creating a function for prediction

# def predict_attrition(input_data):
    
#     inputdata_as_nparray=np.asarray(input_data)
#     inputdata_reshaped=inputdata_as_nparray.reshape(1,-1)
#     prediction_probability=best_model.predict_proba(inputdata_reshaped)
#     prediction=best_model.predict(inputdata_reshaped)
#     print(f"No attrition predicted Probability(class 0) = {prediction_probability[0, 0]}")
#     print(f"Attrition predicted Probability(class 1) = {prediction_probability[0, 1]}")
#     print(prediction)
    
#     if prediction[0]==0:
#       return 'Attrition = "No"'
#     else:
#       return 'Attrition = "Yes"'
  

#Sidebar for navigation

with st.sidebar:
    
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],
                           icons = ['activity','heart','person'],
                           default_index = 0)
    
    
   
#Diabetes Prediction Page

if(selected == 'Diabetes Prediction'):
    
    #Page title
    st.title('Diabetes Prediction using ML')
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
    
    
    #Code for prediction
    diab_diagnosis = ''
    
    #Creating a button for prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0]==1):
            diab_diagnosis = 'The person is Diabetic'
            
        else:
            diab_diagnosis = 'The person is Not Diabetic'
            
            
    st.success(diab_diagnosis)
    
    
    
            
#Heart Disease Prediction Page
if(selected == 'Heart Disease Prediction'):
    
    #Page title
    st.title('Heart Disease Prediction using ML')
    
    age = st.number_input('Age of the Person')
    sex = st.number_input('Sex of the Person')
    cp = st.number_input('Chest pain types')
    trestbps = st.number_input('Resting Blood Pressure')
    chol = st.number_input('Serum Cholestoral in mg/dl')
    fbs = st.number_input('Fasting blood sugar > 120 mg/dl')
    restecg = st.number_input('Resting Electrocardiographic results')
    thalach = st.number_input('Maximum Heart Rate achieved')
    exang = st.number_input('Exercise Induced Angina')
    oldpeak = st.number_input('ST depression induced by exercise')
    slope = st.number_input('Slope of the peak exercise ST segment')
    ca = st.number_input('Mjor vessels colored by flourosopy')
    thal = st.number_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
    
    
    #Code for prediction
    heart_diagnosis = ''
    
    #Creating a button for prediction
    
    if st.button('Heart Test Result'):
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        
        if (heart_prediction[0]==1):
            heart_diagnosis = 'The person is suffering from Heart disease'
            
        else:
            heart_diagnosis = 'The person is Not suffering from Heart disease'
            
            
    st.success(heart_diagnosis)
    
    
    

    
#Parkinsons Prediction Page
if(selected == 'Parkinsons Prediction'):
    
    #Page title
    st.title('Parkinsons Prediction using ML')
    

    fo = st.text_input('MDVP:Fo(Hz)')
    fhi = st.text_input('MDVP:Fhi(Hz)')
    flo = st.text_input('MDVP:Flo(Hz)')
    Jitter_percent = st.text_input('MDVP:Jitter(%)')
    Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
    RAP = st.text_input('MDVP:RAP')
    PPQ = st.text_input('MDVP:PPQ')
    DDP = st.text_input('Jitter:DDP')
    Shimmer = st.text_input('MDVP:Shimmer')
    Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
    APQ3 = st.text_input('Shimmer:APQ3')
    APQ5 = st.text_input('Shimmer:APQ5')
    APQ = st.text_input('MDVP:APQ')
    DDA = st.text_input('Shimmer:DDA')
    NHR = st.text_input('NHR')
    HNR = st.text_input('HNR')
    RPDE = st.text_input('RPDE')
    DFA = st.text_input('DFA')
    spread1 = st.text_input('spread1')
    spread2 = st.text_input('spread2')
    D2 = st.text_input('D2')
    PPE = st.text_input('PPE')
        
        
    #Code for prediction
    parkinsons_diagnosis = ''
        
    #Creating a button for prediction
        
    if st.button('Parkinsons Test Result'):
            parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])
            
            if (parkinsons_prediction[0]==1):
                parkinsons_diagnosis = 'The person is suffering from Parkinsons disease'
                
            else:
                parkinsons_diagnosis = 'The person is Not suffering from Parkinsons disease'
                
                
    st.success(parkinsons_diagnosis)

# creating web app 
def main():
    
    # giving a title
    st.title('EMPLOYEE ATTRITION PREDICTION')
    
    # getting the input from the user  
    age = st.slider('Age:', min_value=18, max_value=65)
    business_travel = st.selectbox("Business Travel (Non-Travel:0, Travel_Rarely:1, Travel_Frequently:2):", [0, 1, 2])
    daily_rate = st.slider('Daily Rate (paid per day):', min_value=500, max_value=1500)
    distance_from_home = st.slider('Distance From Home:', min_value=1, max_value=30)
    education = st.selectbox("Education (Below college:1, College:2, Bachelors:3, Masters:4, Doctor:5):", [1, 2, 3, 4, 5])
    environment_satisfaction = st.slider('Environment Satisfaction:', min_value=1, max_value=4)
    gender = st.selectbox("Gender (Male:0, Female:1):", [0, 1])
    hourly_rate = st.slider('Hourly Rate (paid for each hour of work):', min_value=50, max_value=100)
    job_involvement = st.selectbox("Job Involvement (Low:1, Medium:2, High:3, Very High:4):", [1, 2, 3, 4])
    job_level = st.selectbox('Job Level:', [1, 2, 3, 4, 5])
    job_satisfaction = st.selectbox("Job Satisfaction (Low:1, Medium:2, High:3, Very High:4):", [1, 2, 3, 4])
    marital_status = st.selectbox("Marital Status (Single:0, Married:1, Divorced:2):", [0, 1, 2])
    monthly_income = st.slider('Monthly Income (total earnings within a single month):', min_value=1000, max_value=20000)
    monthly_rate = st.slider('Monthly Rate (predetermined amount that an employee is paid on a monthly basis irrespective of the number of hours worked):', min_value=1000, max_value=25000)
    num_companies_worked = st.slider('Num Companies Worked (number of companies worked):', min_value=0, max_value=10)
    over_18 = st.selectbox("Over 18 (Yes:1, No:0):", [0, 1])
    over_time = st.selectbox("Over Time (Yes:1, No:0):", [0, 1])
    percent_salary_hike = st.slider('Percent Salary Hike (percentage (0-25%) increase salary from one period to another):', min_value=0, max_value=25)
    performance_rating = st.selectbox("Performance Rating (Low:1, Good:2, Excellent:3, Outstanding:4):", [1, 2, 3, 4])
    relationship_satisfaction = st.selectbox("Relationship Satisfaction (Low:1, Medium:2, High:3, Very High:4):", [1, 2, 3, 4])
    standard_hours = st.slider('Standard Hours (number of working hours per week):', min_value=30, max_value=120)
    stock_option_level = st.selectbox('Stock Option Level:', [0, 1, 2, 3])
    total_working_years = st.slider('Total Working Years:', min_value=0, max_value=40)
    training_times_last_year = st.slider('Training Times Last Year:', min_value=0, max_value=10)
    work_life_balance = st.selectbox("Work Life Balance (Bad:1, Good:2, Better:3, Best:4):", [1, 2, 3, 4])
    years_at_company = st.slider('Years At Company:', min_value=0, max_value=40)
    years_in_current_role = st.slider('Years In Current Role:', min_value=0, max_value=20)
    years_since_last_promotion = st.slider('Years Since Last Promotion:', min_value=0, max_value=15, value=2)
    years_with_curr_manager = st.slider('Years With Curr Manager:', min_value=0, max_value=20, value=4)

        
    # Button to trigger the prediction
    if st.button('Predict Attrition'):
        input_data = [age, business_travel, daily_rate, distance_from_home, education, environment_satisfaction,
                      gender, hourly_rate, job_involvement, job_level, job_satisfaction, marital_status,
                      monthly_income, monthly_rate, num_companies_worked, over_18, over_time, percent_salary_hike,
                      performance_rating, relationship_satisfaction, standard_hours, stock_option_level,
                      total_working_years, training_times_last_year, work_life_balance, years_at_company,
                      years_in_current_role, years_since_last_promotion, years_with_curr_manager]
        
        prediction_result = predict_attrition(input_data)
        st.text(prediction_result)
        
        prediction_probability = best_model.predict_proba(np.array(input_data).reshape(1, -1))
        st.text(f'Probability of No Attrition: {prediction_probability[0, 0]:.4f}')
        st.text(f'Probability of Attrition: {prediction_probability[0, 1]:.4f}')

        
if __name__ == '__main__':
    main()
        