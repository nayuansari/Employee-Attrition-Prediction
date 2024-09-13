import streamlit as st
import plotly.express as px
import pandas as pd
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Attrition Analysis & Prediction", page_icon=":bar_chart:", layout="wide")
st.title(':bar_chart: ATTRITION ANALYTICS: FUTURE-READY WORKFORCE')
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

st.write('This web application is designed to analyse & predict employee attrition using a machine learning model trained on various employee-related features. Users can input specific attributes of an employee, such as age, job satisfaction, education level, and more, to receive a prediction on the likelihood of that employee leaving the organization. The application provides insights into the potential attrition risk and aims to assist HR professionals and managers in proactive decision-making and retention strategies.')

fl= st.file_uploader(":file_folder: You can UPLOAD your own dataset to analyse your data:", type=(['csv']))

if fl is not None:
    filename=fl.name
    st.write(filename)
    df=pd.read_csv(filename, encoding="ISO-8859-1")
else:
    df=pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    
col1, col2 = st.columns((2))

st.sidebar.markdown('<div style="color: lightgreen;">* After selecting the options hide the sidebar for better visualization.<br><br> (to hide: click ''(✕)'' at top right corner of the sidebar menu.)</div>', unsafe_allow_html=True)
st.sidebar.header("Choose according to your choices: ")

ef=st.sidebar.multiselect('Select Education Field:', df['EducationField'].unique())
filtered_data = df[df['EducationField'].isin(ef)]

with col1:
    st.subheader("Attrition across Education Field")
    if filtered_data.empty:
        st.warning("Select education field(s) from left sidebar menu to see data.")
    else:
        fig = px.bar(filtered_data,
                    x="EducationField",
                   color="Attrition",title="Count of Education Field by Attrition",
                  labels={"EducationField": "Education Field", "value": "Count"},
                 category_orders={"EducationField": ef},barmode="group", height=400, width=600)
        st.plotly_chart(fig)
    
    
el=st.sidebar.multiselect('Select Education Level:', df['Education'].unique())
filtered_df = df[df['Education'].isin(el)]

with col2:
    st.subheader("Attrition across Education level")
    if filtered_df.empty:
        st.warning("Select education level field(s) from left sidebar menu to see data.")
    else:
        education_mapping = {1: 'Below college',2: 'College',3: 'Bachelors',4: 'Masters',5: 'Doctor'}
        filtered_df['EducationLabel'] = filtered_df['Education'].map(education_mapping)
        fig = px.bar(filtered_df,x='EducationLabel',color='Attrition',
                     title='Count of Education Level by Attrition',
                     labels={'EducationLabel': 'Education Level', 'value': 'Count'},
                     category_orders={'EducationLabel': list(education_mapping.values())},
                     barmode='group',height=400,width=600)
        st.plotly_chart(fig)




with col1:
    chart_type = st.sidebar.selectbox("Select Chart Type for Department: ", ["Countplot", "Pie Chart"])

    if chart_type == "Countplot":
        st.subheader("Attrition across Department")
    
        fig = px.histogram(df, x='Department', color='Attrition',
                           title='Count of Department by Attrition',
                           labels={"Department": "Department", "count": "Count"},
                           barmode='group', height=400, width=600)
        st.plotly_chart(fig)

    elif chart_type == "Pie Chart":
        st.subheader('Department Distribution')  
    
        attrition_counts = df['Department'].value_counts().reset_index()
        attrition_counts.columns = ['Department', 'Count']
    
        fig = px.pie(attrition_counts, 
                     names='Department', values='Count', 
                     title='Distribution of Departments',
                     labels={'Department': 'Department'}, hole=0.3, height=400, width=600)
        st.plotly_chart(fig)

with col2:
    selection = st.sidebar.selectbox("Show data for Job Satisfaction distribution:", ["Yes", "No"])
    if selection == "No":
        st.warning("No data available for job satisfaction distribution.")
    else:
        st.subheader('Job Satisfaction Distribution \n(Low:1, Medium:2, High:3, Very High:4)')
        satisfaction_counts = df['JobSatisfaction'].value_counts().reset_index()
        satisfaction_counts.columns = ['JobSatisfaction', 'Count']
        fig = px.pie(satisfaction_counts, 
                     values='Count', 
                     names='JobSatisfaction',
                     labels={'Count': 'Percentage', 'JobSatisfaction': 'Job Satisfaction Level'},
                     hole=0.5, height=400,width=600)
        st.plotly_chart(fig)


with col1:
    selection1 = st.sidebar.selectbox("Show data for Job Role distribution:", ["Yes", "No"])
    if selection1 == "No":
        st.warning("No data available for job role distribution.")
    else:
        st.subheader("Job Role Distribution")
        jobrole_counts = df['JobRole'].value_counts()
        fig = px.pie(names=jobrole_counts.index, values=jobrole_counts.values,
                     title='Distribution of Job roles',
                     labels={"label": "Job Role", "value": "Count"},
                     hole=0.5, height=400,width=600)
        st.plotly_chart(fig)


with col2:
    st.subheader("Attrition across Job Roles")
    fig = px.bar(df, x='JobRole', color='Attrition', title='Count of Job roles by Attrition',
         labels={'JobRole': 'Job Role', 'value': 'Count'}, barmode='group', height=400,width=600)
    st.plotly_chart(fig)


with col1:
    chart_type = st.sidebar.selectbox("Select Chart Type for Work life balance: ", ["Countplot", "Pie Chart"])
    if chart_type == "Countplot":
        st.subheader("Attrition across Work Life Balance")
        fig = px.histogram(df, x='WorkLifeBalance', color='Attrition',
                           title='(1: "Bad", 2: "Good", 3: "Better", 4: "Best")',
                           labels={"WorkLifeBalance": "Work Life Balance", "count": "Count"},
                           barmode='group', height=400,width=600)
        st.plotly_chart(fig)
    elif chart_type == "Pie Chart":
        st.subheader('Work Life Balance Distribution')
        label_mapping = {1: "Bad", 2: "Good", 3: "Better", 4: "Best"}
        df['WorkLifeBalanceLabel'] = df['WorkLifeBalance'].map(label_mapping)
        work_life_balance_counts = df['WorkLifeBalanceLabel'].value_counts().reset_index()
        work_life_balance_counts.columns = ['Work Life Balance', 'Count']
        fig = px.pie(work_life_balance_counts, 
                     names='Work Life Balance', 
                     values='Count', 
                     title='Distribution of Work-Life Balance',
                     labels={'Work Life Balance': 'Balance Level'},hole=0.3, height=400,width=600)
        st.plotly_chart(fig)


selected_genders = st.sidebar.multiselect("Select Gender:",df["Gender"].unique())
filtered_df1 = df[df["Gender"].isin(selected_genders)]

with col2:
    st.subheader("Job Satisfaction across Gender")
    if filtered_df1.empty:
        st.warning("Select gender from left sidebar menu to see data.")
    else:
        fig = px.histogram(filtered_df1, 
                   x="Gender", 
                   color="JobSatisfaction",
                   title="Count of Gender by Job Satisfaction",
                   labels={'Gender': 'Gender', 'JobSatisfaction': 'Job Satisfaction Level'},
                   category_orders={"Gender": selected_genders},
                   barmode='group', height=400,width=600)
        st.plotly_chart(fig)




# Further analysis
st.markdown("<br>", unsafe_allow_html=True)
sns.set(style="whitegrid", palette="muted")

# Selecting numerical features for plotting
numerical_features = ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate',
                       'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany',
                       'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

# Streamlit app
st.subheader("Distribution of Numerical Features by Attrition")

# Sidebar for feature selection
selected_features = st.sidebar.multiselect("Select Features:", numerical_features)

# Check if features are selected
if selected_features:
    for feature in selected_features:
        # Converting to Plotly
        fig = px.histogram(df, x=feature, color='Attrition', 
                           title=f'Distribution of {feature} by Attrition',
                           marginal="rug",template="plotly_white", nbins=30, height=400, width=600) 
        st.plotly_chart(fig)
else:
    st.warning("Please select one or more features from the left sidebar menu.")


# prediction model
import numpy as np
import pickle

load_model = pickle.load(open('Attrition_Analytics.sav', 'rb'))

best_model = load_model.best_estimator_

# creating a function for prediction

def predict_attrition(input_data):
    
    inputdata_as_nparray=np.asarray(input_data)
    inputdata_reshaped=inputdata_as_nparray.reshape(1,-1)
    prediction_probability=best_model.predict_proba(inputdata_reshaped)
    prediction=best_model.predict(inputdata_reshaped)
    print(f"No attrition predicted Probability(class 0) = {prediction_probability[0, 0]}")
    print(f"Attrition predicted Probability(class 1) = {prediction_probability[0, 1]}")
    print(prediction)
    
    if prediction[0]==0:
      return 'Probability of an employee leaving a company is LOWER.'
    else:
      return 'Probability of an employee leaving a company is HIGHER!'
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
    environment_satisfaction = st.selectbox('Environment Satisfaction (Low:1, Medium:2, High:3, Very High:4):', [1, 2, 3, 4])
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
    if st.button('**Predict Attrition**', key='predict_button', help="Click this button to predict attrition."):
        input_data = [age, business_travel, daily_rate, distance_from_home, education, environment_satisfaction,
                      gender, hourly_rate, job_involvement, job_level, job_satisfaction, marital_status,
                      monthly_income, monthly_rate, num_companies_worked, over_18, over_time, percent_salary_hike,
                      performance_rating, relationship_satisfaction, standard_hours, stock_option_level,
                      total_working_years, training_times_last_year, work_life_balance, years_at_company,
                      years_in_current_role, years_since_last_promotion, years_with_curr_manager]
        
        prediction_result = predict_attrition(input_data)   
        st.markdown(f'<div style="font-size:55px; color:skyblue; font-family:cursive;">{prediction_result}</div>', unsafe_allow_html=True)
        
        prediction_probability = best_model.predict_proba(np.array(input_data).reshape(1, -1))
        st.text(f'Probability of No Attrition: {prediction_probability[0, 0]:.4f}')
        st.text(f'Probability of Attrition: {prediction_probability[0, 1]:.4f}')

        
if __name__ == '__main__':
    main()

# Above the screen
#st.markdown("""<style>.thank-you {position: fixed;bottom: 10px;left: 50%;transform: translateX(-50%);font-size: 24px;color: #333;}
#    </style>""", unsafe_allow_html=True)
    
#st.markdown('<div class="thank-you">Employee Attrition</div>', unsafe_allow_html=True)


# Ending 
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<div style="font-size: 24px; color: #9370DB; text-align: center;">Hope to see you soon :)</div>', unsafe_allow_html=True)
