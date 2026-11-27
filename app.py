import streamlit as st
import pandas as pd
import joblib
model = joblib.load("titanic_model.pkl")
st.title("Titanic Survival")
st.write(
    "Enter passenger information to predict "
    "the probability of survival given a passenger"
)
Pclass=st.selectbox(
    "Passenger Class",
    [1,2,3],
)
Sex=st.selectbox(
    "Sex",
    ["male","female"]
)
Age=st.number_input(
    "Age",
    min_value=0.00,
    max_value=100.0,
    value=30.0
)
SibSp=st.number_input(
    "Number of Siblings",
    min_value=0,
    max_value=10,
    value=0
)
Parch=st.number_input(
    "Number of Parents",
    min_value=0,
    max_value=10,
    value=0
)
Fare=st.number_input(
    "Fare",
    min_value=0.0,
    max_value=32.2,
)
Ticket=st.number_input(
    "Ticket",
    min_value=0.0,
    max_value=50.0,
)
Embarked=st.selectbox(
    "Embarked",
    ["S","C","Q"],
)
input_data = pd.DataFrame([{
    "Pclass": Pclass,
    "Sex_male": 1 if Sex =="male" else 0,
    "Sex_female": 1 if Sex =="female" else 0,
    "Age":Age,
    "SibSp":SibSp,
    "Parch":Parch,
    "Fare":Fare,
    "Ticket":Ticket,
    "Embarked_S": 1 if Embarked == "S" else 0,
    "Embarked_C": 1 if Embarked == "C" else 0,
    "Embarked_Q": 1 if Embarked == "Q" else 0
}])
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    if prediction == 1:
        st.success("Prediction: Passenger Survived")
    else:
        st.error("Prediction: Passenger Did Not Survive")
    st.write(
        f"Estimated Survival Probability: "
        f"{probability*100:.2f}%"
    )
