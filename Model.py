import streamlit as st
import pickle
import numpy as np
import os

st.set_page_config(page_title="Titanic Prediction", layout="wide")

st.title('🚢 Titanic Tragedy Survival Prediction')
st.write("---")

try:
    model_path = 'titanic_tragedy (1).pkl'
    if os.path.exists(model_path):
        model = pickle.load(open(model_path, 'rb'))
    else:
        st.error("❌ Model file not found! Please ensure 'titanic_tragedy (1).pkl' exists.")
        st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

st.subheader("📋 Enter Passenger Details (10 Features)")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Basic Info**")
    pclass = st.selectbox("Passenger Class", [1, 2, 3], help="1: First, 2: Second, 3: Third")
    sex = st.selectbox("Sex", ["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    
with col2:
    st.write("**Family**")
    sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
    
    st.write("**Fare & Port**")
    fare = st.number_input("Fare ($)", min_value=0.0, value=50.0)
    embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"], help="C: Cherbourg, Q: Queenstown, S: Southampton")

with col3:
    st.write("**Additional**")
    has_cabin = st.selectbox("Has Cabin Number", ["Yes", "No"])
    title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Other"])
    is_alone = st.selectbox("Traveling Alone", ["No", "Yes"])

st.write("---")

if st.button("🔮 Predict Survival", use_container_width=True):
    try:
        sex_encoded = 1 if sex == "Male" else 0
        embarked_encoded = {"C": 0, "Q": 1, "S": 2}[embarked]
        has_cabin_encoded = 1 if has_cabin == "Yes" else 0
        is_alone_encoded = 1 if is_alone == "Yes" else 0
        
        
        title_map = {"Mr": 0, "Mrs": 1, "Miss": 2, "Master": 3, "Other": 4}
        title_encoded = title_map[title]
        
        
        input_data = np.array([[
            pclass,       
            sex_encoded,   
            age,          
            sibsp,           
            parch,            
            fare,             
            embarked_encoded,
            has_cabin_encoded,
            title_encoded,    
            is_alone_encoded  
        ]])
        
        prediction = model.predict(input_data)
        
        
        st.write("---")
        if prediction[0] == 1:
            st.success("✅ Passenger Likely SURVIVED!", icon="✅")
            st.balloons()
        else:
            st.warning("⚠️ Passenger Likely DID NOT SURVIVE", icon="⚠️")
            
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")