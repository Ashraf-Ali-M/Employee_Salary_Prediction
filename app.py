
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load the trained model (now expecting a Pipeline) ---
try:
    model = joblib.load("best_model.pkl")
    if not hasattr(model, 'predict'):
        st.error("Error: Loaded model does not have a 'predict' method. Ensure 'best_model.pkl' contains a trained scikit-learn model or pipeline.")
        st.stop()
except FileNotFoundError:
    st.error("Error: 'best_model.pkl' not found. Please ensure the trained model file is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}. Please ensure 'best_model.pkl' is a valid joblib file.")
    st.stop()

# --- 2. Define LabelEncoder Mappings ---
# These mappings are crucial to convert categorical string inputs back to
# the numerical format that your model was trained on.
# They are derived assuming LabelEncoder sorts categories alphabetically.
# If your LabelEncoder produced different mappings, adjust these.

workclass_mapping = {
    'Federal-gov': 0, 'Local-gov': 1, 'Others': 2, 'Private': 3,
    'Self-emp-inc': 4, 'Self-emp-not-inc': 5, 'State-gov': 6
}
workclass_options = sorted(list(workclass_mapping.keys()))

marital_status_mapping = {
    'Divorced': 0, 'Married-AF-spouse': 1, 'Married-civ-spouse': 2,
    'Married-spouse-absent': 3, 'Never-married': 4, 'Separated': 5,
    'Widowed': 6
}
marital_status_options = sorted(list(marital_status_mapping.keys()))

occupation_mapping = {
    'Adm-clerical': 0, 'Armed-Forces': 1, 'Craft-repair': 2, 'Exec-managerial': 3,
    'Farming-fishing': 4, 'Handlers-cleaners': 5, 'Machine-op-inspct': 6,
    'Other-service': 7, 'Others': 8, 'Priv-house-serv': 9, 'Prof-specialty': 10,
    'Protective-serv': 11, 'Sales': 12, 'Tech-support': 13, 'Transport-moving': 14
}
occupation_options = sorted(list(occupation_mapping.keys()))

relationship_mapping = {
    'Husband': 0, 'Not-in-family': 1, 'Other-relative': 2, 'Own-child': 3,
    'Unmarried': 4, 'Wife': 5
}
relationship_options = sorted(list(relationship_mapping.keys()))

race_mapping = {
    'Amer-Indian-Eskimo': 0, 'Asian-Pac-Islander': 1, 'Black': 2, 'Other': 3,
    'White': 4
}
race_options = sorted(list(race_mapping.keys()))

gender_mapping = {
    'Female': 0, 'Male': 1
}
gender_options = sorted(list(gender_mapping.keys()))

native_country_options = [
    'Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic',
    'Ecuador', 'El-Salvador', 'England', 'France', 'Germany', 'Greece',
    'Guatemala', 'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary',
    'India', 'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico',
    'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines', 'Poland',
    'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan', 'Thailand',
    'Trinadad&Tobago', 'United-States', 'Vietnam', 'Yugoslavia'
]
native_country_mapping = {country: i for i, country in enumerate(native_country_options)}
default_native_country_encoded = native_country_mapping.get('United-States', 0)


# --- 3. Preprocessing Function for Input Data ---
def preprocess_data(df):
    """
    Applies the same preprocessing steps as done during model training,
    excluding scaling, as the loaded pipeline will handle it.
    """
    processed_df = df.copy()

    for col in ['workclass', 'occupation']:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].replace('?', 'Others')

    if 'workclass' in processed_df.columns:
        processed_df = processed_df[~processed_df['workclass'].isin(['Without-pay', 'Never-worked'])]

    if 'age' in processed_df.columns:
        processed_df = processed_df[(processed_df['age'] <= 75) & (processed_df['age'] >= 17)]
    if 'educational-num' in processed_df.columns:
        processed_df = processed_df[(processed_df['educational-num'] <= 16) & (processed_df['educational-num'] >= 5)]

    for col_to_drop in ['education', 'experience']:
        if col_to_drop in processed_df.columns:
            processed_df = processed_df.drop(columns=[col_to_drop])

    if 'workclass' in processed_df.columns:
        processed_df['workclass'] = processed_df['workclass'].map(workclass_mapping).fillna(workclass_mapping['Others'])
    if 'marital-status' in processed_df.columns:
        processed_df['marital-status'] = processed_df['marital-status'].map(marital_status_mapping).fillna(marital_status_mapping['Never-married'])
    if 'occupation' in processed_df.columns:
        processed_df['occupation'] = processed_df['occupation'].map(occupation_mapping).fillna(occupation_mapping['Others'])
    if 'relationship' in processed_df.columns:
        processed_df['relationship'] = processed_df['relationship'].map(relationship_mapping).fillna(relationship_mapping['Not-in-family'])
    if 'race' in processed_df.columns:
        processed_df['race'] = processed_df['race'].map(race_mapping).fillna(race_mapping['White'])
    if 'gender' in processed_df.columns:
        processed_df['gender'] = processed_df['gender'].map(gender_mapping).fillna(gender_mapping['Male'])
    if 'native-country' in processed_df.columns:
        processed_df['native-country'] = processed_df['native-country'].map(native_country_mapping).fillna(default_native_country_encoded)

    for col in processed_df.columns:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0)

    final_columns_order = [
        'age', 'workclass', 'fnlwgt', 'educational-num', 'marital-status',
        'occupation', 'relationship', 'race', 'gender', 'capital-gain',
        'capital-loss', 'hours-per-week', 'native-country'
    ]

    reordered_df = pd.DataFrame(columns=final_columns_order)
    for col in final_columns_order:
        if col in processed_df.columns:
            reordered_df[col] = processed_df[col]
        else:
            reordered_df[col] = 0

    reordered_df.reset_index(drop=True, inplace=True)

    return reordered_df


# --- 4. Streamlit App Layout ---
st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns **>50K** or **≤50K** based on various features.")
st.markdown("---")

# --- Sidebar for Input Features ---
st.sidebar.header("Input Employee Details")

# Numerical Inputs
age = st.sidebar.slider("Age", min_value=17, max_value=75, value=30)
educational_num = st.sidebar.slider(
    "Educational Number",
    min_value=5, max_value=16, value=9,
    help="A numerical representation of the highest education level achieved."
)
st.sidebar.markdown(
    """
    **Educational Number Mapping:**
    * **5:** 5th-6th grade
    * **7:** 11th grade
    * **9:** HS-grad (High School Graduate)
    * **10:** Some-college
    * **13:** Bachelors
    * **14:** Masters
    * **16:** Doctorate (PhD)
    """
)

capital_gain = st.sidebar.number_input("Capital Gain (USD)", min_value=0, max_value=100000, value=0, step=100)
capital_loss = st.sidebar.number_input("Capital Loss (USD)", min_value=0, max_value=4500, value=0, step=10)
hours_per_week = st.sidebar.slider("Hours per Week", min_value=1, max_value=99, value=40)

# Categorical Inputs (using predefined options and mappings)
workclass = st.sidebar.selectbox("Workclass", workclass_options, index=workclass_options.index('Private') if 'Private' in workclass_options else 0)
marital_status = st.sidebar.selectbox("Marital Status", marital_status_options, index=marital_status_options.index('Never-married') if 'Never-married' in marital_status_options else 0)
occupation = st.sidebar.selectbox("Occupation", occupation_options, index=occupation_options.index('Tech-support') if 'Tech-support' in occupation_options else 0)
relationship = st.sidebar.selectbox("Relationship", relationship_options, index=relationship_options.index('Not-in-family') if 'Not-in-family' in relationship_options else 0)
race = st.sidebar.selectbox("Race", race_options, index=race_options.index('White') if 'White' in race_options else 0)
gender = st.sidebar.selectbox("Gender", gender_options, index=gender_options.index('Male') if 'Male' in gender_options else 0)
native_country = st.sidebar.selectbox("Native Country", native_country_options, index=native_country_options.index('United-States') if 'United-States' in native_country_options else 0)


# --- 5. Prepare Single Input DataFrame ---
TYPICAL_FNLWGT_VALUE = 200000 # Hardcoded typical value for fnlwgt

input_data_raw = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [TYPICAL_FNLWGT_VALUE],
    'educational-num': [educational_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
})

# --- Display Raw Input in a user-friendly way ---
st.write("### 📝 Your Input Details:")
col1, col2 = st.columns(2) # Use columns for a cleaner layout

with col1:
    st.write(f"**Age:** {age}")
    st.write(f"**Workclass:** {workclass}")
    st.write(f"**Educational Number:** {educational_num}")
    st.write(f"**Marital Status:** {marital_status}")
    st.write(f"**Occupation:** {occupation}")
    st.write(f"**Relationship:** {relationship}")

with col2:
    st.write(f"**Race:** {race}")
    st.write(f"**Gender:** {gender}")
    st.write(f"**Capital Gain:** ${capital_gain:,.0f}")
    st.write(f"**Capital Loss:** ${capital_loss:,.0f}")
    st.write(f"**Hours per Week:** {hours_per_week}")
    st.write(f"**Native Country:** {native_country}")
    st.write(f"**Census Weight (fnlwgt):** {TYPICAL_FNLWGT_VALUE}") # Show the hardcoded value


# Preprocess the single input data
input_data_processed = preprocess_data(input_data_raw)

# Check if processed data is empty (e.g., due to filtering)
if input_data_processed.empty:
    st.warning("Input data was filtered out due to invalid values after preprocessing. Please adjust inputs.")
else:
    # --- 6. Predict Button ---
    if st.button("Predict Salary Class"):
        try:
            predicted_label_string = model.predict(input_data_processed)[0]
            prediction_label = predicted_label_string

            st.success(f"✅ Prediction: **{prediction_label}**")

            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_data_processed)
                # Get the actual class labels from the model's final estimator
                model_classes = model.named_steps['model'].classes_.tolist()
                proba_output = {model_classes[i]: f"{probabilities[0][i]:.2%}" for i in range(len(model_classes))} # Format as percentage
                st.info(f"Confidence: {proba_output}")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.write("Please ensure all input values are valid and the model is correctly loaded.")

