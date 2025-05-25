# app.py (Corrected for NameError and DeprecationWarning)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score
from federated_simulation import (
    load_and_prep_data,
    train_initial_client_models, 
    get_client_predictions_proba,
    poison_predictions_simple_flip,
    NUM_CLIENTS,
    POISONED_CLIENT_INDEX
)

# --- Initialize session state variables ---
def initialize_app_state():
    if 'initial_setup_done' not in st.session_state:
        with st.spinner("Performing initial data load and model training... This may take a few seconds."):
            print("APP_INIT: initial_setup_done not in session_state, calling load_and_train_initial_models_cached()")
            setup_results = load_and_train_initial_models_cached()
            
            if setup_results is None or setup_results[0] is None:
                st.error("Critical Error: Failed during initial app setup (data loading or model training). Please check Framingham.csv and console logs for messages from federated_simulation.py.")
                st.session_state.simulation_ready = False
                st.stop() 

            st.session_state.X_test_fixed, st.session_state.y_test_fixed, \
            st.session_state.feature_names, st.session_state.client_models_list = setup_results
            
            if any(val is None for val in [st.session_state.X_test_fixed, st.session_state.y_test_fixed, 
                                           st.session_state.feature_names, st.session_state.client_models_list]):
                st.error("Critical Error: Essential data components are missing after setup. Dashboard cannot start.")
                st.session_state.simulation_ready = False
                st.stop()
            
            st.session_state.initial_setup_done = True
            st.session_state.simulation_ready = True
            print("INITIALIZATION (app.py): Data loaded and initial client models (list) stored in session_state.")
        
        reset_simulation_dynamic_state(calculate_initial_metrics=True)
    
    dynamic_states_defaults = {
        'attack_active': False, 'aggregation_method': "Mean", 'metrics_history': [],
        'round_counter': 0, 'current_metrics': {},
        'single_patient_prediction_proba': None
    }
    for key, default_val in dynamic_states_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_val
            
    default_patient_ui_inputs = {
        'ui_pat_age': 50, 'ui_pat_sex': "Female", 'ui_pat_education': "High School/GED",
        'ui_pat_smoker': "No", 'ui_pat_cigs': 0, 'ui_pat_bpmeds': "No",
        'ui_pat_stroke': "No", 'ui_pat_hyp': "No", 'ui_pat_diabetes': "No",
        'ui_pat_totchol': 200.0, 'ui_pat_sysbp': 120.0, 'ui_pat_diabp': 80.0,
        'ui_pat_bmi': 25.0, 'ui_pat_hr': 75, 'ui_pat_glucose': 80.0
    }
    for key, default_val in default_patient_ui_inputs.items():
        if key not in st.session_state:
            st.session_state[key] = default_val


@st.cache_data(show_spinner=False)
def load_and_train_initial_models_cached():
    print("CACHE_DATA: load_and_train_initial_models_cached called.")
    data_load_result = load_and_prep_data()
    if data_load_result is None or data_load_result[0] is None:
        return None 
    
    X_pool, y_pool, X_test_fixed, y_test_fixed, feature_names = data_load_result
    
    if any(df is None for df in [X_pool, y_pool, X_test_fixed, y_test_fixed, feature_names]):
        return None
        
    client_models_list = train_initial_client_models(X_pool, y_pool)
    print(f"CACHE_DATA: Trained {len(client_models_list)} client models.")
    return X_test_fixed, y_test_fixed, feature_names, client_models_list


def reset_simulation_dynamic_state(calculate_initial_metrics=False):
    print("RESET_DYNAMIC_STATE: Resetting interactive simulation variables.")
    st.session_state.attack_active = False
    st.session_state.aggregation_method = "Mean"
    st.session_state.metrics_history = []
    st.session_state.round_counter = 0
    st.session_state.current_metrics = {}
    st.session_state.single_patient_prediction_proba = None
    
    default_patient_ui_inputs = {
        'ui_pat_age': 50, 'ui_pat_sex': "Female", 'ui_pat_education': "High School/GED",
        'ui_pat_smoker': "No", 'ui_pat_cigs': 0, 'ui_pat_bpmeds': "No",
        'ui_pat_stroke': "No", 'ui_pat_hyp': "No", 'ui_pat_diabetes': "No",
        'ui_pat_totchol': 200.0, 'ui_pat_sysbp': 120.0, 'ui_pat_diabp': 80.0,
        'ui_pat_bmi': 25.0, 'ui_pat_hr': 75, 'ui_pat_glucose': 80.0
    }
    for key, default_val in default_patient_ui_inputs.items():
        st.session_state[key] = default_val

    if calculate_initial_metrics and st.session_state.get('simulation_ready', False):
        print("RESET_DYNAMIC_STATE: Calculating initial metrics for Round 0.")
        simulate_global_evaluation_logic(is_initial_call=True)
    print("SIMULATION INTERACTIVE STATE RESET")


def simulate_global_evaluation_logic(is_initial_call=False):
    if not is_initial_call:
        st.session_state.round_counter += 1
    current_round = st.session_state.round_counter
    client_probas_this_round = []

    if st.session_state.X_test_fixed.empty or not st.session_state.client_models_list:
        st.warning("Test data or client models not available for simulation round.")
        st.session_state.current_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'mses': {}}
        return

    for i in range(NUM_CLIENTS):
        model = st.session_state.client_models_list[i]
        proba = get_client_predictions_proba(model, st.session_state.X_test_fixed)
        if i == POISONED_CLIENT_INDEX and st.session_state.attack_active:
            proba = poison_predictions_simple_flip(proba)
        client_probas_this_round.append(proba)

    if not client_probas_this_round or not any(p.size > 0 for p in client_probas_this_round):
        st.warning("No client probabilities generated for this round.")
        st.session_state.current_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'mses': {}}
        return

    if st.session_state.aggregation_method == "Mean":
        aggregated_proba = np.array([p for p in client_probas_this_round if p.size > 0]).mean(axis=0)
    else: 
        aggregated_proba = np.median(np.array([p for p in client_probas_this_round if p.size > 0]), axis=0)
    
    if aggregated_proba.size == 0:
        st.warning("Aggregated probabilities are empty.")
        st.session_state.current_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'mses': {}}
        return

    global_preds = (aggregated_proba >= 0.5).astype(int)
    
    acc, prec, rec = 0.0, 0.0, 0.0
    if not st.session_state.y_test_fixed.empty and st.session_state.y_test_fixed.nunique() > 1 and len(global_preds) == len(st.session_state.y_test_fixed):
        acc = accuracy_score(st.session_state.y_test_fixed, global_preds)
        prec = precision_score(st.session_state.y_test_fixed, global_preds, zero_division=0)
        rec = recall_score(st.session_state.y_test_fixed, global_preds, zero_division=0)
    
    mse_scores_this_round = {}
    if client_probas_this_round and all(p.size > 0 for p in client_probas_this_round):
        mean_of_client_probas = np.array(client_probas_this_round).mean(axis=0)
        if mean_of_client_probas.size > 0:
            for i, client_p in enumerate(client_probas_this_round):
                if client_p.size == mean_of_client_probas.size:
                    mse = mean_squared_error(mean_of_client_probas, client_p)
                    client_name = f"Client {i+1}"
                    if i == POISONED_CLIENT_INDEX: client_name += " (Attacker)"
                    mse_scores_this_round[client_name] = mse
                else:
                    mse_scores_this_round[f"Client {i+1} (MSE Error)"] = -1 
            
    st.session_state.current_metrics = {'accuracy': acc, 'precision': prec, 'recall': rec, 'mses': mse_scores_this_round}
    
    if not is_initial_call or not st.session_state.metrics_history:
        st.session_state.metrics_history.append({
            "Round": current_round, "Accuracy": acc, "Precision": prec, "Recall": rec,
            "Attack Active": st.session_state.attack_active,
            "Aggregation": st.session_state.aggregation_method
        })
    print(f"SIM EVAL ROUND {current_round}: Acc={acc:.4f}, Attack={st.session_state.attack_active}, Agg={st.session_state.aggregation_method}")


# --- Main App ---
st.set_page_config(layout="wide", page_title="Interactive FL Demo")
st.title("🛡️ Federated Learning: Interactive Attack & Defense Simulation")

initialize_app_state()

if not st.session_state.get('simulation_ready', False):
    st.stop()

# --- Sidebar Controls ---
st.sidebar.header("Simulation Controls")
if st.sidebar.button("⚠️ Hard Reset (Reload Data & Retrain Initial Models)"):
    st.cache_data.clear()
    st.session_state.clear()
    st.rerun()

attack_active_toggle = st.sidebar.toggle("Activate Attack on Client 5", value=st.session_state.attack_active, key="attack_toggle_widget")
if attack_active_toggle != st.session_state.attack_active:
    st.session_state.attack_active = attack_active_toggle
    simulate_global_evaluation_logic() 
    st.rerun()

aggregation_method_radio = st.sidebar.radio(
    "Aggregation Method", ["Mean", "Median"],
    index=["Mean", "Median"].index(st.session_state.aggregation_method),
    key="aggregation_radio_widget"
)
if aggregation_method_radio != st.session_state.aggregation_method:
    st.session_state.aggregation_method = aggregation_method_radio
    simulate_global_evaluation_logic()
    st.rerun()

if st.sidebar.button("Simulate Global Evaluation Round", key="simulate_next_round_button"):
    simulate_global_evaluation_logic()
    st.rerun()

if st.sidebar.button("Reset Interactive State (Keep Models)", key="soft_reset_button_sidebar"):
    reset_simulation_dynamic_state(calculate_initial_metrics=True)
    st.rerun()

st.sidebar.markdown("---")
# CORRECTED LINE FOR IMAGE:
try:
    st.sidebar.image("feature_table.png", caption="Framingham Dataset Features", use_container_width=True)
except FileNotFoundError:
    st.sidebar.warning("feature_table.png not found.")


# --- Main Display Area ---
st.header("Live Simulation Status")
col1, col2, col3 = st.columns(3)
col1.metric("Current Round", st.session_state.round_counter)
col2.metric("Attack on Client 5", "ACTIVE 💣" if st.session_state.attack_active else "Inactive ✅",
            help="If active, Client 5 provides poisoned (flipped) predictions for the global evaluation.")
col3.metric("Aggregation Method", st.session_state.aggregation_method)

st.markdown("---")
# --- Patient Data Input and Analysis Section ---
patient_inputs_expander = st.expander("👤 Individual Patient Analysis (Simulated Prediction)", expanded=True)
with patient_inputs_expander:
    st.info("Input hypothetical patient data. Click 'Analyse Patient' to see a simulated prediction from the current global model state.")
    
    feature_cols_from_state = st.session_state.get('feature_names', [])
    
    # Define mapping dictionaries *outside* the columns, in a scope accessible by both
    sex_map = {"Female": 0, "Male": 1}
    education_options = {"Unknown":0.0, "Some High School": 1.0, "High School/GED": 2.0, "Some College/Vocational": 3.0, "College Degree+": 4.0}
    current_smoker_map = {"No": 0, "Yes": 1}
    bp_meds_map = {"No": 0, "Yes": 1} # MOVED DEFINITION HERE
    prevalent_stroke_map = {"No": 0, "Yes": 1} # MOVED DEFINITION HERE
    prevalent_hyp_map = {"No": 0, "Yes": 1} # MOVED DEFINITION HERE
    diabetes_map = {"No": 0, "Yes": 1} # MOVED DEFINITION HERE

    patient_form_cols = st.columns(2)
    with patient_form_cols[0]:
        st.session_state.ui_pat_age = st.number_input("Age", min_value=20, max_value=100, value=st.session_state.ui_pat_age, step=1, key="ui_pat_age_k")
        st.session_state.ui_pat_sex = st.selectbox("Sex", options=list(sex_map.keys()), index=list(sex_map.keys()).index(st.session_state.ui_pat_sex), key="ui_pat_sex_k")
        st.session_state.ui_pat_education = st.selectbox("Education Level", options=list(education_options.keys()), index=list(education_options.keys()).index(st.session_state.ui_pat_education), key="ui_pat_education_k")
        st.session_state.ui_pat_smoker = st.selectbox("Current Smoker", options=list(current_smoker_map.keys()), index=list(current_smoker_map.keys()).index(st.session_state.ui_pat_smoker), key="ui_pat_smoker_k")
        st.session_state.ui_pat_cigs = st.number_input("Cigarettes Per Day", min_value=0, max_value=100, value=st.session_state.ui_pat_cigs, step=1, key="ui_pat_cigs_k", disabled=(current_smoker_map[st.session_state.ui_pat_smoker]==0))
        
    with patient_form_cols[1]:
        st.session_state.ui_pat_bpmeds = st.selectbox("On BP Medication", options=list(bp_meds_map.keys()), index=list(bp_meds_map.keys()).index(st.session_state.ui_pat_bpmeds), key="ui_pat_bpmeds_k")
        st.session_state.ui_pat_stroke = st.selectbox("History of Stroke", options=list(prevalent_stroke_map.keys()), index=list(prevalent_stroke_map.keys()).index(st.session_state.ui_pat_stroke), key="ui_pat_stroke_k")
        st.session_state.ui_pat_hyp = st.selectbox("History of Hypertension", options=list(prevalent_hyp_map.keys()), index=list(prevalent_hyp_map.keys()).index(st.session_state.ui_pat_hyp), key="ui_pat_hyp_k")
        st.session_state.ui_pat_diabetes = st.selectbox("Diabetic Status", options=list(diabetes_map.keys()), index=list(diabetes_map.keys()).index(st.session_state.ui_pat_diabetes), key="ui_pat_diabetes_k")
        st.session_state.ui_pat_totchol = st.number_input("Total Cholesterol (mg/dL)", min_value=100.0, max_value=600.0, value=float(st.session_state.ui_pat_totchol), step=1.0, format="%.1f", key="ui_pat_totchol_k")
        st.session_state.ui_pat_sysbp = st.number_input("Systolic BP (mmHg)", min_value=80.0, max_value=300.0, value=float(st.session_state.ui_pat_sysbp), step=0.5, format="%.1f", key="ui_pat_sysbp_k")
        st.session_state.ui_pat_diabp = st.number_input("Diastolic BP (mmHg)", min_value=50.0, max_value=200.0, value=float(st.session_state.ui_pat_diabp), step=0.5, format="%.1f", key="ui_pat_diabp_k")
        st.session_state.ui_pat_bmi = st.number_input("BMI", min_value=15.0, max_value=60.0, value=float(st.session_state.ui_pat_bmi), step=0.01, format="%.2f", key="ui_pat_bmi_k")
        st.session_state.ui_pat_hr = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=int(st.session_state.ui_pat_hr), step=1, key="ui_pat_hr_k")
        st.session_state.ui_pat_glucose = st.number_input("Glucose Level (mg/dL)", min_value=40.0, max_value=400.0, value=float(st.session_state.ui_pat_glucose), step=1.0, format="%.1f", key="ui_pat_glucose_k")

    patient_input_for_df_current = {
        'age': st.session_state.ui_pat_age,
        'male': sex_map[st.session_state.ui_pat_sex],
        'education': education_options[st.session_state.ui_pat_education],
        'currentSmoker': current_smoker_map[st.session_state.ui_pat_smoker],
        'cigsPerDay': float(st.session_state.ui_pat_cigs if current_smoker_map[st.session_state.ui_pat_smoker] == 1 else 0),
        'BPMeds': float(bp_meds_map[st.session_state.ui_pat_bpmeds]),
        'prevalentStroke': float(prevalent_stroke_map[st.session_state.ui_pat_stroke]),
        'prevalentHyp': float(prevalent_hyp_map[st.session_state.ui_pat_hyp]),
        'diabetes': float(diabetes_map[st.session_state.ui_pat_diabetes]),
        'totChol': float(st.session_state.ui_pat_totchol),
        'sysBP': float(st.session_state.ui_pat_sysbp),
        'diaBP': float(st.session_state.ui_pat_diabp),
        'BMI': float(st.session_state.ui_pat_bmi),
        'heartRate': float(st.session_state.ui_pat_hr),
        'glucose': float(st.session_state.ui_pat_glucose)
    }

    if st.button("Analyse This Patient", key="analyze_patient_button_main"):
        patient_df_data_ordered = {}
        if not feature_cols_from_state:
            st.error("Feature names not loaded from simulation. Cannot make patient prediction.")
            st.session_state.single_patient_prediction_proba = None
        else:
            for col in feature_cols_from_state:
                patient_df_data_ordered[col] = patient_input_for_df_current.get(col, 0.0) 
            
            patient_df = pd.DataFrame([patient_df_data_ordered], columns=feature_cols_from_state)
            client_patient_probas = []

            if not patient_df.empty and st.session_state.client_models_list:
                for i in range(NUM_CLIENTS):
                    model = st.session_state.client_models_list[i] 
                    try:
                        proba = get_client_predictions_proba(model, patient_df)[0] 
                        if i == POISONED_CLIENT_INDEX and st.session_state.attack_active:
                            proba = poison_predictions_simple_flip(np.array([proba]))[0]
                        client_patient_probas.append(proba)
                    except Exception as e:
                        st.error(f"Error predicting with client {i+1} model for patient: {e}")
                        client_patient_probas.append(0.5) 
            
            if client_patient_probas:
                if st.session_state.aggregation_method == "Mean":
                    aggregated_patient_proba = np.mean(client_patient_probas)
                else: 
                    aggregated_patient_proba = np.median(client_patient_probas)
                st.session_state.single_patient_prediction_proba = aggregated_patient_proba
            else:
                st.session_state.single_patient_prediction_proba = None
        st.rerun()

    if st.session_state.single_patient_prediction_proba is not None:
        st.markdown("---")
        st.subheader("Current Patient Analysis Results:")
        pred_proba = st.session_state.single_patient_prediction_proba
        risk_level = "High Risk 💔" if pred_proba >= 0.5 else "Low Risk ❤️"
        pred_color = "red" if risk_level == "High Risk 💔" else "green"
        
        st.markdown(f"Predicted 10-Year CHD Risk: <span style='color:{pred_color}; font-size:1.2em; font-weight:bold;'>{pred_proba:.1%} ({risk_level})</span>", unsafe_allow_html=True)
        st.progress(float(pred_proba))
        
        st.markdown("**General Risk Factors (Illustrative based on input):**")
        if patient_input_for_df_current.get('age', 0) > 60: st.markdown(f"- Age ({patient_input_for_df_current.get('age', 'N/A')}): <span style='color:red;'>High Impact</span>", unsafe_allow_html=True)
        if patient_input_for_df_current.get('totChol', 0) > 240: st.markdown(f"- Cholesterol ({patient_input_for_df_current.get('totChol', 'N/A')}): <span style='color:red;'>High Impact</span>", unsafe_allow_html=True)
        if patient_input_for_df_current.get('sysBP', 0) > 140: st.markdown(f"- Systolic BP ({patient_input_for_df_current.get('sysBP', 'N/A')}): <span style='color:red;'>High Impact</span>", unsafe_allow_html=True)
        if patient_input_for_df_current.get('currentSmoker') == 1: st.markdown(f"- Current Smoker (Yes): <span style='color:orange;'>Medium Impact</span>", unsafe_allow_html=True)
        if patient_input_for_df_current.get('diabetes') == 1: st.markdown(f"- Diabetes (Yes): <span style='color:orange;'>Medium Impact</span>", unsafe_allow_html=True)


st.markdown("---")
# Global Metrics Display
st.header("Global Model Performance Dashboard")
current_metrics_display = st.session_state.get('current_metrics', {})
if current_metrics_display.get('accuracy') is not None:
    col_acc, col_prec, col_recall = st.columns(3)
    col_acc.metric("Global Accuracy", f"{current_metrics_display['accuracy']:.2%}")
    col_prec.metric("Global Precision", f"{current_metrics_display.get('precision', 0.0):.2%}")
    col_recall.metric("Global Recall", f"{current_metrics_display.get('recall', 0.0):.2%}")
else:
    st.write("Global performance metrics not available. Simulate a global evaluation round or reset.")


if st.session_state.metrics_history:
    history_df = pd.DataFrame(st.session_state.metrics_history)
    st.subheader("Performance Metrics Over Rounds")
    if not history_df.empty and "Round" in history_df.columns and "Accuracy" in history_df.columns:
        history_df_for_chart = history_df.set_index("Round")
        st.line_chart(history_df_for_chart[['Accuracy', 'Precision', 'Recall']])
    else:
        st.write("Performance history data is not ready for plotting.")
    with st.expander("View Performance History Table"):
        st.dataframe(history_df)
else:
    st.write("No global evaluation rounds run yet.")

mse_scores_display_main = st.session_state.current_metrics.get('mses', {})
if mse_scores_display_main:
    st.subheader("Client MSEs (vs. Current Round's Mean Predictions)")
    mse_df = pd.DataFrame(list(mse_scores_display_main.items()), columns=['Client', 'MSE Value'])
    if not mse_df.empty:
        mse_df_sorted = mse_df.sort_values(by="MSE Value", ascending=False)
        st.bar_chart(mse_df_sorted.set_index('Client'))
        attacker_status_msg = "(Attacker is Active)" if st.session_state.attack_active else "(Attacker is Inactive)"
        st.caption(f"Client 5 is the designated attacker. MSE calculated against the mean of current round's client predictions. {attacker_status_msg}")
    else:
        st.write("MSE scores are empty for this round.")
else:
    st.write("Run a global evaluation round to see MSE scores.")

st.markdown("---")
st.markdown(
    "**How to Use:** Adjust **Simulation Controls** in the sidebar. "
    "The Global Performance metrics and Client MSEs will update automatically based on your selections. "
    "Use the **Patient Analysis** section to get a prediction for hypothetical inputs under current settings."
)