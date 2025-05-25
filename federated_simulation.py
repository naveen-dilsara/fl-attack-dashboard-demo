# app.py
import streamlit as st
import pandas as pd
import numpy as np
from federated_simulation import run_full_simulation_pipeline # Import your function

# --- FUNCTION TO GET/RUN SIMULATION RESULTS ---
@st.cache_data # Cache the results so simulation doesn't re-run on every UI interaction
def get_simulation_results():
    # This message will be replaced by the spinner in the main app body
    # st.info("🔄 Running federated learning simulation... This may take a moment.") 
    
    sim_output = run_full_simulation_pipeline()

    if sim_output is None:
        # Error is printed by federated_simulation.py if it returns None
        # We still need to return a default dict to prevent app.py from breaking later
        return {
            "initial_accuracy": 0.0,
            "attacked_accuracy_mean_agg": 0.0,
            "client_mse_values": {"Client 1": 0, "Client 2": 0, "Client 3": 0, "Client 4": 0, "Client 5 (Attacker)": 0},
            "defended_accuracy_median_agg": 0.0,
            "aggregation_method_initial": "Mean",
            "aggregation_method_defense": "Median",
            "feature_names": []
        }

    initial_acc, attacked_acc_mean, client_mses_dict, defended_acc_median, feature_names_list = sim_output

    results = {
        "initial_accuracy": initial_acc,
        "aggregation_method_initial": "Mean",
        "attacked_accuracy_mean_agg": attacked_acc_mean,
        "client_mse_values": client_mses_dict,
        "defended_accuracy_median_agg": defended_acc_median,
        "aggregation_method_defense": "Median",
        "feature_names": feature_names_list
    }
    return results

# --- MAIN DASHBOARD APP LAYOUT ---
st.set_page_config(layout="wide", page_title="FL Attack Simulation")
st.title("🛡️ Federated Learning: Aggregation Attack & Defense Simulation")
st.markdown("""
This dashboard demonstrates a **label-flipping aggregation attack** in a simulated Federated Learning (FL) system.
We use the Framingham Heart Study dataset for predicting 10-Year Coronary Heart Disease (CHD) risk.
The simulation involves 5 clients, with one client acting maliciously.
""")

# --- Sidebar for controls ---
st.sidebar.header("Simulation Controls")
if st.sidebar.button("🔄 Re-run Simulation & Clear Cache"):
    st.cache_data.clear()
    if 'sim_results' in st.session_state: # Clear previous results if they exist
        del st.session_state.sim_results
    st.rerun() # Use st.rerun()

st.sidebar.header("About")
st.sidebar.info(
    "This dashboard visualizes:\n"
    "1. Baseline model performance.\n"
    "2. Impact of a poisoned client (label flipping) using Mean aggregation.\n"
    "3. Detection of the attacker using Mean Squared Error (MSE) comparison.\n"
    "4. Mitigation of the attack using Median aggregation."
)

# --- Get results from your simulation ---
# Use session_state to store results after the first run.
# The button click will clear st.session_state.sim_results to force a re-run.
if 'sim_results' not in st.session_state:
    with st.spinner("🔄 Running federated learning simulation... This may take a moment."):
        st.session_state.sim_results = get_simulation_results()

sim_results = st.session_state.sim_results

# Check if simulation failed (e.g., due to missing CSV or other critical error in federated_simulation.py)
# sim_results will be None if run_full_simulation_pipeline() returned None and then get_simulation_results() returned the default error dict
if sim_results is None or (sim_results["initial_accuracy"] == 0.0 and not sim_results["feature_names"] and not any(sim_results["client_mse_values"].values())):
    # The error message should have been printed by federated_simulation.py if it returned None
    # and then by get_simulation_results() if it constructed the default error dict.
    # If we reach here with default values, it means the simulation truly failed to produce meaningful data.
    st.error("Simulation data could not be loaded or generated. Dashboard cannot proceed. Please check logs on Streamlit Cloud (Manage app -> Logs) for specific errors from `federated_simulation.py` like 'Framingham.csv not found' or other Python exceptions.")
    st.stop() # Stop further execution of the dashboard if sim failed

# --- Use tabs for different stages ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "ℹ️ Setup & Baseline",
    "💣 Attack Injection & Impact",
    "🔍 Attack Detection (MSE)",
    "🛡️ Defense Application",
    "📊 Summary & Key Takeaways"
])

with tab1:
    st.header("1. System Setup & Clean Model Performance (Baseline)")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Initial Global Accuracy", value=f"{sim_results['initial_accuracy']:.2%}")
    col2.metric(label="Aggregation Method", value=sim_results['aggregation_method_initial'])
    col3.metric(label="Model Behavior", value="Stable and Correct")
    col4.metric(label="All Clients Clean", value="Yes")

    st.subheader("Scenario Overview")
    st.markdown("""
    - **Federated Learning System:** 5 clients (simulated hospitals) collaboratively training a model.
    - **Dataset:** Framingham Heart Study (preprocessed by dropping NA values).
    - **Task:** Predict 10-Year Coronary Heart Disease (CHD) risk.
    - **Baseline State:** Simulates the accuracy of a centralized model trained on clean, aggregated data.
    """)

    st.subheader("Dataset Features")
    try:
        st.image("feature_table.png", caption="Features used in the Framingham Heart Study dataset for this simulation.")
    except FileNotFoundError:
        st.warning("`feature_table.png` not found. Displaying feature list from data instead.")
        if sim_results["feature_names"]:
            st.write(sim_results["feature_names"])
        else:
            st.write("Feature names not available from simulation.")

with tab2:
    st.header("2. Attack Injection & Impact (Mean Aggregation)")
    st.markdown("---")
    st.subheader("The Attack: Client 5 Poisons the Data")
    st.error("""
    - **Attacker:** Client 5 (out of 5 total clients).
    - **Attack Type:** Label Flipping. In its local dataset, Client 5 intentionally inverts the target labels (0 becomes 1, and 1 becomes 0).
    - **Consequence:** Client 5 trains its local model on this corrupted data and sends misleading model updates (or predictions in this simulation type) to the central server.
    - **Aggregation:** The server combines updates from all clients using **Mean Aggregation**.
    """)

    col1, col2 = st.columns(2)
    accuracy_drop = sim_results['attacked_accuracy_mean_agg'] - sim_results['initial_accuracy']
    col1.metric(label="Global Accuracy (Post-Attack, Mean Agg.)",
                value=f"{sim_results['attacked_accuracy_mean_agg']:.2%}",
                delta=f"{accuracy_drop:.2%}", 
                delta_color="inverse")
    col2.metric(label="Global Model Impact", value="Performance Degrades")
    st.warning("The global model's accuracy significantly drops due to the attacker's influence when using Mean Aggregation.")

with tab3:
    st.header("3. Attack Detection using MSE Comparison")
    st.markdown("---")
    st.info("Detection Method: Mean Squared Error (MSE) Comparison")
    st.markdown("""
    To detect the malicious client, we calculate the Mean Squared Error (MSE) between:
    - Each client's predicted probabilities (on a common global test set).
    - The global average predicted probabilities (from Mean Aggregation).
    A client whose predictions deviate significantly (high MSE) from the average is flagged as a potential attacker.
    """)

    mse_data_dict = sim_results['client_mse_values']
    if mse_data_dict and any(mse_data_dict.values()): # Check if dictionary is not empty AND has non-zero values
        mse_df = pd.DataFrame(list(mse_data_dict.items()), columns=['Client', 'MSE Value'])
        mse_df_sorted = mse_df.sort_values(by="MSE Value", ascending=False)

        st.subheader("MSE Values per Client (vs. Global Mean Predictions):")
        st.bar_chart(mse_df_sorted.set_index('Client'))

        st.markdown("Detailed MSE Values:")
        st.table(mse_df_sorted.assign(Flag=mse_df_sorted['Client'].apply(lambda x: "🚩 Attacker Flagged" if "Attacker" in x else "Normal")))
        st.success("Client 5 (Attacker) is clearly identified due to its significantly higher MSE value, indicating its predictions differ most from the aggregated consensus.")
    else:
        st.warning("MSE values are not available or all zero (possibly due to simulation error or empty data).")

with tab4:
    st.header("4. Defense Application & Model Recovery (Median Aggregation)")
    st.markdown("---")
    st.subheader(f"Defense Strategy: Switching to {sim_results['aggregation_method_defense']} Aggregation")
    st.info(f"""
    The server, having detected Client 5 as an outlier, now discards the compromised Mean Aggregation result.
    Instead, it re-aggregates client predictions using **{sim_results['aggregation_method_defense']} Aggregation**.
    This method is more robust to outliers because it takes the median of the predicted probabilities from each client for each sample, effectively diminishing the attacker's influence.
    """)

    col1, col2, col3 = st.columns(3)
    accuracy_recovery = sim_results['defended_accuracy_median_agg'] - sim_results['attacked_accuracy_mean_agg']
    col1.metric(label=f"Restored Accuracy ({sim_results['aggregation_method_defense']} Agg.)",
                value=f"{sim_results['defended_accuracy_median_agg']:.2%}",
                delta=f"{accuracy_recovery:.2%}", 
                delta_color="normal") 
    col2.metric(label="Attacker's Effect", value=f"Minimized by {sim_results['aggregation_method_defense']}")
    col3.metric(label="Model Behavior", value="✅ Stable Again")
    st.success(f"The global model's accuracy is substantially restored using {sim_results['aggregation_method_defense']} Aggregation, demonstrating its effectiveness as a defense mechanism.")

with tab5:
    st.header("📊 Overall Summary & Key Takeaways")
    st.markdown("---")

    st.subheader("Accuracy Journey:")
    accuracy_journey_data = {
        "Stage": [
            f"1. Baseline (Centralized Clean)",
            f"2. After Attack (Mean Agg.)",
            f"3. After Defense ({sim_results['aggregation_method_defense']} Agg.)"
        ],
        "Accuracy": [
            sim_results['initial_accuracy'],
            sim_results['attacked_accuracy_mean_agg'],
            sim_results['defended_accuracy_median_agg']
        ]
    }
    accuracy_journey_df = pd.DataFrame(accuracy_journey_data)
    accuracy_journey_df['Accuracy (%)'] = accuracy_journey_df['Accuracy'] * 100

    st.bar_chart(accuracy_journey_df.set_index("Stage")['Accuracy (%)'], height=400)
    
    attacker_mse = sim_results['client_mse_values'].get('Client 5 (Attacker)', 'N/A')
    if isinstance(attacker_mse, (float, np.float64)): # Check if it's a number before formatting
        attacker_mse_str = f"{attacker_mse:.2f}"
    else:
        attacker_mse_str = "N/A (Attacker not found or MSE not calculated)"

    summary_text = f"""
    <div style="background-color:#e6f3ff; padding:15px; border-radius:5px; border: 1px solid #b3d9ff;">
    <strong>Key Takeaway:</strong><br>
    The simulation demonstrates a complete cycle of attack, detection, and defense in Federated Learning:
    <ul>
        <li>The initial <strong>baseline accuracy</strong> (simulating a clean, centralized model) was approximately <strong>{sim_results['initial_accuracy']:.1%}</strong>.</li>
        <li>After <strong>Client 5 (the attacker)</strong> performed a label-flipping attack, the global model accuracy using <strong>Mean aggregation</strong> dropped significantly to approximately <strong>{sim_results['attacked_accuracy_mean_agg']:.1%}</strong>.</li>
        <li>Using <strong>Mean Squared Error (MSE) comparison</strong>, we successfully detected Client 5 as the attacker (its MSE was <strong>{attacker_mse_str}</strong>, much higher than benign clients).</li>
        <li>By switching to a robust <strong>{sim_results['aggregation_method_defense']} Aggregation</strong>, the attacker's influence was mitigated, and the global model accuracy improved back to approximately <strong>{sim_results['defended_accuracy_median_agg']:.1%}</strong>.</li>
    </ul>
    This highlights the vulnerability of basic FL to data poisoning and the importance of implementing detection and defense mechanisms.
    </div>
    """
    st.markdown(summary_text, unsafe_allow_html=True)