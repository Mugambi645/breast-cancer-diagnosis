import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os


@st.cache_data
def get_clean_data():
    """Load and preprocess the dataset."""
    try:
        data = pd.read_csv("data/data.csv")
        data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
        data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
        return data
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()


def load_models():
    """Load the ML model and scaler."""
    try:
        with open("model/model.pkl", "rb") as model_file, open("model/scaler.pkl", "rb") as scaler_file:
            model = pickle.load(model_file)
            scaler = pickle.load(scaler_file)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None


def add_sidebar():
    """Create sidebar with user inputs."""
    st.sidebar.header("🔬 Cell Nuclei Measurements")

    data = get_clean_data()
    if data.empty:
        return {}

    input_dict = {}
    feature_columns = data.columns.drop("diagnosis")

    for col in feature_columns:
        input_dict[col] = st.sidebar.slider(
            col.replace("_", " ").title(),
            min_value=float(0),
            max_value=float(data[col].max()),
            value=float(data[col].mean()),
        )

    return input_dict


def get_scaled_values(input_dict, data):
    """Normalize input values based on dataset min-max scaling."""
    X = data.drop(['diagnosis'], axis=1)
    return {key: (value - X[key].min()) / (X[key].max() - X[key].min()) for key, value in input_dict.items()}


def get_radar_chart(input_data):
    """Generate a radar chart for visualization."""
    categories = [
        "Radius", "Texture", "Perimeter", "Area", "Smoothness",
        "Compactness", "Concavity", "Concave Points", "Symmetry", "Fractal Dimension"
    ]

    input_data = get_scaled_values(input_data, get_clean_data())

    fig = go.Figure()

    for category, name in zip(
        ['mean', 'se', 'worst'], ["Mean Value", "Standard Error", "Worst Value"]
    ):
        fig.add_trace(go.Scatterpolar(
            r=[input_data[f"{c.lower()}_{category}"] for c in categories],
            theta=categories,
            fill="toself",
            name=name
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
    )

    return fig


def add_predictions(input_data):
    """Predict and display the results."""
    model, scaler = load_models()
    if not model or not scaler:
        st.error("Model or scaler could not be loaded.")
        return

    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[0]

    st.subheader("🩺 Cell Cluster Prediction")
    st.write("The predicted diagnosis is:")

    if prediction[0] == 0:
        st.success("✅ **Benign**")
    else:
        st.error("⚠️ **Malignant**")

    st.write(f"**Probability of being Benign:** {prob[0]:.2f}")
    st.write(f"**Probability of being Malignant:** {prob[1]:.2f}")

    st.caption("⚠️ This tool assists medical professionals but does not replace a professional diagnosis.")


def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load CSS styles
    if os.path.exists("assets/style.css"):
        with open("assets/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Get user inputs from sidebar
    input_data = add_sidebar()

    # Display title and description
    st.title("🩺 Breast Cancer Predictor")
    st.write(
        "This AI-powered app predicts whether a breast mass is benign or malignant based on "
        "nuclei measurements. Adjust the values in the sidebar to simulate different cases."
    )

    # Layout: Radar Chart & Predictions
    col1, col2 = st.columns([3, 1])

    with col1:
        st.plotly_chart(get_radar_chart(input_data))

    with col2:
        add_predictions(input_data)


if __name__ == "__main__":
    main()
