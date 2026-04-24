import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Health Prediction App",
    page_icon="🩺",
    layout="wide"
)

# Model theta
theta = np.array([0.669, 1.025, -0.680, 0.605, 0.385, 0.232])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_health(vegetables, fast_food, exercise, water, meals):
    x = np.array([1, vegetables, fast_food, exercise, water, meals])
    probability = sigmoid(np.dot(x, theta))
    result = "Healthy" if probability >= 0.5 else "Unhealthy"
    return result, probability

# CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef6ff 0%, #ffffff 45%, #f4fbf8 100%);
}

.hero {
    background: linear-gradient(135deg, #0b3d91, #1f8ef1);
    padding: 45px;
    border-radius: 25px;
    color: white;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

.hero h1 {
    font-size: 48px;
    margin-bottom: 10px;
}

.hero p {
    font-size: 20px;
}

.card {
    background: white;
    padding: 30px;
    border-radius: 22px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    border: 1px solid #e5eef8;
}

.result-card {
    background: white;
    padding: 35px;
    border-radius: 22px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    border-left: 8px solid #1f8ef1;
}

.footer {
    text-align: center;
    color: #666;
    margin-top: 35px;
}
</style>
""", unsafe_allow_html=True)

# Hero section
st.markdown("""
<div class="hero">
    <h1>🩺 Health Prediction App</h1>
    <p>Predict whether a person is healthy using lifestyle habits and logistic regression</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1.1, 0.9], gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📝 Enter Lifestyle Details")

    vegetables = st.slider("🥦 Vegetables eaten per week", 0, 7, 3)
    fast_food = st.slider("🍔 Fast food eaten per week", 0, 7, 2)
    exercise = st.slider("🏃 Exercise days per week", 0, 7, 3)
    water = st.slider("💧 Water intake per day (liters)", 0.0, 5.0, 2.0, 0.5)
    meals = st.slider("🍽️ Meals per day", 1, 6, 3)

    predict_button = st.button("🔍 Predict Health Status", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.header("📊 Prediction Result")

    if predict_button:
        result, probability = predict_health(
            vegetables, fast_food, exercise, water, meals
        )

        if result == "Healthy":
            st.success("✅ Prediction: Healthy")
        else:
            st.error("⚠️ Prediction: Unhealthy")

        st.metric("Probability of Being Healthy", f"{probability * 100:.2f}%")
        st.progress(float(probability))

        st.subheader("Your Input Summary")
        st.write(f"🥦 Vegetables: **{vegetables} times/week**")
        st.write(f"🍔 Fast Food: **{fast_food} times/week**")
        st.write(f"🏃 Exercise: **{exercise} days/week**")
        st.write(f"💧 Water: **{water} liters/day**")
        st.write(f"🍽️ Meals: **{meals} meals/day**")

    else:
        st.info("Fill in the lifestyle details and click the prediction button.")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    <p>Machine Learning Mini Project | Logistic Regression Model</p>
</div>
""", unsafe_allow_html=True)