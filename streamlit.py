import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("final_model.h5")
labels = ["Rendah", "Sedang", "Tinggi"]

# Custom CSS for modern look and feel
st.markdown("""
    <style>
        /* General Layout */
        body {
            background-color: #121212;  /* Dark background for the app */
            color: #e0e0e0;  /* Light grey text */
            font-family: 'Arial', sans-serif;
        }

        .title {
            font-size: 40px;
            font-weight: bold;
            color: #FFDE00;  /* Bright yellow for the title */
            text-align: center;
            margin-top: 20px;
        }

        /* Prediction Result Styling */
        .prediction {
            font-size: 28px;
            color: #32CD32;  /* Green color for positive feedback */
            font-weight: bold;
            text-align: center;
            padding: 15px;
            border-radius: 15px;
            background-color: rgba(0, 255, 0, 0.2); /* Transparent background */
            margin-top: 20px;
        }

        /* Input field labels */
        .input-label {
            font-size: 18px;
            color: #1E90FF;  /* Bright blue for labels */
            font-weight: bold;
        }

        /* Info box styling */
        .info-box {
            color: #FFFFFF;  /* White text for readability */
            padding: 20px;
            border-radius: 15px;
            margin-top: 10px;
        }

        /* Button Styling */
        .stButton>button {
            background-color: #FF6347;  /* Red-orange button for interaction */
            color: white;
            font-size: 18px;
            padding: 15px 30px;
            border-radius: 12px;
            width: 100%;
            transition: all 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #FF4500;  /* Slightly darker on hover */
            transform: scale(1.1); /* Hover effect */
        }

        /* Sidebar Styling */
        .sidebar .sidebar-content {
            background-color: #1e1e1e;  /* Dark sidebar background */
        }

        /* Small Animations */
        .stButton>button:active {
            transform: scale(0.95);  /* Click effect */
        }
    </style>
""", unsafe_allow_html=True)

# Title of the application with custom style
st.markdown('<p class="title">🔮 Stress Level Detection 🔮</p>', unsafe_allow_html=True)

# Info box with more detailed and engaging description
st.markdown("""
    <div class="info-box">
        <h5>🔧 Masukkan nilai untuk kelembapan, temperatur, dan jumlah langkah Anda di bawah ini untuk mendapatkan prediksi tingkat stres Anda.</h5>
        <p>🧠 Model ini dapat memprediksi tingkat stres Anda berdasarkan kelembapan udara, temperatur lingkungan, dan jumlah langkah yang Anda tempuh.</p>
    </div>
""", unsafe_allow_html=True)

# Manual input fields for user to type the values
humidity = st.text_input("🌫️ Masukkan kelembapan (humidity):", value="50", help="Kelembapan udara dalam persen")
temperature = st.text_input("🌡️ Masukkan temperatur (°C):", value="22", help="Temperatur udara dalam derajat Celsius")
step_count = st.text_input("🚶‍♂️ Masukkan jumlah langkah (step count):", value="5000", help="Jumlah langkah yang telah ditempuh")

# Function to convert Celsius to Fahrenheit
def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

# Button to trigger prediction with engaging emoji
if st.button("🔮 Prediksi", key="predict"):
    try:
        # Convert inputs to numeric values
        humidity = float(humidity)
        temperature = float(temperature)
        step_count = int(step_count)

        # Prepare input data
        new_data = np.array([[humidity, celsius_to_fahrenheit(temperature), step_count]])

        # Prediction using the model
        prediction = model.predict(new_data)
        idx = np.argmax(prediction, axis=1)[0]  # Get the predicted class index

        # Displaying the prediction result with better formatting
        st.markdown(f'<p class="prediction">Prediksi: <strong>{labels[idx]}</strong></p>', unsafe_allow_html=True)

        # Display information for each label
        if idx == 0:
            st.markdown("""
                <div class="info-box">
                    <h5>💡 Stres Tingkat Rendah</h5>
                    <p>Anda berada dalam kondisi yang cukup rileks. Lanjutkan dengan aktivitas yang menyenangkan untuk menjaga suasana hati Anda!</p>
                </div>
            """, unsafe_allow_html=True)
        elif idx == 1:
            st.markdown("""
                <div class="info-box">
                    <h5>⚠️ Stres Tingkat Sedang</h5>
                    <p>Ada beberapa faktor yang mungkin membuat Anda sedikit stres. Pertimbangkan untuk mengambil waktu sejenak untuk beristirahat atau meditasi.</p>
                </div>
            """, unsafe_allow_html=True)
        elif idx == 2:
            st.markdown("""
                <div class="info-box">
                    <h5>🚨 Stres Tingkat Tinggi</h5>
                    <p>Tingkat stres Anda cukup tinggi. Pastikan untuk melakukan relaksasi atau berbicara dengan seseorang yang bisa membantu Anda meredakan stres.</p>
                </div>
            """, unsafe_allow_html=True)

    except ValueError:
        st.error("🚨 Pastikan semua input valid (angka)!")
    except Exception as e:
        st.error(f"🚨 Terjadi kesalahan: {e}")
