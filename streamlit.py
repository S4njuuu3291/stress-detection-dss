import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("final_model.h5")
labels = ["Rendah", "Sedang", "Tinggi"]

# Custom CSS for modern look and feel
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: #e0e0e0;
            font-family: 'Arial', sans-serif;
        }

        .title {
            font-size: 40px;
            font-weight: bold;
            color: #FFDE00;
            text-align: center;
            margin-top: 20px;
        }

        .prediction {
            font-size: 28px;
            color: #32CD32;
            font-weight: bold;
            text-align: center;
            padding: 15px;
            border-radius: 15px;
            background-color: rgba(0, 255, 0, 0.2);
            margin-top: 20px;
        }

        .input-label {
            font-size: 18px;
            color: #1E90FF;
            font-weight: bold;
        }

        .info-box {
            color: #FFFFFF;
            padding: 5px;
            border-radius: 15px;
            margin-top: 10px;
            margin-bottom: 10px;
            background-color: rgba(255, 255, 255, 0.1);
        }

        .stButton>button {
            background-color: #FF6347;
            color: white;
            font-size: 18px;
            padding: 15px 30px;
            border-radius: 12px;
            width: 100%;
            transition: all 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #FF4500;
            transform: scale(1.1);
        }

        .stButton>button:active {
            transform: scale(0.95);
        }
    </style>
""", unsafe_allow_html=True)

# Title of the application
st.markdown('<p class="title">🔮 Stress Level Detection 🔮</p>', unsafe_allow_html=True)

# Info box with a detailed description
st.markdown("""
    <div class="info-box">
        <h4>📝 Tentang Prediksi</h4>
        <p><strong>Kelembaban</strong> - Saat Anda merasa stres, suhu tubuh Anda meningkat, yang memicu kelenjar keringat untuk bekerja. Keringat ini dianggap sebagai Tingkat Kelembaban.</p>
        <p><strong>Suhu</strong> - Suhu tubuh seseorang saat stres.</p>
        <p><strong>Jumlah Langkah</strong> - Jumlah langkah yang ditempuh seseorang saat menghadapi situasi stres.</p>
        <p><strong>Tingkat Stres</strong> - Berdasarkan ketiga faktor di atas, Tingkat Stres Anda akan diprediksi menjadi Tinggi, Sedang, atau Rendah.</p>
    </div>
""", unsafe_allow_html=True)

# Input fields for user
humidity = st.text_input("🌫️ Masukkan kelembapan (humidity):", value="50")
temperature = st.text_input("🌡️ Masukkan temperatur (°C):", value="22")
step_count = st.text_input("🚶‍♂️ Masukkan jumlah langkah (step count):", value="50")

# Convert Celsius to Fahrenheit
def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

# Prediction button
if st.button("🔮 Prediksi", key="predict"):
    try:
        # Convert inputs to numeric values
        humidity = float(humidity)
        temperature = float(temperature)
        step_count = int(step_count)

        # Prepare input data
        new_data = np.array([[humidity, celsius_to_fahrenheit(temperature), step_count]])

        # Predict stress level
        prediction = model.predict(new_data)
        idx = np.argmax(prediction, axis=1)[0]

        # Display prediction result
        st.markdown(f'<p class="prediction">Prediksi: <strong>{labels[idx]}</strong></p>', unsafe_allow_html=True)

        # Provide insights for each stress level
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
