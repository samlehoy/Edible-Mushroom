import streamlit as st
import pandas as pd
import pickle

# Load model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Load data asli untuk opsi input
url = "https://raw.githubusercontent.com/dataset-machine-learning/mushroom/refs/heads/main/mushroom.csv"
df_original = pd.read_csv(url, sep=';')
df_original.drop(columns=['gill-spacing', 'gill-size', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'spore-print-color'], inplace=True)

# Streamlit UI
st.title("🍄 Prediksi Jamur: Bisa Dimakan atau Beracun?")
st.markdown("Isi karakteristik jamur di bawah ini:")

user_input = {}
for col in df_original.columns:
    if col != "class":
        options = sorted(df_original[col].unique())
        user_input[col] = st.selectbox(f"{col}", options)

# Ubah input jadi DataFrame
input_df = pd.DataFrame([user_input])

# Encoding sama seperti pelatihan (LabelEncoder fitted ulang berdasarkan data original)
input_encoded = pd.DataFrame()
for col in input_df.columns:
    encoder = pd.factorize(df_original[col])[1]
    input_encoded[col] = input_df[col].apply(lambda x: list(encoder).index(x))

# Prediksi
if st.button("Prediksi"):
    prediction = model.predict(input_encoded)
    hasil = "☠️ Beracun!" if prediction[0] == 1 else "✅ Bisa Dimakan"
    st.success(f"Hasil: {hasil}")
