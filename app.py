import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load model
model = joblib.load("model.pkl")

# Load dataset asli untuk referensi encoding
url = "https://raw.githubusercontent.com/dataset-machine-learning/mushroom/refs/heads/main/mushroom.csv"
original_df = pd.read_csv(url, sep=';')

# Kolom fitur yang digunakan (sesuai dengan model terakhir)
columns = [
    'cap-shape', 'cap-color', 'cap-surface', 'bruises',
    'stalk-shape', 'stalk-root',
    'veil-color', 'population', 'habitat'
]

# Label Indonesia
label_indonesia = {
    'cap-shape': 'Bentuk Tudung',
    'cap-color': 'Warna Tudung',
    'cap-surface': 'Permukaan Tudung',
    'bruises': 'Memar',
    'stalk-shape': 'Bentuk Batang',
    'stalk-root': 'Akar Batang',
    'veil-color': 'Warna Selubung',
    'population': 'Populasi',
    'habitat': 'Habitat',
}

# Mapping nilai fitur ke Bahasa Indonesia
opsi_mapping = {
    'cap-shape': {'bell': 'Lonceng', 'conical': 'Kerucut', 'convex': 'Cembung', 'flat': 'Datar', 'knobbed': 'Tonjolan', 'sunken': 'Cekung'},
    'cap-surface': {'fibrous': 'Berserat', 'groovesmooth': 'Beralur', 'scaly': 'Bersisik', 'smooth': 'Halus'},
    'cap-color': {'brown': 'Coklat', 'buff': 'Coklat Kekuningan', 'cinnamon': 'Kayu Manis', 'gray': 'Abu-abu', 'green': 'Hijau', 'pink': 'Merah Muda', 'purple': 'Ungu', 'red': 'Merah', 'white': 'Putih', 'yellow': 'Kuning'},
    'bruises': {'bruises': 'Ya', 'no': 'Tidak'},
    'stalk-shape': {'enlarging': 'Membesar', 'tapering': 'Mengecil'},
    'stalk-root': {'?': '?', 'bulbous': 'Menggelembung', 'club': 'Tongkat', 'equal': 'Sama Besar', 'rooted': 'Berakar Kuat'},
    'veil-color': {'brown': 'Coklat', 'orange': 'Oranye', 'white': 'Putih', 'yellow': 'Kuning'},
    'population': {'abundant': 'Sangat Banyak', 'clustered': 'Bergerombol', 'numerous': 'Banyak', 'scattered': 'Tersebar', 'several': 'Beberapa', 'solitary': 'Sendirian'},
    'habitat': {'grasses': 'Rerumputan', 'leaves': 'Dedaunan', 'meadows': 'Padang Rumput', 'paths': 'Jalan Setapak', 'urban': 'Perkotaan', 'waste': 'Tempat Sampah', 'woods': 'Hutan'},
}

# Judul aplikasi
st.set_page_config(page_title="Prediksi Jamur", page_icon="🍄")
st.title("🍄 Prediksi Jamur: Bisa Dimakan atau Beracun?")

# Ambil input dari pengguna
user_input = {}
for col in columns:
    options = sorted(original_df[col].unique())
    label = label_indonesia.get(col, col)
    if col in opsi_mapping:
        opsi_indo = [opsi_mapping[col][opt] for opt in options]
        mapping_balik = {opsi_mapping[col][opt]: opt for opt in options}
        selected = st.selectbox(label, opsi_indo)
        user_input[col] = mapping_balik[selected]
    else:
        user_input[col] = st.selectbox(label, options)

# Prediksi saat tombol ditekan
if st.button("🔍 Prediksi Sekarang"):
    input_df = pd.DataFrame([user_input])
    input_encoded = pd.DataFrame()

    try:
        for col in columns:
            le = LabelEncoder().fit(original_df[col])
            input_encoded[col] = le.transform(input_df[col])
    except Exception as e:
        st.error(f"⚠️ Error saat encoding input: {e}")
        st.stop()

    # Prediksi
    # Susun ulang urutan kolom agar sesuai dengan saat training
    input_encoded = input_encoded[model.feature_names_in_]

    # Prediksi
    prediction = model.predict(input_encoded)[0]
    proba = model.predict_proba(input_encoded)[0]

    edible_prob = proba[0] * 100  # diasumsikan 0 = edible
    poisonous_prob = proba[1] * 100

    if prediction == 0:
        st.success(f"✅ Jamur ini DIPERKIRAKAN **AMAN DIMAKAN**")
    else:
        st.error(f"☠️ Jamur ini DIPERKIRAKAN **BERACUN**")

    st.markdown(f"### 📊 Probabilitas:")
    st.markdown(f"- ✅ Bisa dimakan: **{edible_prob:.2f}%**")
    st.markdown(f"- ☠️ Beracun: **{poisonous_prob:.2f}%**")
