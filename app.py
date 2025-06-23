import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load model dan data
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

url = "https://raw.githubusercontent.com/dataset-machine-learning/mushroom/refs/heads/main/mushroom.csv"
df = pd.read_csv(url, sep=';')

# Kolom fitur yang digunakan
features = [
    'cap-shape', 'cap-surface', 'cap-color', 'bruises',
    'gill-color', 'stalk-shape', 'stalk-root',
    'stalk-color-above-ring', 'stalk-color-below-ring',
    'veil-color', 'population', 'habitat'
]

# Label dalam Bahasa Indonesia
label_indonesia = {
    'cap-shape': 'Bentuk Tudung',
    'cap-surface': 'Permukaan Tudung',
    'cap-color': 'Warna Tudung',
    'bruises': 'Memar',
    'gill-color': 'Warna Bilah',
    'stalk-shape': 'Bentuk Batang',
    'stalk-root': 'Akar Batang',
    'stalk-color-above-ring': 'Warna Batang Atas Cincin',
    'stalk-color-below-ring': 'Warna Batang Bawah Cincin',
    'veil-color': 'Warna Selubung',
    'population': 'Populasi',
    'habitat': 'Habitat',
}

# Mapping nilai ke Bahasa Indonesia
opsi_mapping = {
    'cap-shape': {'bell': 'Lonceng', 'conical': 'Kerucut', 'convex': 'Cembung', 'flat': 'Datar', 'knobbed': 'Tonjolan', 'sunken': 'Cekung'},
    'cap-surface': {'fibrous': 'Berserat', 'groovesmooth': 'Beralur', 'scaly': 'Bersisik', 'smooth': 'Halus'},
    'cap-color': {'brown': 'Coklat', 'buff': 'Coklat Kekuningan', 'cinnamon': 'Kayu Manis', 'gray': 'Abu-abu', 'green': 'Hijau', 'pink': 'Merah Muda', 'purple': 'Ungu', 'red': 'Merah', 'white': 'Putih', 'yellow': 'Kuning'},
    'bruises': {'bruises': 'Ya', 'no': 'Tidak'},
    'gill-color': {'black': 'Hitam', 'brown': 'Coklat', 'buff': 'Coklat Kekuningan', 'chocolate': 'Coklat Tua', 'gray': 'Abu-abu', 'green': 'Hijau', 'orange': 'Oranye', 'pink': 'Merah Muda', 'purple': 'Ungu', 'red': 'Merah', 'white': 'Putih', 'yellow': 'Kuning'},
    'stalk-shape': {'enlarging': 'Membesar', 'tapering': 'Mengecil'},
    'stalk-root': {'?': '?', 'bulbous': 'Menggelembung', 'club': 'Tongkat', 'equal': 'Sama Besar', 'rooted': 'Berakar Kuat'},
    'stalk-color-above-ring': {'brown': 'Coklat', 'buff': 'Coklat Kekuningan', 'cinnamon': 'Kayu Manis', 'gray': 'Abu-abu', 'orange': 'Oranye', 'pink': 'Merah Muda', 'red': 'Merah', 'white': 'Putih', 'yellow': 'Kuning'},
    'stalk-color-below-ring': {'brown': 'Coklat', 'buff': 'Coklat Kekuningan', 'cinnamon': 'Kayu Manis', 'gray': 'Abu-abu', 'orange': 'Oranye', 'pink': 'Merah Muda', 'red': 'Merah', 'white': 'Putih', 'yellow': 'Kuning'},
    'veil-color': {'brown': 'Coklat', 'orange': 'Oranye', 'white': 'Putih', 'yellow': 'Kuning'},
    'population': {'abundant': 'Sangat Banyak', 'clustered': 'Bergerombol', 'numerous': 'Banyak', 'scattered': 'Tersebar', 'several': 'Beberapa', 'solitary': 'Sendirian'},
    'habitat': {'grasses': 'Rerumputan', 'leaves': 'Dedaunan', 'meadows': 'Padang Rumput', 'paths': 'Jalan Setapak', 'urban': 'Perkotaan', 'waste': 'Tempat Sampah', 'woods': 'Hutan'},
}

st.title("🍄 Prediksi Jamur: Bisa Dimakan atau Beracun?")

user_input = {}
for feature in features:
    opsi_asli = sorted(df[feature].unique())
    if feature in opsi_mapping:
        opsi_bhs = [opsi_mapping[feature].get(opt, opt) for opt in opsi_asli]
        opsi_label_map = {opsi_mapping[feature].get(opt, opt): opt for opt in opsi_asli}
        selected_bhs = st.selectbox(f"{label_indonesia.get(feature, feature)}", opsi_bhs)
        user_input[feature] = opsi_label_map[selected_bhs]
    else:
        selected = st.selectbox(f"{label_indonesia.get(feature, feature)}", opsi_asli)
        user_input[feature] = selected

if st.button("🔍 Prediksi Sekarang!"):
    input_df = pd.DataFrame([user_input])
    input_encoded = pd.DataFrame()

    for col in features:
        le = LabelEncoder().fit(df[col])
        try:
            input_encoded[col] = le.transform(input_df[col])
        except ValueError as e:
            st.error(f"Gagal encode fitur `{col}`: {e}")
            st.stop()

    prediction = model.predict(input_encoded)
    hasil = '✅ Bisa Dimakan (Edible)' if prediction[0] == 0 else '⚠️ Beracun (Poisonous)'

    st.subheader("Hasil Prediksi")
    st.markdown(f"## {hasil}")