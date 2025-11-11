# stocks_web.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# scaler global untuk web
scaler = MinMaxScaler(feature_range=(0, 1))


def preprocess_df(df: pd.DataFrame):
    """
    df: dataframe hasil pd.read_csv dari file upload.
    Membersihkan format angka (misal 'Rp8,575') lalu scaling 0-1.
    """

    # Buang kolom yang tidak dipakai kalau ada
    df = df.drop(columns=['Adj Close', 'Date'], errors='ignore')

    # Buang kolom kosong semua (misalnya 'Unnamed: 7' dll)
    df = df.dropna(axis=1, how='all')

    # --- BERSIHKAN FORMAT ANGKA ---
    cleaned = pd.DataFrame()
    for col in df.columns:
        # ubah ke string lalu hilangkan semua karakter selain angka, minus, dan titik
        s = df[col].astype(str).str.replace(r'[^0-9\.\-]', '', regex=True)
        # konversi ke numeric; yang gagal -> NaN
        cleaned[col] = pd.to_numeric(s, errors='coerce')

    # buang kolom yang semua nilainya NaN
    cleaned = cleaned.dropna(axis=1, how='all')

    if cleaned.shape[1] == 0:
        raise ValueError(
            "Tidak ada kolom numerik setelah preprocessing. "
            "Periksa format CSV (misalnya angka masih berisi teks selain angka)."
        )

    data_scaled = scaler.fit_transform(cleaned.values)
    return data_scaled


def construct_time_frames(data, frame_size=64):
    """
    data: numpy array 2D (n_samples, n_features)
    Menghasilkan x, y seperti di Colab:
      - x: window sepanjang frame_size
      - y: nilai kolom pertama di langkah berikutnya
    """
    num_samples = data.shape[0]

    if num_samples <= frame_size:
        raise ValueError(
            f"Data terlalu sedikit untuk frame_size={frame_size}. "
            f"Minimal {frame_size + 1} baris, tetapi hanya {num_samples}."
        )

    x = [data[i-frame_size: i] for i in range(frame_size, num_samples)]
    y = [data[i, 0:1]          for i in range(frame_size, num_samples)]

    return np.array(x), np.array(y)