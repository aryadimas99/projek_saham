import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

SCALER_PATH = "scaler.save"


if os.path.exists(SCALER_PATH):
    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception:
        scaler = MinMaxScaler(feature_range=(0, 1))
else:
    scaler = MinMaxScaler(feature_range=(0, 1))


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:


    df = df.drop(columns=['Adj Close', 'Date'], errors='ignore')
    df = df.dropna(axis=1, how='all')

    cleaned = pd.DataFrame()
    for col in df.columns:
        s = df[col].astype(str).str.replace(r'[^0-9\.\-]', '', regex=True)
        cleaned[col] = pd.to_numeric(s, errors='coerce')

    cleaned = cleaned.dropna(axis=1, how='all')
    if cleaned.shape[1] == 0:
        raise ValueError("Tidak ada kolom numerik setelah cleaning. Periksa header CSV dan isi kolom.")
    return cleaned


def transform_and_scale(cleaned: pd.DataFrame, expected_cols: list | None = None):

    global scaler

    if expected_cols is not None:
        missing = [c for c in expected_cols if c not in cleaned.columns]
        if len(missing) == 0:
            final = cleaned[expected_cols].copy()
        else:
            raise ValueError(f"CSV tidak memiliki kolom yang diharapkan: {missing}. Kolom tersedia: {list(cleaned.columns)}")
    else:

        final = cleaned.reindex(sorted(cleaned.columns), axis=1).copy()


    if os.path.exists(SCALER_PATH):
        data_scaled = scaler.transform(final.values)
    else:
        data_scaled = scaler.fit_transform(final.values)
        try:
            joblib.dump(scaler, SCALER_PATH)
        except Exception:
           
            print("Warning: gagal menyimpan scaler ke disk. Lanjut tanpa menyimpan.")

    return data_scaled, final


def construct_time_frames(data: np.ndarray, frame_size: int = 64):
   
    num_samples = data.shape[0]
    if num_samples <= frame_size:
        raise ValueError(f"Data terlalu sedikit untuk frame_size={frame_size}. Minimal {frame_size+1}, tetapi hanya {num_samples}.")

    x = [data[i-frame_size: i] for i in range(frame_size, num_samples)]
    y = [data[i, 0:1] for i in range(frame_size, num_samples)]

    return np.array(x), np.array(y)