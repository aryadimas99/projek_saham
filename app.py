# app.py 
import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

from stocks_web import clean_numeric, construct_time_frames

MODEL_PATH = "bgru_lstm_model.h5"

# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Pastikan ter-upload ke repo.")
model = load_model(MODEL_PATH)

# Inspect model expected input shape
model_input_shape = model.input_shape 
print("Model input_shape:", model_input_shape)
n_features_expected = int(model_input_shape[-1])
print("Model expects n_features:", n_features_expected)

COMMON_FEATURE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']


def predict_and_plot(csv_bytes, frame_size=64, max_preds=10):
    """
    csv_bytes: binary file dari gr.File(type="binary")
    Mengembalikan: (preds_df, pil_img, df_head)
    Behaviour: FIT scaler on the uploaded CSV (same as VS Code)
    """
    if csv_bytes is None:
        raise ValueError("Silakan upload file CSV.")

    df = pd.read_csv(io.BytesIO(csv_bytes))

    cleaned = clean_numeric(df)

    use_cols = None
    if all(c in cleaned.columns for c in COMMON_FEATURE_COLS) and len(COMMON_FEATURE_COLS) == n_features_expected:
        use_cols = COMMON_FEATURE_COLS
    else:
        if cleaned.shape[1] == n_features_expected:
            use_cols = sorted(cleaned.columns)
        else:
            raise ValueError(
                f"Jumlah fitur tidak cocok. Model mengharapkan {n_features_expected} fitur per timestep, "
                f"tetapi CSV hanya memiliki {cleaned.shape[1]} kolom numerik. "
                f"Pastikan CSV memiliki kolom fitur yang sama seperti saat model dilatih."
            )

    # --- FIT SCALER DIRECTLY ON UPLOADED CSV (mimic local behavior) ---
    final_cleaned = cleaned[use_cols].copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(final_cleaned.values)
    # ---------------------------------------------------------------

    x, y = construct_time_frames(data_scaled, frame_size=int(frame_size))

    if x.shape[-1] != n_features_expected:
        raise ValueError(
            f"Setelah preprocessing, jumlah fitur per timestep adalah {x.shape[-1]}, "
            f"tetapi model mengharapkan {n_features_expected}."
        )


    y_pred = model.predict(x)

    preds_list = y_pred[:max_preds].flatten().tolist()
    preds_df = pd.DataFrame({"prediction": preds_list})

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(y.flatten(), label="Actual")
    ax.plot(y_pred.flatten(), label="Predicted")
    ax.set_title("Prediksi Harga Saham (BiGRU + LSTM)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Harga (scaled)")
    ax.legend()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    pil_img = Image.open(buf).convert("RGB")

    df_head = final_cleaned.head(10).reset_index(drop=True)

    return preds_df, pil_img, df_head


# ---- Gradio UI ----
with gr.Blocks() as demo:
    gr.Markdown("# Prediksi Harga Saham — BiGRU + LSTM")

    with gr.Row():
        csv_in = gr.File(label="Upload CSV (historical prices)", file_count="single", type="binary")
        frame_input = gr.Number(value=64, label="frame_size", precision=0)

    predict_btn = gr.Button("Predict")
    status_label = gr.Label(label="Status")
    preds_output = gr.Dataframe(headers=["prediction"], label="Predictions (first rows)")
    plot_output = gr.Image(type="pil", label="Plot")
    df_head_output = gr.Dataframe(label="CSV cleaned (first 10 rows)")

    def _wrap(csv_bytes, frame_size):
        try:
            preds_df, pil_img, df_head = predict_and_plot(csv_bytes, frame_size=int(frame_size), max_preds=10)
            return "Selesai", preds_df, pil_img, df_head
        except Exception as e:
            return f"Error: {e}", None, None, None

    predict_btn.click(_wrap, inputs=[csv_in, frame_input], outputs=[status_label, preds_output, plot_output, df_head_output])

if __name__ == "__main__":
    demo.launch()
