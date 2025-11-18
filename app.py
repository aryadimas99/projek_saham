import io
import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from tensorflow.keras.models import load_model

from stocks_web import preprocess_df, construct_time_frames

MODEL_PATH = "bgru_lstm_model.h5"
model = load_model(MODEL_PATH)

def predict_and_plot(csv_bytes, frame_size=64, max_preds=30):
    if csv_bytes is None:
        return "Silakan upload file CSV.", None

    try:
        # baca CSV dari bytes
        df = pd.read_csv(io.BytesIO(csv_bytes))

        data_scaled = preprocess_df(df)
        x, y = construct_time_frames(data_scaled, frame_size=frame_size)

        if x.shape[0] == 0:
            return f"Data terlalu sedikit untuk frame_size={frame_size}.", None

        y_pred = model.predict(x)

        # ambil sebagian hasil untuk ditampilkan
        preds_list = y_pred[:max_preds].flatten().tolist()

        # buat plot ke buffer
        fig, ax = plt.subplots()
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

        return preds_list, buf  # Gradio akan menampilkan gambar dari buffer

    except Exception as e:
        return f"Terjadi error saat memproses file: {e}", None

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Prediksi Harga Saham — BiGRU + LSTM")
    with gr.Row():
        csv_in = gr.File(label="Upload CSV (historical prices)", file_count="single", type="bytes")
        frame_input = gr.Number(value=64, label="frame_size", precision=0)
    predict_btn = gr.Button("Predict")
    output_text = gr.Textbox(label="Pesan / Prediksi singkat")
    preds_output = gr.Dataframe(headers=["prediction"], label="Predictions (first rows)")
    plot_output = gr.Image(type="pil", label="Plot")

    def _wrap(csv_bytes, frame_size):
        result, buf = predict_and_plot(csv_bytes, frame_size=int(frame_size))
        if isinstance(result, str):
            return result, None, None
        # result = list of floats
        df_preds = pd.DataFrame({"prediction": result})
        return "Selesai", df_preds, buf

    predict_btn.click(_wrap, inputs=[csv_in, frame_input], outputs=[output_text, preds_output, plot_output])

if __name__ == "__main__":
    demo.launch()
