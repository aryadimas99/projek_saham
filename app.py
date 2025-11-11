# app.py
from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, GRU, BatchNormalization

from stocks_web import preprocess_df, construct_time_frames

app = Flask(__name__)

MODEL_PATH = "bgru_lstm_model.h5"
model = load_model(MODEL_PATH)

# Folder untuk simpan gambar plot
PLOT_PATH = os.path.join("static", "plot.png")
os.makedirs("static", exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None
    plot_url = None
    message = None

    if request.method == "POST":
        file = request.files.get("file")

        if not file:
            message = "Silakan pilih file CSV terlebih dahulu."
        else:
            try:
                df = pd.read_csv(file)

                data_scaled = preprocess_df(df)
                x, y = construct_time_frames(data_scaled, frame_size=64)

                if x.shape[0] == 0:
                    message = "Data terlalu sedikit untuk frame_size=64."
                else:
                    y_pred = model.predict(x)

                    plt.figure()
                    plt.plot(y, label="Actual")
                    plt.plot(y_pred, label="Predicted")
                    plt.title("Prediksi Harga Saham (BiGRU + LSTM)")
                    plt.xlabel("Time")
                    plt.ylabel("Harga (scaled)")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(PLOT_PATH)
                    plt.close()

                    plot_url = "/" + PLOT_PATH.replace("\\", "/")

                    predictions = y_pred[:30].flatten().tolist()

            except Exception as e:
                message = f"Terjadi error saat memproses file: {e}"

    return render_template(
        "index.html",
        predictions=predictions,
        plot_url=plot_url,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)
