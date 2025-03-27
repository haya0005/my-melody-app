from flask import Flask, request, send_file
from flask_cors import CORS
import os
import traceback
from infer import infer_midi_from_wav

app = Flask(__name__)
CORS(app)  # ✅ Reactや他のドメインからのアクセスを許可

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return {"error": "ファイルがありません"}, 400

    file = request.files["file"]
    input_path = os.path.join(os.path.dirname(__file__), "input.wav")
    file.save(input_path)

    try:
        midi_path = infer_midi_from_wav(input_path)
        abs_path = os.path.abspath(midi_path)  # ✅ 絶対パスに変換
        return send_file(abs_path, as_attachment=True)
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}, 500

if __name__ == "__main__":
    # ✅ Render用に環境変数PORTから取得（ローカルでは5000）
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
