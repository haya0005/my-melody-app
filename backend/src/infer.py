# infer.py
import torch
import torchaudio
import os
import subprocess

from utils import unpack_sequence, token_seg_list_to_midi
from train import LitTranscriber
from utils import rms_normalize_wav  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# モデルの初期化（Flask起動時に一度だけ行う）
args = {
    "n_mels": 128,
    "sample_rate": 16000,
    "n_fft": 1024,
    "hop_length": 128,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# checkpoint のパスは、必要に応じて環境変数化や引数で制御してもよい
checkpoint_path = "/Users/yagihayato/Downloads/my-melody-app/backend/data/0210_cqt_hop128_epochepoch=390-vali_lossvali_loss=0.40.ckpt"
model = LitTranscriber(transcriber_args=args, lr=1e-4, lr_decay=0.99)
model = LitTranscriber.load_from_checkpoint(checkpoint_path=checkpoint_path).to(device)
model.eval()

def convert_to_pcm_wav(input_path, output_path):
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-acodec", "pcm_s16le",  # Linear PCM 16bit
        "-ar", "16000",          # サンプリング周波数をモデルと同じに
        output_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("ffmpeg conversion failed:")
        print(result.stderr.decode())
        raise RuntimeError("ffmpeg conversion failed")

def infer_midi_from_wav(input_wav_path: str) -> str:
    """
    WAVファイルからMIDIファイルを生成し、ファイルとして保存する
    Args:
        input_wav_path (str): 入力WAVファイルのパス
    Returns:
        str: 出力されたMIDIファイルのパス
    """
    # 0. WAVファイルをPCM形式に変換
    converted_path = os.path.join(BASE_DIR, "converted_input.wav")
    convert_to_pcm_wav(input_wav_path, converted_path)

    # 1. RMS正規化（正規化後ファイル名は固定でも一時ファイルでもOK）
    normalized_path = os.path.join(BASE_DIR, "tmp_normalized.wav")
    rms_normalize_wav(converted_path, normalized_path, target_rms=0.1)

    # 2. 音声読み込みと前処理
    waveform, sr = torchaudio.load(normalized_path)
    waveform = waveform.mean(0).to(device)

    if sr != model.transcriber.sr:
        waveform = torchaudio.functional.resample(
            waveform, sr, model.transcriber.sr
        ).to(device)

    # 3. 推論
    with torch.no_grad():
        output_tokens = model(waveform)

    # 4. トークンからMIDI生成
    unpadded_tokens = unpack_sequence(output_tokens.cpu().numpy())
    unpadded_tokens = [t[1:] for t in unpadded_tokens]  # 開始トークン除去
    est_midi = token_seg_list_to_midi(unpadded_tokens)

    # 5. 保存して返す
    midi_path = os.path.join(BASE_DIR, "output.mid")
    est_midi.write(midi_path)
    print(f"MIDI saved at: {midi_path}")
    return midi_path
