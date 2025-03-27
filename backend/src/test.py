import humtrans_seq2seq
from humtrans_seq2seq import LitTranscriber
from humtrans_seq2seq import infer_single_wav, visualize_and_play_midi
from base64 import b64decode
#import ipywidgets as widgets
#from IPython.display import Javascript, display
#from IPython.display import Audio

import soundfile as sf
import numpy as np
import librosa
import librosa.display
import torch

def normalize_rms(audio, target_rms=0.1):
    rms = np.sqrt(np.mean(audio**2))  # 現在のRMSを計算
    if rms == 0:  # 無音の場合は正規化不要
        print("⚠️ 無音ファイルのため、正規化をスキップします。")
        return audio
    scaling_factor = target_rms / rms  # スケール係数を計算
    normalized_audio = audio * scaling_factor  # 音量調整
    return normalized_audio

def rms_normalize_wav(input_wav, output_wav, target_rms=0.1):
    audio, sr = librosa.load(input_wav, sr=None) # 音声ファイルを読み込み
    normalized_audio = normalize_rms(audio, target_rms) # RMS正規化を適用
    sf.write(output_wav, normalized_audio, sr) # 正規化した音声を保存
    #print(f"正規化されたWAVファイルを保存: {output_wav}")

"""# チェックポイントロードで推論"""

# モデルの引数
args = {
    "n_mels": 128,
    "sample_rate": 16000,
    "n_fft": 1024,
    "hop_length": 128,
}

lightning_module = LitTranscriber(
    transcriber_args=args,
    lr=1e-4, # 0.0001
    lr_decay=0.99, # 各エポックの度に0.99倍して減衰させる
)

# 元のWAVファイルパス
wav_path = "data/humming.wav"

# RMS正規化後のWAVファイルパス
normalized_wav_path = "data/humming_normalized.wav"

# WAVをRMS正規化
rms_normalize_wav(wav_path, normalized_wav_path, target_rms=0.1)

# チェックポイントのパス
checkpoint_path = "data/0210_cqt_hop128_epochepoch=390-vali_lossvali_loss=0.40.ckpt"

# 推論結果を保存するMIDIファイルパス
midi_save_path = "data/output_midi.mid"

# 推論を実行（正規化後のWAVを使用）
infer_single_wav(lightning_module, normalized_wav_path, midi_save_path, checkpoint_path=checkpoint_path, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))

audio = visualize_and_play_midi(midi_save_path)
