# main.py
from infer import infer_midi_from_wav

input_wav = "/Users/yagihayato/Downloads/my-melody-app/backend/data/humming.wav"
midi_path = infer_midi_from_wav(input_wav)
print("推論完了: ", midi_path)
