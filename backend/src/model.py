import torch
import torch.nn as nn
from utils import split_audio_into_segments
from utils import LogMelspec
from utils import LogCQT
from transformers import T5Config, T5ForConditionalGeneration


class Seq2SeqTranscriber(nn.Module):
    def __init__(
        self, n_mels: int, sample_rate: int, n_fft: int, hop_length: int, voc_dict: dict
    ):
        super().__init__()
        self.infer_max_len = 200 # 推論時の最大シーケンス長。
        self.voc_dict = voc_dict # トークン辞書と
        self.n_voc_token = voc_dict["n_voc"] # トークンの数を保持。
        self.t5config = T5Config.from_pretrained("google/t5-v1_1-small") # Googleの事前学習済み T5 モデル（小型バージョン）の設定をロード。
        # カスタム設定を T5 の設定に追加:
        custom_configs = {
            "vocab_size": self.n_voc_token, # トークン辞書のサイズ。
            "pad_token_id": voc_dict["pad"], # パディングトークンID。
            "d_model": 96, # モデルの隠れ次元数（ここではメルバンド数に設定）。
        }

        for k, v in custom_configs.items():
            self.t5config.__setattr__(k, v)

        self.transformer = T5ForConditionalGeneration(self.t5config) # カスタム設定を適用した T5 モデルをロード。
        # self.melspec = LogMelspec(sample_rate, n_fft, n_mels, hop_length) # LogMelspec クラスを使用して、音声波形を対数メルスペクトログラムに変換するモジュールを作成。
        # CQT モデルインスタンス作成
        self.log_cqt = LogCQT(16000, 84, 128, 12)
        self.sr = sample_rate # サンプルレートをインスタンス変数として保存。

    # モデルの学習時に呼び出され、損失値を計算します。
    def forward(self, wav, labels):
        # spec = self.melspec(wav).transpose(-1, -2) # 音声波形（wav）をメルスペクトログラム（spec）に変換。LogMelspec クラスのforwardを実行
        spec = self.log_cqt(wav).transpose(-1, -2)
        # ※ .transpose(-1, -2): T5 モデルは通常 [バッチ, 時間ステップ, 次元] の形状を期待するため、周波数軸（メルバンド）と時間軸を入れ替えます。
        #  T5 モデルのフォワードパス
        print("sepc.shape: ", spec.shape)  # (1, n_bins, time)
        outs = self.transformer.forward(
            inputs_embeds=spec, return_dict=True, labels=labels
        )
        return outs # outs は辞書形式で損失値や出力トークン列を含む。

    # 入力音声波形（wav）から推定トークン列を生成する関数
    def infer(self, wav):
        """
        Infer the transcription of a single audio file.
        The input audio file is split into segments of 2 seconds
        before passing to the transformer.
        """
        wav_segs = split_audio_into_segments(wav, self.sr) # 音声波形を固定長（例: 2秒）に分割。
        #spec = self.melspec(wav_segs).transpose(-1, -2) # 各セグメントをメルスペクトログラムに変換。
        spec = self.log_cqt(wav_segs).transpose(-1, -2)
        # generate: T5 モデルの推論モードを使用して、トークン列を生成。
        outs = self.transformer.generate(
            inputs_embeds=spec,
            max_length=self.infer_max_len, # 推論時の最大出力長。
            num_beams=5, # ビームサーチを無効化し、単純なグリーディーサーチ。
            do_sample=False, # サンプリングを無効化。
            return_dict_in_generate=False,
        )
        return outs #推論結果として生成されたトークン列を返します。 #形状: (セグメント数, 最大トークン長)

