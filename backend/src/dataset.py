import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import tqdm

import torch.utils.data as data

from utils import MIDITokenExtractor
from config import voc_single_track
from config import FRAME_PER_SEC, FRAME_STEP_SIZE_SEC, AUDIO_SEGMENT_SEC, SEGMENT_N_FRAMES  

"""# Dataset
Uses MAESTRO v3.0.0 dataset.
"""

class AMTDatasetBase(data.Dataset):
    def __init__(
        self,
        flist_audio, # オーディオファイルのパスをリスト形式で渡す
        flist_midi, # MIDIファイルのパスをリスト形式で渡す
        sample_rate, # オーディオファイルのサンプリングレートを指定。全てのオーディオがこれにリサンプリングされる。
        voc_dict, # トークン定義を渡す
        apply_pedal=True,
        whole_song=False,
    ):
        super().__init__()
        self.midi_filelist = flist_midi
        self.audio_filelist = flist_audio
        self.audio_metalist = [torchaudio.info(f) for f in flist_audio] # 各オーディオファイルのメタ情報（サンプルレート、フレーム数など）を収集します。
        self.voc_dict = voc_dict
        # 各MIDIファイルを MIDITokenExtractor を使ってトークン化し、その結果をリストとして保持します。
        self.midi_list = [
            MIDITokenExtractor(f, voc_dict, apply_pedal)
            for f in tqdm.tqdm(self.midi_filelist, desc="load dataset")
        ]
        self.sample_rate = sample_rate
        self.whole_song = whole_song

    def __len__(self):
        return len(self.audio_filelist)

    def __getitem__(self, index):
        """
        Return a pair of (audio, tokens) for the given index.
        On the training stage, return a random segment from the song.
        On the test stage, return the audio and MIDI of the whole song.
        """
        if not self.whole_song:
            return self.getitem_segment(index)
        else:
            return self.getitem_wholesong(index)

    def getitem_segment(self, index, start_pos=None): # 対象ファイルを指定するindexとセグメントの開始位置（フレーム単位）。Noneの場合はランダムに選択
        metadata = self.audio_metalist[index]
        num_frames = metadata.num_frames # オーディオの全体の「サンプル数」。
        sample_rate = metadata.sample_rate
        duration_y = round(num_frames / float(sample_rate) * FRAME_PER_SEC) # オーディオ全体の長さをフレーム単位に変換
        midi_item = self.midi_list[index]

        # セグメントの開始位置と終了位置（フレーム単位）を決定。
        if start_pos is None: # np.random.randint を使用して、オーディオ全体からランダムに開始位置を選択。
            segment_start = np.random.randint(duration_y - SEGMENT_N_FRAMES)
        else: # start_pos が指定されている場合
            segment_start = start_pos
        segment_end = segment_start + SEGMENT_N_FRAMES
        # オーディオセグメントのサンプル単位の開始位置
        segment_start_sample = round(
            segment_start * FRAME_STEP_SIZE_SEC * sample_rate
        )

        # セグメント範囲（segment_start ～ segment_end）に対応するMIDIトークン列を抽出。
        segment_tokens = midi_item.get_segment_tokens(segment_start, segment_end)
        segment_tokens = torch.from_numpy(segment_tokens).long() # NumPy配列をPyTorchテンソルに変換。long()でテンソルのデータ型を64ビット整数（long）に設定。

        # 指定されたセグメント範囲のオーディオデータを読み込む。
        # frame_offset から始まる範囲を num_frames サンプル分読み込む。
        y_segment, _ = torchaudio.load(
            self.audio_filelist[index],
            frame_offset=segment_start_sample,
            num_frames=round(AUDIO_SEGMENT_SEC * sample_rate),
        )
        y_segment = y_segment.mean(0) # オーディオが複数チャンネルの場合（例: ステレオ）、チャンネルを平均してモノラルに変換。

        # サンプルレートのリサンプリング
        # オーディオデータのサンプルレートが self.sample_rate と異なる場合、指定されたサンプルレートにリサンプリング。
        if sample_rate != self.sample_rate:
            y_segment = torchaudio.functional.resample(
                y_segment,
                sample_rate,
                self.sample_rate,
                resampling_method="kaiser_window",  # Kaiserウィンドウによるリサンプリングアルゴリズムを適用。
            )
        return y_segment, segment_tokens

    def getitem_wholesong(self, index):
        """
        Return a pair of (audio, midi) for the given index.
        """
        y, sr = torchaudio.load(self.audio_filelist[index]) # 読み込まれた波形データ（テンソル形式）。形状は (チャンネル数, サンプル数)。
        y = y.mean(0) # モノラル化
        # サンプルレートのリサンプリング
        if sr != self.sample_rate:
            y = torchaudio.functional.resample(
                y, sr, self.sample_rate,
                resampling_method="kaiser_window"
            )
        midi = self.midi_list[index].pm
        return y, midi

    # collateはバッチにまとめる役割の関数
    def collate_wholesong(self, batch): # batch: データセットから取り出された複数のデータ（オーディオとMIDIのペア）のリスト。
        # b[0]で各データペアの0番目の要素、つまりオーディオデータを取り出す。
        # torch.stack([...], dim=0): 複数のテンソルを新しい次元（バッチ次元）で結合。
        # 出力: テンソルの形状は (バッチサイズ, サンプル数)。
        batch_audio = torch.stack([b[0] for b in batch], dim=0)
        midi = [b[1] for b in batch] # バッチ内の各曲のMIDIデータをリストとしてまとめる。
        return batch_audio, midi # テンソル, リスト

    def collate_batch(self, batch): # データセットから取り出されたセグメント化されたオーディオテンソルとセグメント化されたMIDIトークン列のリスト。
        # b[0]で各データペアの0番目の要素、つまりオーディオデータを取り出す。
        # torch.stack([...], dim=0): 複数のテンソルを新しい次元（バッチ次元）で結合。
        # 出力: テンソルの形状は (バッチサイズ, サンプル数)。
        batch_audio = torch.stack([b[0] for b in batch], dim=0)
        batch_tokens = [b[1] for b in batch] # バッチ内の各セグメントのトークン列をテンソル？リスト形式で取得。

        # バッチ内のMIDIトークン列の長さを揃えるためにパディング
        # torch.nn.utils.rnn.pad_sequence は、異なる長さのシーケンス（テンソルリスト）をパディングして同じ長さに揃えるためのPyTorchユーティリティ（すべてのテンソルは同じ次元数である必要があります（長さ以外は一致）。）
        # batch_first = True: パディング後のテンソル形状を (バッチサイズ, 最大長さ) に設定
        batch_tokens_pad = torch.nn.utils.rnn.pad_sequence(
            batch_tokens, batch_first=True, padding_value=self.voc_dict["pad"]
        )
        return batch_audio, batch_tokens_pad # テンソル, テンソル　(バッチサイズ, サンプル数), (バッチサイズ, 最大トークンの長さ)


class CustomDataset(AMTDatasetBase):
    def __init__(
        self,
        midi_root: str = "/content/drive/MyDrive/B4/Humtrans/midi",
        wav_root: str = "/content/wav_rms",
        split: str = "train",
        sample_rate: int = 16000,
        apply_pedal: bool = True,
        whole_song: bool = False,
    ):
        """
        MIDIとWAVのペアをロードするデータセットクラス

        Args:
            midi_root (str): MIDIファイルが保存されているルートフォルダ
            wav_root (str): WAVファイルが保存されているフォルダ
            split (str): 使用するデータセットの分割 ('train', 'valid', 'test')
            sample_rate (int): サンプルレート
            apply_pedal (bool): ペダルの適用
            whole_song (bool): 曲全体をロードするか
        """
        # MIDIフォルダのパスを設定
        self.midi_root = f"/content/filtered_{split}_midi"
        self.wav_root = wav_root
        self.sample_rate = sample_rate
        self.split = split

        # MIDIとWAVのペアを見つける
        flist_midi, flist_audio = self._get_paired_files()

        # 親クラスのコンストラクタを呼び出し
        super().__init__(
            flist_audio,
            flist_midi,
            sample_rate,
            voc_dict=voc_single_track,
            apply_pedal=apply_pedal,
            whole_song=whole_song,
        )


    def _get_paired_files(self):
        """
        MIDIフォルダとWAVフォルダからペアとなるファイルリストを作成する

        Returns:
            flist_midi (list): 対応するMIDIファイルのリスト
            flist_audio (list): 対応するWAVファイルのリスト
        """
        flist_midi = []
        flist_audio = []

        # MIDIフォルダからMIDIファイルを取得
        midi_files = [f for f in os.listdir(self.midi_root) if f.endswith(".mid")]

        for midi_file in midi_files:
            # MIDIファイルのパスを構築
            midi_path = os.path.join(self.midi_root, midi_file)

            # WAVファイルのパスを構築 (拡張子を変更)
            wav_file = os.path.splitext(midi_file)[0] + ".wav"
            wav_path = os.path.join(self.wav_root, wav_file)

            # WAVファイルが存在するか確認
            if os.path.exists(wav_path):
                flist_midi.append(midi_path)
                flist_audio.append(wav_path)
            else:
                print(f"対応するWAVファイルが見つかりません: {midi_file}")

        print(f"{self.split}データセット: {len(flist_midi)} ペアのMIDI-WAVが見つかりました。")
        return flist_midi, flist_audio
