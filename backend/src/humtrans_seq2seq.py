
import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from nnAudio.features import CQT

import lightning.pytorch as pl
import torchaudio

import tqdm
import pretty_midi as pm
import mir_eval
import transformers
from transformers import T5Config, T5ForConditionalGeneration

import matplotlib.pyplot as plt
import mir_eval.sonify as sonify
#import mir_eval.display as display



SR = 16000
AUDIO_SEGMENT_SEC = 2.0
SEGMENT_N_FRAMES = 200
FRAME_STEP_SIZE_SEC = 0.01
FRAME_PER_SEC = 100


pl.seed_everything(1234)

"""# MIDI-like token"""

# MIDI-like token definitions

N_NOTE = 128
N_TIME = 205
N_SPECIAL = 3

voc_single_track = {
    "pad": 0,
    "eos": 1,
    "endtie": 2,
    "note": N_SPECIAL, # 3から始まるという意味
    "onset": N_SPECIAL+N_NOTE,
    "time": N_SPECIAL+N_NOTE+2,
    "n_voc": N_SPECIAL+N_NOTE+2+N_TIME+3,
    "keylist": ["pad", "eos", "endtie", "note", "onset", "time"]}



"""## Tokenizer
Given a MIDI file, the notes are first extended according to its pedal signal (`pm_apply_pedal()`). Then the notes are splitted into segments and transformed into token lists (`get_segment_tokens()`).
"""

from operator import attrgetter
import dataclasses

@dataclasses.dataclass
class Event:
    prog: int
    onset: bool
    pitch: int

class MIDITokenExtractor:
    """
    ・MIDIデータ（音符、タイミング、ペダル情報など）を抽出し、トークン列に変換する。
    ・セグメント単位でMIDIデータを分割し、各セグメントをトークンとして表現。
    """
    def __init__(self, midi_path, voc_dict, apply_pedal=True):
        """
        ・MIDIデータを読み込み、必要に応じてサステインペダルを適用する初期化処理を行う。
        """
        self.pm = pm.PrettyMIDI(midi_path)  # MIDIファイルをPrettyMIDIで読み込む
        if apply_pedal:
            self.pm_apply_pedal(self.pm) # サステインペダル処理を適用
        self.voc_dict = voc_dict # トークンの定義辞書
        self.multi_track = "instrument" in voc_dict # マルチトラック対応のフラグ

    def pm_apply_pedal(self, pm: pm.PrettyMIDI, program=0):
        """
        Apply sustain pedal by stretching the notes in the pm object.
        """
        # 1: Record the onset positions of each notes as a dictionary
        onset_dict = dict()
        for note in pm.instruments[program].notes:
            if note.pitch in onset_dict:
                onset_dict[note.pitch].append(note.start)
            else:
                onset_dict[note.pitch] = [note.start]
        for k in onset_dict.keys():
            onset_dict[k] = np.sort(onset_dict[k])

        # 2: Record the pedal on/off state of each time frame
        arr_pedal = np.zeros(
            round(pm.get_end_time()*FRAME_PER_SEC)+100, dtype=bool)
        pedal_on_time = -1
        list_pedaloff_time = []
        for cc in pm.instruments[program].control_changes:
            if cc.number == 64:
                if (cc.value > 0) and (pedal_on_time < 0):
                    pedal_on_time = round(cc.time*FRAME_PER_SEC)
                elif (cc.value == 0) and (pedal_on_time >= 0):
                    pedal_off_time = round(cc.time*FRAME_PER_SEC)
                    arr_pedal[pedal_on_time:pedal_off_time] = True
                    list_pedaloff_time.append(cc.time)
                    pedal_on_time = -1
        list_pedaloff_time = np.sort(list_pedaloff_time)

        # 3: Stretch the notes (modify note.end)
        for note in pm.instruments[program].notes:
            # 3-1: Determine whether sustain pedal is on at note.end. If not, do nothing.
            # 3-2: Find the next note onset time and next pedal off time after note.end.
            # 3-3: Extend note.end till the minimum of next_onset and next_pedaloff.
            note_off_frame = round(note.end*FRAME_PER_SEC)
            pitch = note.pitch
            if arr_pedal[note_off_frame]:
                next_onset = np.argwhere(onset_dict[pitch] > note.end)
                next_onset = np.inf if len(
                    next_onset) == 0 else onset_dict[pitch][next_onset[0, 0]]
                next_pedaloff = np.argwhere(list_pedaloff_time > note.end)
                next_pedaloff = np.inf if len(
                    next_pedaloff) == 0 else list_pedaloff_time[next_pedaloff[0, 0]]
                new_noteoff_time = max(note.end, min(next_onset, next_pedaloff))
                new_noteoff_time = min(new_noteoff_time, pm.get_end_time())
                note.end = new_noteoff_time

    def get_segment_tokens(self, start, end):
        """
        Transform a segment of the MIDI file into a sequence of tokens.
        """
        dict_event = dict() # a dictionary that maps time to a list of events.

        def append_to_dict_event(time, item):
            if time in dict_event:
                dict_event[time].append(item)
            else:
                dict_event[time] = [item]

        list_events = []        # events section
        list_tie_section = []   # tie section

        for instrument in self.pm.instruments:
            prog = instrument.program
            for note in instrument.notes:
                note_end = round(note.end * FRAME_PER_SEC) # 音符終了時刻（フレーム単位）
                note_start = round(note.start * FRAME_PER_SEC) # 音符開始時刻（フレーム単位）
                if (note_end < start) or (note_start >= end):
                    # セグメント外の音符は無視
                    continue
                if (note_start < start) and (note_end >= start):
                    # If the note starts before the segment, but ends in the segment
                    # it is added to the tie section.
                    # セグメント開始時より前に始まり、セグメント内で終了する音符（ピッチ）をタイセクションに追加
                    list_tie_section.append(self.voc_dict["note"] + note.pitch)
                    if note_end < end:
                        # セグメント内で終了する場合、イベントに終了時刻を記録
                        append_to_dict_event(
                            note_end - start, Event(prog, False, note.pitch)
                        )
                    continue
                assert note_start >= start
                # セグメント内で開始
                append_to_dict_event(note_start - start, Event(prog, True, note.pitch))
                if note_end < end:
                    # セグメント内で終了
                    append_to_dict_event(
                        note_end - start, Event(prog, False, note.pitch)
                    )

        cur_onset = None
        cur_prog = -1

        # トークン列の生成
        """
        具体例
        dict_event = {
          0: [Event(prog=0, onset=True, pitch=60)],
          5: [Event(prog=0, onset=False, pitch=60)]
        }
        """
        for time in sorted(dict_event.keys()): # 現在の相対時間（time）をトークン化し、イベント列に追加
            list_events.append(self.voc_dict["time"] + time) # self.voc_dict["time"]はtimeトークンの開始ID(133)。これに相対時間を足す
            for event in sorted(dict_event[time], key=attrgetter("pitch", "onset")): # 同一時間内でピッチを昇順に、onset→offsetの順になるようにソートする。
                if cur_onset != event.onset:
                    cur_onset = event.onset # オンセットオフセットが変わった場合に更新
                    list_events.append(self.voc_dict["onset"] + int(event.onset)) # オンセットオフセットトークンを追加
                list_events.append(self.voc_dict["note"] + event.pitch) # 音符トークンを追加

        # Concatenate tie section, endtie token, and event section
        list_tie_section.append(self.voc_dict["endtie"]) # ID:2を追加
        list_events.append(self.voc_dict["eos"]) # ID: 1を追加
        tokens = np.concatenate((list_tie_section, list_events)).astype(int)
        return tokens


"""## Detokenizer
Transforms a list of MIDI-like token sequences into a MIDI file.
"""

def parse_id(voc_dict: dict, id: int):
    """
    トークンIDを解析し、トークンの種類名とその相対IDを返す関数。

    引数:
        - voc_dict: トークン定義を含む辞書
            例: {
                "pad": 0,
                "eos": 1,
                "endtie": 2,
                "note": 3,
                "onset": 131,
                "time": 133,
                ...
                "keylist": ["pad", "eos", "endtie", "note", "onset", "time"]
            }
        - id: 解析対象のトークンID（整数）

    戻り値:
        - token_name: トークンの種類名（例: "note", "time", "onset" など）
        - token_id: トークン種類内での相対ID（整数値）

    処理の流れ:
        1. `keylist` を使ってトークンの種類を順番に確認。
        2. 入力IDが各トークン種類の開始位置より小さい場合、該当するトークン種類を決定。
        3. 決定した種類の開始位置を引いて相対IDを計算。
        4. 結果をタプル (token_name, token_id) として返す。
    """
    keys = voc_dict["keylist"]  # トークンの種類リストを取得
    token_name = keys[0]       # デフォルトの種類を "pad" に設定

    # トークンの種類を特定する
    for k in keys:
        if id < voc_dict[k]:   # 現在の種類の開始位置より小さい場合、前の種類が該当
            break
        token_name = k         # 現在の種類名を更新

    # 該当種類内での相対IDを計算
    token_id = id - voc_dict[token_name]

    return token_name, token_id  # 種類名と相対IDを返す



def to_second(n):
    """
    フレーム数を秒単位に変換する関数。

    引数:
        - n: フレーム数（整数または浮動小数点数）

    戻り値:
        - n * FRAME_STEP_SIZE_SEC: 秒単位の値

    処理の流れ:
        1. `FRAME_STEP_SIZE_SEC` は1フレームの長さ（秒）を表す定数。
        2. 引数のフレーム数 `n` を1フレームの秒数で乗算して、秒単位に変換。

    例:
        FRAME_STEP_SIZE_SEC = 0.0116  # 1フレーム = 0.0116秒
        frames = 100
        seconds = to_second(frames)
        print(seconds)  # 結果: 1.16秒
    """
    return n * FRAME_STEP_SIZE_SEC



def find_note(list, n):
    """
    タプルリストの最初の要素から指定された値を検索し、そのインデックスを返す関数。

    引数:
        - list: タプルのリスト。例: [(60, 1.0), (62, 0.5), (64, 0.25)]
        - n: 検索対象の値（例: 62）

    戻り値:
        - インデックス (整数): n が最初に現れるインデックス。
        - -1: n が見つからない場合。

    処理手順:
        1. リスト内包表記でタプルの最初の要素を抽出し、新しいリスト `li_elem` を作成。
        2. `li_elem.index(n)` で n のインデックスを取得。
        3. 見つからなかった場合、例外 `ValueError` が発生し、-1 を返す。

    使用例:
        list = [(60, 1.0), (62, 0.5), (64, 0.25)]
        n = 62
        find_note(list, n)  # 結果: 1 (62が2番目)
    """
    li_elem = [a for a, _ in list]  # 最初の要素だけを抽出したリスト
    try:
        idx = li_elem.index(n)  # n が存在する場合、そのインデックスを取得
    except ValueError:
        return -1  # 存在しない場合は -1 を返す
    return idx  # 見つかった場合、そのインデックスを返す



def token_seg_list_to_midi(token_seg_list: list):
    """
    トークン列リストをMIDIファイルに変換する関数。

    引数:
        - token_seg_list: トークン列のリスト（例: [[3, 131, 133], [4, 132, 134]])。
          各トークンは音符、タイミング、オンセットなどをエンコードしている。

    戻り値:
        - midi_data: PrettyMIDIオブジェクト（MIDIデータ）。
    """
    # MIDIデータと楽器の初期化
    midi_data = pm.PrettyMIDI()
    piano_program = pm.instrument_name_to_program("Acoustic Grand Piano")
    piano = pm.Instrument(program=piano_program)

    list_onset = []  # 開始時刻を記録するリスト ※次のセグメント処理を行うときにlist_onsetには、最終音とタイの可能性がある音が記録されている
    cur_time = 0     # 前回のセグメントの終了時間

    # トークンセグメントごとの処理
    for token_seg in token_seg_list:
        list_tie = []           # タイ結合された音符
        cur_relative_time = -1  # セグメント内の相対時間
        cur_onset = -1          # 現在のオンセット状態
        tie_end = False         # タイ結合が終了したかどうか

        for token in token_seg:
            # トークンを解析
            token_name, token_id = parse_id(voc_single_track, token)

            if token_name == "note":
                # 音符処理
                if not tie_end: # タイ結合の場合
                    list_tie.append(token_id)   # タイのnote番号(相対ID)を追加
                elif cur_onset == 1:
                    list_onset.append((token_id, cur_time + cur_relative_time))  # 開始時刻を記録
                elif cur_onset == 0:
                    # 終了処理
                    i = find_note(list_onset, token_id)
                    if i >= 0:
                        start = list_onset[i][1]
                        end = cur_time + cur_relative_time
                        if start < end:  # 開始時刻 < 終了時刻の場合のみ追加
                            new_note = pm.Note(100, token_id, start, end)
                            piano.notes.append(new_note)
                        list_onset.pop(i)

            elif token_name == "onset":
                # オンセット/オフセットの更新
                if tie_end:
                    if token_id == 1:
                        cur_onset = 1  # 開始
                    elif token_id == 0:
                        cur_onset = 0  # 終了

            elif token_name == "time":
                # 相対時間の更新
                if tie_end:
                    cur_relative_time = to_second(token_id)

            elif token_name == "endtie":
                # タイ結合終了処理
                tie_end = True
                for note, start in list_onset: # list_onsetには前回のセグメントで未処理の最終音が含まれる
                    if note not in list_tie: # list_onsetにあるにも関わらず、list_tieにない場合
                        if start < cur_time:
                            new_note = pm.Note(100, note, start, cur_time) # 前回のセグメントの終了時をendとする（end=cur_time）
                            piano.notes.append(new_note)
                        list_onset.remove((note, start))

        # 現在の時間を更新
        cur_time += AUDIO_SEGMENT_SEC

    # 楽器をMIDIデータに追加
    midi_data.instruments.append(piano)
    return midi_data

# 基本的に、Time → On/Off → Noteトークンの順に並べることで一つの音符イベント
# --------------------------------------------------------------
# elif token_name == "endtie": の動作例（タイの時の処理の仕方）
# --------------------------------------------------------------
# 前提条件:
# list_onset = [(60, 0.5), (62, 0.6)]  # 音符とその開始時刻
# cur_time = 1.0  # 現在のセグメント終了時刻
# トークン列: [62, 2, 133, 132, 62, 1]
#   - 62: "note" タイセクションに含まれる音符
#   - 2: "endtie" タイセクション終了
#   - 133: "time" 相対時間 0.2とする
#   - 132: "onset" オフセット（音符終了イベント）
#   - 62: "note" 終了する音符
#   - 1: "eos" セグメント終了

# 処理の流れ:
# 1. タイセクション処理:
#    - list_tie = [62]  # タイセクションに音符62を追加
#
# 2. endtie処理:
#    - 音符60 (ピッチ60) はタイセクションに含まれないため終了処理:　（タイではなく、前回セグメントの最終音の処理）
#        - ノート: pm.Note(100, 60, 0.5, 1.0)
#    - 音符62 (ピッチ62) は次の処理に継続。
#    - 更新後の状態:
#        - piano.notes = [pm.Note(100, 60, 0.5, 1.0)]
#        - list_onset = [(62, 0.6)]  # 音符62が継続
#
# 3. イベントセクション処理:
#    - time トークン (133):
#        - 相対時間: cur_relative_time = 0.2
#    - onset トークン (132):
#        - オフセット（音符終了イベント）を設定。
#    - note トークン (62):
#        - ピッチ62のノートを終了。
#        - 開始時刻: 0.6, 終了時刻: 1.2 (cur_time + cur_relative_time)
#        - ノート: pm.Note(100, 62, 0.6, 1.2)
#
# 4. セグメント終了 (1):
#    - 更新後の状態:
#        - piano.notes = [
#            pm.Note(100, 60, 0.5, 1.0),  # セグメント1で終了
#            pm.Note(100, 62, 0.6, 1.2)   # セグメント途中で終了
#          ]
#        - list_onset = []  # 全音符が終了
# --------------------------------------------------------------

"""# Dataset
Uses MAESTRO v3.0.0 dataset.
"""

from pathlib import Path

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


class Maestro(AMTDatasetBase):
    def __init__(
        self,
        n_files=-1,
        sample_rate=44100,
        split="test",
        apply_pedal=True,
        whole_song=False,
    ):
        data_path = "../../../Data2/maestro-v3.0.0/"
        df_metadata = pd.read_csv(os.path.join(data_path, "maestro-v3.0.0.csv"))
        flist_audio = []
        flist_midi = []
        list_title = []
        for row in range(len(df_metadata)):
            if df_metadata["split"][row] == split: # splitを満たす行のみを処理
                f_audio = os.path.join(data_path, df_metadata["audio_filename"][row]) # audioのフルパスを作成
                f_midi = os.path.join(data_path, df_metadata["midi_filename"][row]) # MIDIのフルパスを作成
                assert os.path.exists(f_audio) and os.path.exists(f_midi)
                flist_audio.append(f_audio)
                flist_midi.append(f_midi)
                list_title.append(df_metadata["canonical_title"][row]) # 曲名をlist_titleに追加
        if n_files > 0: # -1ではない時、使用するファイル数を制御
            flist_audio = flist_audio[:n_files]
            flist_midi = flist_midi[:n_files]
        # 作成したオーディオとMIDIのリスト（flist_audio, flist_midi）を AMTDatasetBase のコンストラクタに渡して初期化。
        super().__init__(
            flist_audio,
            flist_midi,
            sample_rate,
            voc_dict=voc_single_track,
            apply_pedal=apply_pedal,
            whole_song=whole_song,
        )
        self.list_title = list_title # 曲名のリスト（list_title）をインスタンス変数として保存


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

"""# Evaluation function"""

def extract_midi(midi: pm.PrettyMIDI, program=0): # MIDIデータを読み込んだ PrettyMIDI オブジェクト, MIDIチャンネル（楽器番号）を指定。
    intervals = [] # 音符ごとの開始時間と終了時間のペアを格納したNumPy配列
    pitches = [] # 音符ごとの音高（MIDIノート番号）のNumPy配列。
    pm_notes = midi.instruments[program].notes # programで指定された対象楽器に含まれる全てのノート情報を取得。
    """
    例;
    instruments = [
      Instrument 0 (Piano): [Note(start=0.5, end=1.0, pitch=60), ...],
      Instrument 1 (Violin): [Note(start=1.0, end=1.5, pitch=62), ...]
    ]
    """
    # ノートを順番に処理
    for note in pm_notes:
        intervals.append((note.start, note.end)) # 音符の開始・終了時間のペアを intervals に追加。
        pitches.append(note.pitch) # 音符の音高を pitches に追加。

    return np.array(intervals), np.array(pitches) # intervals: 2D配列（各行が1つの音符の開始・終了時間を表す。）, pitches: 1D配列（各要素が1つの音符の音高（ピッチ）を表す。）


def evaluate_midi(est_midi: pm.PrettyMIDI, ref_midi: pm.PrettyMIDI, program=0):
    est_intervals, est_pitches = extract_midi(est_midi, program)
    ref_intervals, ref_pitches = extract_midi(ref_midi, program)

    # mir_eval ライブラリの transcription モジュールを使って、音符の一致度を評価します。
    dict_eval = mir_eval.transcription.evaluate(
        ref_intervals, ref_pitches, est_intervals, est_pitches, onset_tolerance=0.05)

    return dict_eval # dict_eval: 評価結果の辞書。

#################0308ここまでやった#######################

"""# Sequence-to-sequence transcriber

The transformer model is built with the configuration of "T5v1.1-small" model released by Google.
"""

# 固定長のセグメントに分割する関数
def split_audio_into_segments(y: torch.Tensor, sr: int): # オーディオデータ（テンソル）, オーディオのサンプルレート
    audio_segment_samples = round(AUDIO_SEGMENT_SEC * sr) # 1セグメントの長さをサンプル数で計算
    pad_size = audio_segment_samples - (y.shape[-1] % audio_segment_samples) # セグメントサイズできっちり分割できるようにpadするサイズを計算
    """
    例: オーディオデータの長さが 100000 サンプルで、audio_segment_samples = 88200 の場合
    pad_size = 88200 - (100000 % 88200)
         = 88200 - 11800
         = 76400
    pad後;
    y = [0.1, 0.2, 0.3, ..., 0.9, 0.0, 0.0, ..., 0.0]  # 長さ: 176400 サンプル
    """
    y = F.pad(y, (0, pad_size)) # padを追加
    assert (y.shape[-1] % audio_segment_samples) == 0 # 割り切れない場合assertをする
    n_chunks = y.shape[-1] // audio_segment_samples # セグメント数を計算
    # 固定長のセグメントに分割
    y_segments = torch.chunk(y, chunks=n_chunks, dim=-1) # torch.chunk: テンソルを指定した数（n_chunks）に分割、dim=-1: サンプル次元（最後の次元）で分割
    return torch.stack(y_segments, dim=0) # 分割したセグメントを1つのテンソルにまとめる。dim=0: セグメント数をバッチ次元として結合。
    # 形状: (セグメント数, セグメント長)。

# 推論時に作られたpadされたseqから有効な部分を抽出する関数
def unpack_sequence(x: torch.Tensor, eos_id: int=1):
    seqs = [] # 各シーケンスを切り出して保存するリストを初期化。
    max_length = x.shape[-1] # シーケンスの最大長を取得。whileループの範囲をチェックするために使用。
    for seq in x: # テンソル x の各シーケンス（行）を処理
        # eosトークンを探す
        start_pos = 0
        pos = 0
        while (pos < max_length) and (seq[pos] != eos_id): # 現在地が最大長を超えない&現在地がeosではない場合
            pos += 1
        # ループ終了後：pos には、終了トークン（eos_id）の位置（またはシーケンスの末尾）が格納されます。
        end_pos = pos+1
        seqs.append(seq[start_pos:end_pos])  #開始位置（start_pos）から終了位置（end_pos）までの部分を切り出し、リスト seqs に追加。
    return seqs

class LogMelspec(nn.Module):
    def __init__(self, sample_rate, n_fft, n_mels, hop_length):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft, # FFT（高速フーリエ変換）に使用するポイント数
            hop_length=hop_length, # ストライド（フレームのシフト幅（サンプル単位））。通常、n_fft // 4 などの値を設定
            f_min=20.0, # メルフィルタバンクの最小周波数。20Hz 以上が推奨（人間の聴覚範囲）。
            n_mels=n_mels, # メルスペクトログラムの周波数軸方向の次元数（メルバンド数）。
            mel_scale="slaney", # メル尺度を計算する方法."slaney": より音響的に意味のあるスケール
            norm="slaney", # メルフィルタバンクの正規化方法. "slaney" を指定するとフィルタバンクがエネルギーで正規化
            power=1, # 出力スペクトログラムのエネルギースケール, 1: 振幅スペクトログラム。
        )
        self.eps = 1e-5 # 対数計算時のゼロ除算エラーを防ぐための閾値。メルスペクトログラムの値が 1e-5 未満の場合、この値に置き換える。

    def forward(self, x): # 入力: モノラル: (バッチサイズ, サンプル数) or  ステレオ: (バッチサイズ, チャンネル数, サンプル数)
        spec = self.melspec(x) # 入力波形 x からメルスペクトログラムを計算, 出力：メルスペクトログラム: (バッチサイズ, メルバンド数, フレーム数)
        safe_spec = torch.clamp(spec, min=self.eps) # メルスペクトログラムの最小値を self.eps に制限。値が非常に小さい場合でも、対数計算が可能。
        log_spec = torch.log(safe_spec) #メルスペクトログラムを対数スケールに変換。
        return log_spec # (バッチサイズ, メルバンド数, フレーム数) のテンソル。各値は対数スケールのメルスペクトログラム。


class LogCQT(nn.Module):
    def __init__(self, sample_rate, n_bins, hop_length, bins_per_octave):
        super().__init__()
        self.cqt = CQT(
            sr=sample_rate,
            hop_length=hop_length,
            fmin=32.7,  # 低周波数の最小値 (通常は32.7Hz, C1)
            fmax=8000,  # 最高周波数
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            verbose=False
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # GPUに載せる
        self.eps = 1e-5  # ゼロ除算を防ぐ閾値

    def forward(self, x):
        x = x.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # GPUに送る
        cqt_spec = self.cqt(x)  # (B, n_bins, time)
        safe_spec = torch.clamp(cqt_spec, min=self.eps)  # 小さな値をカット
        log_cqt = torch.log(safe_spec)  # 対数変換
        return log_cqt  # (B, n_bins, time)


    """
    利用例：
    import torch

    # サンプルレート: 16kHz、FFTポイント: 1024、メルバンド: 80、ホップ長: 256
    log_melspec = LogMelspec(sample_rate=16000, n_fft=1024, n_mels=80, hop_length=256)

    # サンプル波形データ（例: バッチサイズ1、長さ16000サンプル）
    x = torch.randn(1, 16000)

    # メルスペクトログラムを計算
    log_mel = log_melspec(x)

    print(log_mel.shape)  # 出力: (1, 80, フレーム数)

    """

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

"""# Trainer"""

class LitTranscriber(pl.LightningModule):
    def __init__(
        self,
        transcriber_args: dict, # Seq2SeqTranscriber モジュールの初期化に必要な引数を含む辞書. 例えばn_melsやn_fft, hop_lengthなど
        lr: float, # 学習率（Learning Rate）。
        lr_decay: float = 1.0, # 学習率の減衰率。デフォルト値は 1.0（減衰なし）。
        lr_decay_interval: int = 1, # 学習率の減衰間隔（エポック単位）。
    ):
        super().__init__()
        self.save_hyperparameters() # 渡された引数をハイパーパラメータとして保存（PyTorch Lightning機能）。
        self.voc_dict = voc_single_track # voc_dict をインスタンス変数に保存。
        self.n_voc = self.voc_dict["n_voc"] # トークンの総数（n_voc）も保存。

        # 渡された引数 transcriber_args を展開して Seq2SeqTranscriber モジュールを初期化。トークン辞書（voc_dict）を追加で渡す。
        self.transcriber = Seq2SeqTranscriber(
            **transcriber_args, voc_dict=self.voc_dict
        )
        # 引数から渡された学習率や減衰関連の設定を保存。
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_decay_interval = lr_decay_interval

    # 推論時に Seq2SeqTranscriber の推論機能（infer メソッド）を呼び出す関数
    def forward(self, y: torch.Tensor):
        transcriber_infer = self.transcriber.infer(y) # 入力波形（y）を Seq2SeqTranscriber の infer メソッドに渡して推論を実行。
        return transcriber_infer #推論結果として生成されたトークン列を返します。

    # モデルのトレーニング中に1バッチの処理を実行する関数
    def training_step(self, batch, batch_idx):
        y, t = batch # 入力波形（y）と正解トークン列（t）を分割。
        tf_out = self.transcriber(y, t) # 入力データを Seq2SeqTranscriber に渡してtranscriberのforwardで損失値（loss）を取得。出力は、損失値やロジット（各トークンのスコア）を含むオブジェクト。
        loss = tf_out.loss # T5モデルの出力に含まれる損失値（loss）。学習時の損失計算は、CrossEntropyLoss に基づいています。
        t = t.detach() # 正解トークン（t）を計算グラフから切り離して、後続の計算が逆伝播に影響しないようにする。
        mask = t != self.voc_dict["pad"] # 正解トークン列のうち、パディングトークン（pad）を無視するためのマスクを作成。

        # モデルの出力（logits）から最も高いスコアのトークン（argmax(-1)）を取得し、マスクされた正解トークン（t[mask]）と比較。
        # 正しく予測されたトークンの数を、マスクされたトークン全体の数で割って精度を計算。
        accr = (tf_out.logits.argmax(-1)[mask] == t[mask]).sum() / mask.sum()
        #損失値（loss）と精度（accr）を PyTorch Lightning のログ機能を使って記録。
        self.log("train_loss", loss)
        self.log("train_accr", accr)
        return loss # 計算された損失を返します。この値はPyTorch Lightningがバックプロパゲーションを行うために使用します。

    # モデルの検証時に1バッチの処理を実行する関数
    def validation_step(self, batch, batch_idx):
        assert not self.transcriber.training # モデルがトレーニングモードでないことを確認
        y, t = batch
        tf_out = self.transcriber(y, t)
        loss = tf_out.loss
        t = t.detach()
        mask = t != self.voc_dict["pad"]
        accr = (tf_out.logits.argmax(-1)[mask] == t[mask]).sum() / mask.sum()
        self.log("vali_loss", loss)
        self.log("vali_accr", accr)
        return loss

    def test_step(self, batch, batch_idx):
        y, ref_midi = batch # 音声データ（y）と参照MIDI（ref_midi）を分割
        # テストデータセットでは、通常1つのサンプルを1バッチとして処理するため、インデックス0の要素を取り出します。
        y = y[0]
        ref_midi = ref_midi[0]
        with torch.no_grad(): # 推論中は勾配計算が不要なため、この文脈を使ってメモリ消費を削減。
            est_tokens = self.forward(y) # 入力音声 y を用いてモデルに推論を実行し、MIDIトークン列（est_tokens）を生成。モデルの forward メソッドを呼び出して推論を実行。この時forward内では分割されて実行される, 形状： (セグメント数, 最大トークン長)
            unpadded_tokens = unpack_sequence(est_tokens.cpu().numpy()) # 推論結果（est_tokens）からパディングを取り除き、有効なトークン列だけを取得。GPU上のテンソルをCPUに移動してNumPy配列に変換
            unpadded_tokens = [t[1:] for t in unpadded_tokens] # 各トークン列から開始トークン（<sos>など）を除外。
            est_midi = token_seg_list_to_midi(unpadded_tokens) # パディングを除去したトークン列（unpadded_tokens）をMIDI形式に変換。
        dict_eval = evaluate_midi(est_midi, ref_midi) # 参照MIDI（ref_midi）と推論MIDI（est_midi）を比較し、性能を評価。
        # 評価結果をPyTorch Lightningのログ機能で記録。
        dict_log = {}
        for key in dict_eval:
            dict_log["test/" + key] = dict_eval[key]
        self.log_dict(dict_log, batch_size=1)

    def train_dataloader(self):
        # 事前に定義されたデータセットクラス。音声（波形データ）とMIDIファイルをペアとしてロードします。
        dset = CustomDataset(
            split="train",                      # データセットの分割
            sample_rate=16000,
            apply_pedal=True,
            whole_song=False
        )
        # データローダーの作成
        return data.DataLoader(
            dataset=dset, # 使用するデータセットとして、初期化済みの Maestro を指定。
            collate_fn=dset.collate_batch, # バッチ処理時のデータ整形関数。
            batch_size=64, # 1バッチに含めるサンプル数。
            shuffle=True, # データセットをランダムにシャッフル。
            pin_memory=True, # データをピン留め（ページロック）してメモリに保存。
            num_workers=32, # データローディングに使用するプロセス数。
        )

    """
    val_dataloader関数を定義する必要がある？
    """
    def val_dataloader(self):
        # 事前に定義されたデータセットクラス。音声（波形データ）とMIDIファイルをペアとしてロードします。
        dset = CustomDataset(
            split="valid",                      # データセットの分割
            sample_rate=16000,
            apply_pedal=True,
            whole_song=False
        )
        # データローダーの作成
        return data.DataLoader(
            dataset=dset, # 使用するデータセットとして、初期化済みの Maestro を指定。
            collate_fn=dset.collate_batch, # バッチ処理時のデータ整形関数。
            batch_size=32, # 1バッチに含めるサンプル数。
            shuffle=True, # データセットをランダムにシャッフル。
            pin_memory=True, # データをピン留め（ページロック）してメモリに保存。
            num_workers=16, # データローディングに使用するプロセス数。
        )
    def test_dataloader(self):
        dset = CustomDataset(
            split="test",                      # データセットの分割
            sample_rate=16000,
            apply_pedal=True,
            whole_song=True
        )
        return data.DataLoader(
            dataset=dset,
            collate_fn=dset.collate_wholesong,
            batch_size=1,
            shuffle=False, # テストデータの順序を固定。理由: 再現性を確保するため。
            pin_memory=True,
        )

    # モデルの学習時に、損失関数の値を最小化するようにパラメータを更新するオプティマイザ（最適化手法）を指定する関数
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    # Adamオプティマイザ の改良版であり、L2正則化の代わりに ウェイトデカイ（Weight Decay） を導入したオプティマイザ。
    # self.parameters(): モデルの学習可能なすべてのパラメータ（重みとバイアス）を取得。

# Commented out IPython magic to ensure Python compatibility.
# 学習開始前に実行
# %load_ext tensorboard
# %tensorboard --logdir /content/drive/MyDrive/B4/Humtrans/m2m100/logs/transcription




# 推論処理
def infer_single_wav(model, wav_path, midi_save_path, checkpoint_path=None, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    指定したWAVファイルで推論を行い、MIDIファイルを保存する。
    Args:
        model (LitTranscriber): 学習済みのLightningModuleモデル
        wav_path (str): 推論するWAVファイルのパス
        midi_save_path (str): 保存するMIDIファイルのパス
        checkpoint_path (str): モデルのチェックポイントパス
        device (str): "cuda"または"cpu"
    """
    # チェックポイントを読み込む
    if checkpoint_path is not None:
        print(f"Loading checkpoint from: {checkpoint_path}")
        model = LitTranscriber.load_from_checkpoint(checkpoint_path).to(device)
    else:
        model = model.to(device)

    # WAVファイルの読み込みと前処理
    waveform, sr = torchaudio.load(wav_path)
    waveform = waveform.mean(0).to(device)  # モノラル化し、デバイスに移動

    if sr != model.transcriber.sr:
        waveform = torchaudio.functional.resample(
            waveform, sr, model.transcriber.sr
        ).to(device)  # サンプリング後もデバイスに移動

    # 推論
    model.eval()
    with torch.no_grad():
        print(f"Input shape before passing to model: {waveform.shape}")
        output_tokens = model(waveform)  # モデルのforwardメソッドを使用
        print(f"Output tokens after passing to model: {output_tokens}")

    # トークンをMIDIに変換
    unpadded_tokens = unpack_sequence(output_tokens.cpu().numpy())
    unpadded_tokens = [t[1:] for t in unpadded_tokens]  # 開始トークンを削除
    print("unpadded_tokens: ", unpadded_tokens)
    est_midi = token_seg_list_to_midi(unpadded_tokens)

    # MIDIファイルの保存
    est_midi.write(midi_save_path)
    print(f"MIDI saved at: {midi_save_path}")



import pretty_midi
import matplotlib.pyplot as plt

def visualize_and_play_midi(midi_path, soundfont_path="/usr/share/sounds/sf2/FluidR3_GM.sf2"):
    """
    MIDIファイルをピアノロールとして表示し、音声として再生する。
    Args:
        midi_path (str): 再生するMIDIファイルのパス
        soundfont_path (str): 使用するサウンドフォントのパス
    """
    # MIDIファイルを読み込み
    midi_data = pretty_midi.PrettyMIDI(midi_path)

    # ピアノロールを取得
    piano_roll = midi_data.get_piano_roll(fs=100)

    # ピアノロールを表示
    plt.figure(figsize=(12, 6))
    plt.imshow(piano_roll, aspect='auto', origin='lower', cmap='gray_r')
    plt.title("Piano Roll")
    plt.xlabel("Time (frames)")
    plt.ylabel("Pitch (MIDI Note Number)")
    plt.colorbar(label="Velocity")
    plt.show()

