from operator import attrgetter
import dataclasses
import numpy as np
import pretty_midi as pm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from nnAudio.features import CQT
import soundfile as sf

from config import FRAME_PER_SEC, FRAME_STEP_SIZE_SEC, AUDIO_SEGMENT_SEC
from config import voc_single_track

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
    """
    return n * FRAME_STEP_SIZE_SEC



def find_note(list, n):
    """
    タプルリストの最初の要素から指定された値を検索し、そのインデックスを返す関数。
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


# 固定長のセグメントに分割する関数
def split_audio_into_segments(y: torch.Tensor, sr: int): # オーディオデータ（テンソル）, オーディオのサンプルレート
    audio_segment_samples = round(AUDIO_SEGMENT_SEC * sr) # 1セグメントの長さをサンプル数で計算
    pad_size = audio_segment_samples - (y.shape[-1] % audio_segment_samples) # セグメントサイズできっちり分割できるようにpadするサイズを計算

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

def normalize_rms_torch(audio_tensor, target_rms=0.1):
    rms = torch.sqrt(torch.mean(audio_tensor**2)).item()
    if rms < 1e-6:
        print("音が非常に小さいため、最小値でスケーリングします")
        rms = 1e-6
    scaling_factor = target_rms / rms
    return audio_tensor * scaling_factor

def rms_normalize_wav(input_path, output_path, target_rms=0.1):
    waveform, sr = torchaudio.load(input_path)
    waveform = waveform.mean(0, keepdim=True)  # モノラル化
    normalized = normalize_rms_torch(waveform, target_rms)
    torchaudio.save(output_path, normalized, sample_rate=sr)

