import numpy as np
import pretty_midi as pm
import mir_eval

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