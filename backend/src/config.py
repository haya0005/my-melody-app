import lightning.pytorch as pl

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