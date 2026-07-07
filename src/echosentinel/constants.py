"""Project-wide constants: the PS-12 class map and audio standardization targets.

These values implement hard requirements from the PS-12 problem statement and
must not be changed casually: category ids and names appear verbatim in the
submission JSON, and all audio in the system is standardized to TARGET_SR
mono float32 before any feature extraction.
"""

# PS-12 Stage-1 official class map (label -> JSON category name).
CLASS_MAP: dict[int, str] = {
    1: "vessel",
    2: "marine_animal",
    3: "natural_sound",
    4: "other_anthropogenic",
}

CLASS_NAMES: list[str] = [CLASS_MAP[i] for i in sorted(CLASS_MAP)]
NAME_TO_ID: dict[str, int] = {v: k for k, v in CLASS_MAP.items()}
NUM_CLASSES: int = len(CLASS_MAP)

# Audio standardization. 32 kHz keeps everything the test hardware captured
# (observed test SRs: 8k-64k Hz; Nyquist 16 kHz covers vessel tonals, whale
# calls, dolphin whistles and surviving click energy) without wasting compute.
TARGET_SR: int = 32_000

# Feature front-end geometry shared by all models: 10 ms hop at 32 kHz.
N_FFT: int = 1024
HOP_LENGTH: int = 320
N_MELS: int = 64
FMIN: float = 20.0

FRAMES_PER_SECOND: float = TARGET_SR / HOP_LENGTH  # 100.0
