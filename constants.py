import torch

# --- Константы TTS и Аудио ---
DEFAULT_LANGUAGE = 'ru'
DEFAULT_MODEL_ID = 'v4_ru'
DEFAULT_VOICES = ["kseniya", "xenia", "oleg", "zahar", "julia", "dmitry"]
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_DEVICE_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Константы Обработки ---
MAX_CHUNK_LENGTH = 450
NOISE_THRESHOLD = 0.005
NORMALIZE_TARGET_PEAK = 0.9

# --- Константы FFmpeg и Файлов ---
FFMPEG_BITRATE = '192k'
TEMP_DIR_PREFIX = "pdfspeaker_"

# --- Константы GUI ---
HIGHLIGHT_BG = "yellow"
HIGHLIGHT_FG = "black"