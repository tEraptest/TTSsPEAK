import numpy as np
import scipy.io.wavfile
from pathlib import Path

from constants import NOISE_THRESHOLD, NORMALIZE_TARGET_PEAK

def reduce_noise(audio_np: np.ndarray, noise_threshold: float = NOISE_THRESHOLD) -> np.ndarray:
    """Уменьшает фоновый шум с помощью порогового шумоподавления."""
    return np.where(np.abs(audio_np) > noise_threshold, audio_np, 0)

def normalize_audio(audio_np: np.ndarray, target_peak: float = NORMALIZE_TARGET_PEAK) -> np.ndarray:
    """Нормализует аудио и применяет шумоподавление."""
    audio_np = reduce_noise(audio_np)
    peak = np.max(np.abs(audio_np))
    if peak > 0:
        factor = target_peak / peak
        return audio_np * factor
    return audio_np

def change_audio_speed(audio_np: np.ndarray, speed_multiplier: float) -> np.ndarray:
    """Изменяет скорость аудио с помощью интерполяции numpy."""
    if speed_multiplier == 1.0:
        return audio_np
    print(f"Применение скорости {speed_multiplier:.1f}x к аудио")
    indices = np.arange(0, len(audio_np), speed_multiplier)
    original_indices = np.arange(len(audio_np))
    resampled_audio_np = np.interp(indices, original_indices, audio_np)
    return normalize_audio(resampled_audio_np)

def save_audio_to_wav(file_path: Path, audio_np: np.ndarray, sample_rate: int):
    """Сохраняет numpy массив аудио в WAV файл."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        wav_data = (audio_np * 32767).astype(np.int16)
        scipy.io.wavfile.write(str(file_path), sample_rate, wav_data)
        print(f"Аудио сохранено в файл: {file_path}")
    except Exception as e:
        print(f"Ошибка записи WAV файла {file_path}: {e}")
        raise 