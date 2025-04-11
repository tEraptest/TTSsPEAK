import torch
import numpy as np
from typing import Optional, Any
from constants import DEFAULT_LANGUAGE, DEFAULT_MODEL_ID, DEFAULT_SAMPLE_RATE, DEFAULT_DEVICE_TYPE
import audio_utils

class TTSManager:
    """Управляет загрузкой и использованием модели Silero TTS."""
    def __init__(self,
                 language: str = DEFAULT_LANGUAGE,
                 model_id: str = DEFAULT_MODEL_ID,
                 sample_rate: int = DEFAULT_SAMPLE_RATE,
                 device_type: str = DEFAULT_DEVICE_TYPE):
        self.language = language
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.device = torch.device(device_type)
        self.tts_model: Optional[Any] = None
        self.is_ready: bool = False
        print(f"TTSManager инициализирован с устройством: {self.device}")

    def load_model(self):
        """Загружает модель TTS. Выполнять в отдельном потоке."""
        if self.is_ready:
            print("Модель TTS уже загружена.")
            return True, f"Модель TTS ({self.model_id}) уже загружена."

        print(f"Загрузка модели Silero TTS ({self.model_id})...")
        try:
            model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                      model='silero_tts',
                                      language=self.language,
                                      speaker=self.model_id,
                                      trust_repo=True)
            model.to(self.device)
            self.tts_model = model
            self.is_ready = True
            msg = f"Модель Silero TTS ({self.model_id}) успешно загружена на {self.device}."
            print(msg)
            return True, msg
        except Exception as e:
            error_msg = f"Критическая ошибка загрузки Silero TTS: {e}"
            print(error_msg)
            self.tts_model = None
            self.is_ready = False
            return False, error_msg

    def generate_chunk(self, text_to_speak: str, voice: str, speed_multiplier: float = 1.0) -> Optional[np.ndarray]:
        """Генерирует аудио фрагмент для заданного текста и голоса."""
        if not self.is_ready or not self.tts_model:
            print("Ошибка генерации: TTS модель не готова или не загружена.")
            return None
        if not text_to_speak:
            print("Ошибка генерации: Текст для озвучивания пуст.")
            return None

        try:
            cleaned_text = ' '.join(text_to_speak.split())
            if not cleaned_text:
                return None

            if len(cleaned_text) > 1000:
                 print(f"Предупреждение: Фрагмент длинный ({len(cleaned_text)}), возможны проблемы. Текст: '{cleaned_text[:60]}...'")

            print(f"Генерация аудио для: '{cleaned_text[:60]}...' (Скорость: {speed_multiplier:.1f}x, Голос: '{voice}')")

            audio_tensor = self.tts_model.apply_tts(text=cleaned_text,
                                                    speaker=voice,
                                                    sample_rate=self.sample_rate,
                                                    put_accent=True,
                                                    put_yo=True)

            if isinstance(audio_tensor, torch.Tensor):
                audio_np = audio_tensor.detach().cpu().numpy()
            else:
                audio_np = np.array(audio_tensor, dtype=np.float32)

            audio_processed = audio_utils.normalize_audio(audio_np)
            if speed_multiplier != 1.0:
                audio_processed = audio_utils.change_audio_speed(audio_processed, speed_multiplier)
                audio_processed = audio_utils.normalize_audio(audio_processed)

            return audio_processed

        except Exception as e:
            print(f"Ошибка во время генерации TTS для фрагмента: '{text_to_speak[:60]}...' Ошибка: {e}")
            return None