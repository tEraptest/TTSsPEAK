import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import fitz
import torch
import sounddevice as sd
import numpy as np
import threading
import time
import os
import subprocess
import scipy.io.wavfile
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple, Any # Добавлено для типизации

# --- Константы ---
DEFAULT_LANGUAGE = 'ru'
DEFAULT_MODEL_ID = 'v4_ru'
DEFAULT_VOICES = ["kseniya", "xenia", "oleg", "zahar", "julia", "dmitry"]
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_DEVICE_TYPE = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'

MAX_CHUNK_LENGTH = 450
NOISE_THRESHOLD = 0.005
NORMALIZE_TARGET_PEAK = 0.9
FFMPEG_BITRATE = '192k'
TEMP_DIR_PREFIX = "pdfspeaker_"
HIGHLIGHT_BG = "yellow"
HIGHLIGHT_FG = "black"
# --- ---

# --- Функции обработки аудио (можно вынести в отдельный модуль audio_utils.py) ---

def reduce_noise(audio_np: np.ndarray, noise_threshold: float = NOISE_THRESHOLD) -> np.ndarray:
    """Reduces background noise by applying a noise gate."""
    return np.where(np.abs(audio_np) > noise_threshold, audio_np, 0)

def normalize_audio(audio_np: np.ndarray, target_peak: float = NORMALIZE_TARGET_PEAK) -> np.ndarray:
    """Normalizes audio and applies a noise gate."""
    audio_np = reduce_noise(audio_np)  # Apply noise reduction
    peak = np.max(np.abs(audio_np))
    if peak > 0:
        factor = target_peak / peak
        return audio_np * factor
    return audio_np

def change_audio_speed(audio_np: np.ndarray, speed_multiplier: float) -> np.ndarray:
    """Changes audio speed using numpy interpolation."""
    if speed_multiplier == 1.0:
        return audio_np
    print(f"Применение скорости {speed_multiplier:.1f}x к аудио")
    indices = np.arange(0, len(audio_np), speed_multiplier)
    # Use np.linspace for potentially better handling of endpoints if needed
    original_indices = np.arange(len(audio_np))
    resampled_audio_np = np.interp(indices, original_indices, audio_np)
    # Нормализуем полученный сигнал после изменения скорости
    return normalize_audio(resampled_audio_np) # Normalize again after speed change

def save_audio_to_wav(file_path: Path, audio_np: np.ndarray, sample_rate: int):
    try:
        # Ensure the directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert to int16 for WAV standard
        wav_data = (audio_np * 32767).astype(np.int16)
        scipy.io.wavfile.write(str(file_path), sample_rate, wav_data)
        print(f"Аудио сохранено в файл: {file_path}")
    except Exception as e:
        print(f"Ошибка записи WAV файла {file_path}: {e}")
        raise # Re-raise exception to be handled by caller

# --- Функция разбивки текста (можно вынести в text_utils.py) ---
def split_text_into_chunks(text: str, max_length: int = MAX_CHUNK_LENGTH) -> List[str]:
    # ... (код функции без изменений, но с типизацией)
    chunks = []
    current_pos = 0
    while current_pos < len(text):
        end_pos = min(current_pos + max_length, len(text))
        split_index = -1
        # Prefer splitting at sentence endings
        for sep in ['.', '?', '!']:
            found_index = text.rfind(sep, current_pos, end_pos)
            if found_index != -1:
                split_index = max(split_index, found_index)

        if split_index != -1 and end_pos < len(text): # Split at sentence end
            chunk = text[current_pos : split_index + 1]
            current_pos = split_index + 1
        elif end_pos < len(text): # Need to split mid-text
            # Try splitting at the last space
            space_index = text.rfind(' ', current_pos, end_pos)
            if space_index != -1 and space_index > current_pos:
                chunk = text[current_pos : space_index]
                current_pos = space_index + 1
            else: # No space found, hard split
                chunk = text[current_pos : end_pos]
                current_pos = end_pos
        else: # Last chunk
            chunk = text[current_pos:]
            current_pos = len(text)

        cleaned_chunk = chunk.strip()
        if cleaned_chunk:
            chunks.append(cleaned_chunk)
    return chunks

class PdfReaderApp:
    def __init__(self, root_window: tk.Tk):
        self.root = root_window
        self.root.title("PDF Reader & Speaker (Русский TTS)")
        self.root.geometry("1000x750")

        # TTS Configuration (может быть загружено из конфиг файла)
        self.language = DEFAULT_LANGUAGE
        self.model_id = DEFAULT_MODEL_ID
        self.voices = DEFAULT_VOICES
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.device = torch.device(DEFAULT_DEVICE_TYPE)

        # TTS State
        self.tts_model: Optional[Any] = None # Using Any for loaded model type
        self.tts_ready: bool = False

        # PDF State
        self.pdf_document: Optional[fitz.Document] = None
        self.pdf_path: Optional[Path] = None
        self.current_page_num: int = 0
        self.total_pages: int = 0

        # Playback State
        self.playing_thread: Optional[threading.Thread] = None
        self.stop_playback_flag = threading.Event()
        self.speech_speed: float = 1.0
        self.temp_audio_files: List[Path] = [] # Use Path objects
        self.current_temp_dir: Optional[Path] = None
        self.playback_was_stopped: bool = False

        # Widgets (объявление здесь, инициализация в _create_widgets)
        self.btn_load: Optional[ttk.Button] = None
        self.entry_start_page: Optional[ttk.Entry] = None
        self.entry_end_page: Optional[ttk.Entry] = None
        self.btn_play: Optional[ttk.Button] = None
        self.btn_stop: Optional[ttk.Button] = None
        self.btn_save_mp3: Optional[ttk.Button] = None
        self.btn_save_stopped_mp3: Optional[ttk.Button] = None
        self.btn_prev: Optional[ttk.Button] = None
        self.lbl_page_display: Optional[ttk.Label] = None
        self.btn_next: Optional[ttk.Button] = None
        self.speed_slider: Optional[ttk.Scale] = None
        self.speed_label: Optional[ttk.Label] = None
        self.voice_combobox: Optional[ttk.Combobox] = None
        self.pdf_image_label: Optional[ttk.Label] = None
        self.text_display: Optional[scrolledtext.ScrolledText] = None
        self.status_label: Optional[ttk.Label] = None

        self._create_widgets()
        self.update_status("Инициализация...")

        threading.Thread(target=self._initial_tts_load, daemon=True).start()

    def _initial_tts_load(self):
        """Loads the TTS model in a separate thread."""
        self.update_status("Загрузка модели Silero TTS...")
        try:
            model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                      model='silero_tts',
                                      language=self.language,
                                      speaker=self.model_id,
                                      trust_repo=True)
            model.to(self.device)
            self.tts_model = model
            self.tts_ready = True
            status_msg = f"Модель Silero TTS ({self.model_id}, {self.voices[0]}) загружена."
            print(status_msg)

            # Обновление состояния GUI должно выполняться в основном потоке
            self.root.after(0, self.update_status, status_msg)
            self.root.after(0, self.enable_controls)
        except Exception as e:
            error_msg = f"Критическая ошибка загрузки Silero TTS: {e}"
            print(error_msg)
           
            def show_error():
                self.update_status("Ошибка загрузки TTS модели. Озвучивание недоступно.")
                if self.root.winfo_exists():
                    try:
                        messagebox.showerror("Ошибка TTS", f"Не удалось загрузить модель Silero TTS.\nОшибка: {e}")
                    except tk.TclError:
                        print("Не удалось показать messagebox (root уничтожен?).")
            
            if self.root and self.root.winfo_exists():
                 self.root.after(0, show_error)

    def _generate_audio_chunk(self, text_to_speak: str, speed_multiplier: float, voice: str) -> Optional[np.ndarray]:
        """Generates a single audio chunk using the loaded TTS model."""
        if not self.tts_ready or not self.tts_model or not text_to_speak:
            print("TTS не готов, модель не загружена или текст пустой.")
            return None

        try:
            cleaned_text = ' '.join(text_to_speak.split())
            if not cleaned_text:
                return None

            if len(cleaned_text) > 1000:
                 print(f"Предупреждение: Фрагмент слишком длинный ({len(cleaned_text)}), возможны проблемы. Текст: '{cleaned_text[:60]}...'")

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

            audio_processed = normalize_audio(audio_np) 
            if speed_multiplier != 1.0:
                audio_processed = change_audio_speed(audio_processed, speed_multiplier)
                audio_processed = normalize_audio(audio_processed)

            return audio_processed

        except Exception as e:
            print(f"Ошибка во время генерации TTS для фрагмента: {e}")
            return None 

    def _play_audio_chunk(self, audio_np: Optional[np.ndarray]) -> bool:
        """Plays a numpy audio array."""
        if audio_np is None or audio_np.size == 0:
            print("Нет аудио данных для воспроизведения.")
            return False
        try:
            print(f"Воспроизведение аудио фрагмента (длина: {len(audio_np)/self.sample_rate:.2f} сек)...")
            sd.play(audio_np, self.sample_rate)
            sd.wait()
            print("Воспроизведение фрагмента завершено.")
            return True
        except Exception as e:
            print(f"Ошибка воспроизведения аудио: {e}")
            sd.stop()
            return False

    # --- Методы для виджетов и логики приложения ---

    def _create_widgets(self):
        self._create_top_frame()
        self._create_display_pane()
        self._create_status_bar()

    def _create_top_frame(self):
        top_frame = ttk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        self._create_control_panel(top_frame)
        self._create_nav_speed_panel(top_frame)

    def _create_control_panel(self, parent_frame):
        control_frame = ttk.LabelFrame(parent_frame, text="Управление", padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_load = ttk.Button(control_frame, text="Загрузить PDF", command=self.select_pdf)
        self.btn_load.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        lbl_start = ttk.Label(control_frame, text="С:")
        lbl_start.grid(row=0, column=1, padx=(10, 0), pady=5, sticky="w")
        self.entry_start_page = ttk.Entry(control_frame, width=5, state=tk.DISABLED)
        self.entry_start_page.grid(row=0, column=2, padx=2, pady=5, sticky="w")

        lbl_end = ttk.Label(control_frame, text="До:")
        lbl_end.grid(row=0, column=3, padx=(10, 0), pady=5, sticky="w")
        self.entry_end_page = ttk.Entry(control_frame, width=5, state=tk.DISABLED)
        self.entry_end_page.grid(row=0, column=4, padx=2, pady=5, sticky="w")

        self.btn_play = ttk.Button(control_frame, text="▶ Озвучить", command=self.play_range, state=tk.DISABLED)
        self.btn_play.grid(row=0, column=5, padx=5, pady=5, sticky="w")

        self.btn_stop = ttk.Button(control_frame, text="⏹ Стоп", command=self.stop_audio, state=tk.DISABLED)
        self.btn_stop.grid(row=0, column=6, padx=5, pady=5, sticky="w")

        save_frame = ttk.Frame(control_frame)
        save_frame.grid(row=1, column=0, columnspan=7, sticky="ew", pady=5)

        self.btn_save_mp3 = ttk.Button(save_frame, text="💾 Сохранить MP3 (полное)", command=self.save_full_audio_to_mp3, state=tk.DISABLED)
        self.btn_save_mp3.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        self.btn_save_stopped_mp3 = ttk.Button(save_frame, text="💾 Сохранить MP3 (остановленное)", command=self.save_stopped_audio_to_mp3, state=tk.DISABLED)
        self.btn_save_stopped_mp3.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        control_frame.columnconfigure(0, weight=1)


    def _create_nav_speed_panel(self, parent_frame):
        nav_speed_frame = ttk.LabelFrame(parent_frame, text="Навигация и настройки", padding="10")
        nav_speed_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        self.btn_prev = ttk.Button(nav_speed_frame, text="< Пред.", command=self.prev_page, state=tk.DISABLED)
        self.btn_prev.grid(row=0, column=0, padx=5, pady=5)

        self.lbl_page_display = ttk.Label(nav_speed_frame, text="Страница: - / -", width=15, anchor="center")
        self.lbl_page_display.grid(row=0, column=1, padx=5, pady=5)

        self.btn_next = ttk.Button(nav_speed_frame, text="След. >", command=self.next_page, state=tk.DISABLED)
        self.btn_next.grid(row=0, column=2, padx=5, pady=5)

        nav_speed_frame.grid_columnconfigure(3, weight=1)

        lbl_speed = ttk.Label(nav_speed_frame, text="Скорость:")
        lbl_speed.grid(row=0, column=4, padx=(10, 0), pady=5, sticky="e")

        self.speed_slider = ttk.Scale(nav_speed_frame, from_=0.5, to=2.0, length=150, value=self.speech_speed, orient=tk.HORIZONTAL, command=self.update_speed)
        self.speed_slider.grid(row=0, column=5, padx=5, pady=5, sticky="ew")

        self.speed_label = ttk.Label(nav_speed_frame, text=f"{self.speech_speed:.1f}x", width=5)
        self.speed_label.grid(row=0, column=6, padx=(0, 5), pady=5)

        lbl_voice = ttk.Label(nav_speed_frame, text="Голос:")
        lbl_voice.grid(row=0, column=7, padx=(10, 0), pady=5, sticky="e")
        self.voice_combobox = ttk.Combobox(nav_speed_frame, values=self.voices, state="readonly", width=10)
        self.voice_combobox.grid(row=0, column=8, padx=5, pady=5, sticky="w")
        if self.voices:
            self.voice_combobox.set(self.voices[0])

        nav_speed_frame.grid_columnconfigure(5, weight=1)

    def _create_display_pane(self):
        display_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        display_pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 5))
        pdf_frame = ttk.Frame(display_pane, width=500, height=600) 
        pdf_frame.pack_propagate(False) 
        self.pdf_image_label = ttk.Label(pdf_frame, anchor=tk.CENTER, background="gray")
        self.pdf_image_label.pack(fill=tk.BOTH, expand=True)
        display_pane.add(pdf_frame, weight=2) 

        text_frame = ttk.Frame(display_pane, width=400, height=600) 
        text_frame.pack_propagate(False)
        text_container = ttk.Frame(text_frame)
        text_container.pack(fill=tk.BOTH, expand=True)
        self.text_display = scrolledtext.ScrolledText(text_container, wrap=tk.WORD, state=tk.DISABLED, height=10, relief=tk.FLAT, bd=0) # Minimal border
        self.text_display.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.text_display.tag_configure("highlight", background=HIGHLIGHT_BG, foreground=HIGHLIGHT_FG)
        display_pane.add(text_frame, weight=1)


    def _create_status_bar(self):
        self.status_label = ttk.Label(self.root, text="Загрузите PDF файл...", relief=tk.SUNKEN, anchor=tk.W, padding="2 5")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)


    def update_speed(self, value: str):
        """Updates speech speed from slider."""
        try:
            self.speech_speed = round(float(value), 1)
            if self.speed_label:
                self.speed_label.config(text=f"{self.speech_speed:.1f}x")
        except ValueError:
            pass


    def enable_controls(self, playback_running: bool = False):
        """Enables/disables controls based on state."""
        if not self.root or not self.root.winfo_exists():
            return

        is_ready = self.tts_ready and self.pdf_document is not None

        normal_state = tk.NORMAL
        disabled_state = tk.DISABLED

        play_state = disabled_state if playback_running else (normal_state if is_ready else disabled_state)
        stop_state = normal_state if playback_running else disabled_state

        nav_range_state = disabled_state if playback_running else (normal_state if is_ready else disabled_state)

        load_state = disabled_state if playback_running else normal_state

        save_full_state = disabled_state if playback_running else (normal_state if is_ready else disabled_state)
        can_save_stopped = is_ready and not playback_running and self.playback_was_stopped and self.temp_audio_files
        save_stopped_state = normal_state if can_save_stopped else disabled_state


        if self.btn_play: self.btn_play.config(state=play_state)
        if self.btn_stop: self.btn_stop.config(state=stop_state)
        if self.entry_start_page: self.entry_start_page.config(state=nav_range_state)
        if self.entry_end_page: self.entry_end_page.config(state=nav_range_state)
        if self.btn_prev: self.btn_prev.config(state=nav_range_state)
        if self.btn_next: self.btn_next.config(state=nav_range_state)
        if self.btn_load: self.btn_load.config(state=load_state)
        if self.btn_save_mp3: self.btn_save_mp3.config(state=save_full_state)
        if self.btn_save_stopped_mp3: self.btn_save_stopped_mp3.config(state=save_stopped_state)
        if self.speed_slider: self.speed_slider.config(state=normal_state if is_ready else disabled_state) # Allow speed change anytime ready
        if self.voice_combobox: self.voice_combobox.config(state=normal_state if is_ready else disabled_state) # Allow voice change anytime ready

        if self.text_display: self.text_display.config(state=tk.DISABLED)


    def select_pdf(self):
        """Opens a dialog to select a PDF file and loads it."""
        filepath_str = filedialog.askopenfilename(
            title="Выберите PDF файл",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")]
        )
        if not filepath_str:
            return

        filepath = Path(filepath_str)
        self.stop_audio()
        self.cleanup_temp_files()

        try:
            if self.pdf_document:
                print("Закрытие предыдущего PDF...")
                self.pdf_document.close()
                self.pdf_document = None

            print(f"Загрузка PDF: {filepath}")
            self.pdf_document = fitz.open(filepath)
            self.pdf_path = filepath
            self.total_pages = len(self.pdf_document)
            self.current_page_num = 0

            self.update_status(f"Загружен: {filepath.name}, Страниц: {self.total_pages}")

            if self.entry_start_page:
                self.entry_start_page.config(state=tk.NORMAL)
                self.entry_start_page.delete(0, tk.END)
                self.entry_start_page.insert(0, "1")
            if self.entry_end_page:
                self.entry_end_page.config(state=tk.NORMAL)
                self.entry_end_page.delete(0, tk.END)
                self.entry_end_page.insert(0, str(self.total_pages))

            self.show_page(self.current_page_num) 
            self.enable_controls()

        except fitz.fitz.FileNotFoundError:
             error_msg = f"Файл не найден: {filepath}"
             print(error_msg)
             messagebox.showerror("Ошибка загрузки PDF", error_msg)
             self.reset_pdf_state()
        except fitz.fitz.FileDataError as e:
             error_msg = f"Не удалось открыть файл (возможно, поврежден или не PDF):\n{filepath}\n\nОшибка: {e}"
             print(error_msg)
             messagebox.showerror("Ошибка загрузки PDF", error_msg)
             self.reset_pdf_state()
        except Exception as e:
            error_msg = f"Неизвестная ошибка при открытии PDF:\n{filepath}\n\nОшибка: {e}"
            print(error_msg)
            messagebox.showerror("Ошибка загрузки PDF", error_msg)
            self.reset_pdf_state()

    def reset_pdf_state(self):
         """Resets variables related to the loaded PDF."""
         if self.pdf_document:
             try:
                 self.pdf_document.close()
             except Exception as e:
                 print(f"Ошибка при закрытии PDF в reset_pdf_state: {e}")
         self.pdf_document = None
         self.pdf_path = None
         self.total_pages = 0
         self.current_page_num = -1
         self.update_status("Ошибка загрузки PDF или PDF не загружен.")
         self.clear_display()
         self.enable_controls()


    def clear_display(self):
        """Clears the PDF image and text display areas."""
        if self.pdf_image_label:
            self.pdf_image_label.config(image='')
            self.pdf_image_label.image = None # Keep reference to avoid GC issues
        if self.text_display:
            try:
                self.text_display.config(state=tk.NORMAL)
                self.text_display.delete('1.0', tk.END)
                self.text_display.config(state=tk.DISABLED)
            except tk.TclError as e:
                 print(f"Ошибка Tkinter при очистке текстового поля: {e}")
        if self.lbl_page_display:
            self.lbl_page_display.config(text="Страница: - / -")


    def show_page(self, page_index: int):
        """Displays the specified page number (image and text)."""
        if not self.pdf_document or not (0 <= page_index < self.total_pages):
            print(f"Попытка показать некорректную страницу: {page_index+1}")
            return

        self.current_page_num = page_index
        self.update_page_label()
        try:
            page = self.pdf_document.load_page(page_index)
            zoom = 1.5
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            def _update_image():
                if not self.pdf_image_label or not self.pdf_image_label.winfo_exists(): return
                img_tk = ImageTk.PhotoImage(img)
                self.pdf_image_label.config(image=img_tk)
                self.pdf_image_label.image = img_tk

            self.root.after(0, _update_image)

        except Exception as e:
            print(f"Ошибка рендеринга страницы {page_index + 1}: {e}")
            def _clear_image():
                 if not self.pdf_image_label or not self.pdf_image_label.winfo_exists(): return
                 self.pdf_image_label.config(image='')
                 self.pdf_image_label.image = None
            self.root.after(0, _clear_image)
            self.update_status(f"Ошибка отображения страницы {page_index + 1}")
        try:
            text = page.get_text("text", sort=True).strip()

            def _update_text():
                if not self.text_display or not self.text_display.winfo_exists(): return
                try:
                    self.text_display.config(state=tk.NORMAL)
                    self.text_display.delete('1.0', tk.END)
                    if text:
                        self.text_display.insert('1.0', text)
                    else:
                        self.text_display.insert('1.0', f"[Страница {page_index + 1} не содержит извлекаемого текста]")
                    self.text_display.config(state=tk.DISABLED)
                    self.text_display.see("1.0") # Scroll to top
                    self.text_display.tag_remove("highlight", "1.0", tk.END)
                except tk.TclError as e_tk:
                    print(f"Ошибка Tkinter при обновлении текста: {e_tk}")
                except Exception as e_upd:
                     print(f"Неизвестная ошибка при обновлении текста: {e_upd}")


            self.root.after(0, _update_text)

        except Exception as e:
            print(f"Ошибка извлечения текста со страницы {page_index + 1}: {e}")
            def _update_text_error():
                 if not self.text_display or not self.text_display.winfo_exists(): return
                 try:
                    self.text_display.config(state=tk.NORMAL)
                    self.text_display.delete('1.0', tk.END)
                    self.text_display.insert('1.0', f"[Ошибка извлечения текста со стр. {page_index + 1}]")
                    self.text_display.config(state=tk.DISABLED)
                 except tk.TclError as e_tk:
                      print(f"Ошибка Tkinter при обновлении текста (ошибка): {e_tk}")
            self.root.after(0, _update_text_error)


    def update_page_label(self):
        """Updates the page number display label."""
        if self.lbl_page_display and self.lbl_page_display.winfo_exists():
            if self.pdf_document and self.total_pages > 0:
                txt = f"Стр: {self.current_page_num + 1} / {self.total_pages}"
            else:
                txt = "Стр: - / -"
            self.lbl_page_display.config(text=txt)

    def prev_page(self):
        """Goes to the previous page."""
        if self.pdf_document and self.current_page_num > 0:
            self.show_page(self.current_page_num - 1)

    def next_page(self):
        """Goes to the next page."""
        if self.pdf_document and self.current_page_num < self.total_pages - 1:
            self.show_page(self.current_page_num + 1)

    def _validate_page_range(self) -> Optional[Tuple[int, int]]:
        """Validates the start and end page numbers from the entries."""
        if not self.pdf_document:
             messagebox.showwarning("Нет PDF", "Сначала загрузите PDF файл.")
             return None
        if not self.entry_start_page or not self.entry_end_page:
             messagebox.showerror("Ошибка GUI", "Элементы управления диапазоном страниц не инициализированы.")
             return None

        try:
            start_page_one_based = int(self.entry_start_page.get())
            end_page_one_based = int(self.entry_end_page.get())

            start_page_idx = start_page_one_based - 1
            end_page_idx = end_page_one_based - 1

            if not (0 <= start_page_idx < self.total_pages):
                raise ValueError(f"Начальная страница '{start_page_one_based}' вне допустимого диапазона (1-{self.total_pages}).")
            if not (0 <= end_page_idx < self.total_pages):
                 raise ValueError(f"Конечная страница '{end_page_one_based}' вне допустимого диапазона (1-{self.total_pages}).")
            if start_page_idx > end_page_idx:
                raise ValueError("Начальная страница не может быть больше конечной.")

            return start_page_idx, end_page_idx

        except ValueError as ve:
            messagebox.showerror("Ошибка диапазона", f"Некорректный диапазон страниц.\n{ve}")
            return None
        except Exception as e:
             messagebox.showerror("Ошибка ввода", f"Некорректный ввод в полях страниц: {e}")
             return None

    def play_range(self):
        """Starts playing audio for the selected page range."""
        if not self.tts_ready:
            messagebox.showerror("TTS не готов", "Модель озвучивания не загружена или не готова.")
            return
        if self.playing_thread and self.playing_thread.is_alive():
            messagebox.showinfo("Занято", "Воспроизведение уже идет. Остановите текущее.")
            return

        page_range = self._validate_page_range()
        if page_range is None:
            return

        start_page_idx, end_page_idx = page_range

        # Prepare for playback
        self.stop_playback_flag.clear() # Ensure flag is reset
        self.playback_was_stopped = False
        self.cleanup_temp_files() # Clear any previous temp files before starting new playback
        self.temp_audio_files = [] # Reset list

        try:
            # Create a *new* temporary directory for this playback session
            self.current_temp_dir = Path(tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX + "play_"))
            print(f"Создана временная папка для воспроизведения: {self.current_temp_dir}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось создать временную директорию: {e}")
            return

        selected_voice = self.voice_combobox.get() if self.voice_combobox else self.voices[0]

        # Start playback in a separate thread
        thread_args = (start_page_idx, end_page_idx, selected_voice, self.current_temp_dir)
        self.playing_thread = threading.Thread(target=self._play_audio_thread, args=thread_args, daemon=True)

        self.enable_controls(playback_running=True) # Disable controls, enable Stop
        self.update_status(f"Запуск озвучивания страниц {start_page_idx + 1}-{end_page_idx + 1}...")
        self.playing_thread.start()


    def _play_audio_thread(self, start_page_idx: int, end_page_idx: int, voice: str, temp_dir: Path):
        """
        Worker thread for generating and playing audio page by page.
        Manages temporary WAV files within the provided temp_dir.
        """
        print(f"Поток озвучки: Начало для стр {start_page_idx + 1}-{end_page_idx + 1}, Голос: '{voice}', Папка: {temp_dir}")
        playback_successful = True # Flag to track if playback completed without errors/stops

        try:
            for page_num in range(start_page_idx, end_page_idx + 1):
                # --- Check for stop signal or closed PDF ---
                if self.stop_playback_flag.is_set():
                    print(f"Поток озвучки: Остановка обнаружена перед страницей {page_num + 1}.")
                    playback_successful = False
                    break
                if not self.pdf_document:
                    print(f"Поток озвучки: PDF документ закрыт перед страницей {page_num + 1}.")
                    playback_successful = False
                    break # Exit loop if PDF is closed

                # --- Update GUI (Show page, update status) ---
                self.root.after(0, lambda p=page_num: self.show_page(p))
                self.root.after(0, self.update_status, f"Обработка страницы {page_num + 1} / {self.total_pages}...")
                # Brief pause allows GUI to update and checks stop flag again
                time.sleep(0.1)
                if self.stop_playback_flag.is_set():
                    print(f"Поток озвучки: Остановка обнаружена после обновления GUI для стр {page_num + 1}.")
                    playback_successful = False
                    break

                # --- Get Page Text ---
                try:
                    page = self.pdf_document.load_page(page_num)
                    full_page_text = page.get_text("text", sort=True).strip()
                    page = None # Release page object
                except Exception as e:
                    print(f"Поток озвучки: Ошибка чтения текста стр {page_num + 1}: {e}")
                    self.root.after(0, self.update_status, f"Ошибка чтения текста стр {page_num + 1}. Пропуск.")
                    time.sleep(1) # Pause briefly on error
                    continue # Skip to next page

                if not full_page_text:
                    self.root.after(0, self.update_status, f"Стр. {page_num + 1}: нет текста для озвучивания. Пропуск.")
                    time.sleep(0.5)
                    continue

                # --- Split Text and Process Chunks ---
                text_chunks = split_text_into_chunks(full_page_text)
                num_chunks = len(text_chunks)
                print(f"Стр. {page_num + 1}: {num_chunks} фрагмент(ов) текста.")

                for i, chunk in enumerate(text_chunks):
                    if self.stop_playback_flag.is_set():
                        print(f"Поток озвучки: Остановка обнаружена перед фрагментом {i+1} стр {page_num + 1}.")
                        playback_successful = False
                        break # Break inner loop (chunks)
                    if not self.pdf_document: # Check again
                        print(f"Поток озвучки: PDF закрыт перед фрагментом {i+1} стр {page_num + 1}.")
                        playback_successful = False
                        break

                    # Update status and highlight text in GUI
                    self.root.after(0, self.update_status, f"Стр. {page_num + 1}: Озвучивание фрагмента {i+1}/{num_chunks}...")
                    self.root.after(0, lambda fp=full_page_text, c=chunk: self.highlight_text(fp, c))

                    # Generate audio for the chunk
                    audio_np = self._generate_audio_chunk(chunk, self.speech_speed, voice)

                    if audio_np is not None and audio_np.size > 0:
                        # Save chunk to a temporary WAV file for potential later use (saving stopped audio)
                        temp_wav_path = temp_dir / f"chunk_{page_num:04d}_{i:04d}.wav"
                        try:
                            # Use the dedicated save function
                            save_audio_to_wav(temp_wav_path, audio_np, self.sample_rate)
                            self.temp_audio_files.append(temp_wav_path) # Add path to the list
                            print(f"Временный файл сохранен: {temp_wav_path}")
                        except Exception as write_e:
                            print(f"Ошибка записи временного WAV файла {temp_wav_path}: {write_e}")
                            # Decide if this is critical: maybe continue playback but disable saving stopped audio?
                            # For now, let's try playing anyway.

                        # Play the generated audio chunk
                        play_success = self._play_audio_chunk(audio_np)
                        if not play_success:
                            # Error during playback (e.g., sound device issue)
                            self.root.after(0, self.update_status, f"Ошибка воспроизведения на стр. {page_num + 1}. Остановка.")
                            self.stop_playback_flag.set() # Signal stop
                            playback_successful = False
                            break # Break inner loop

                    elif audio_np is None: # TTS generation failed for this chunk
                         self.root.after(0, self.update_status, f"Ошибка генерации TTS на стр. {page_num + 1}. Остановка.")
                         print(f"Ошибка TTS для фрагмента: '{chunk[:60]}...'")
                         self.stop_playback_flag.set() # Signal stop on TTS error
                         playback_successful = False
                         break # Break inner loop

                    # Final check in inner loop
                    if self.stop_playback_flag.is_set():
                        print(f"Поток озвучки: Остановка обнаружена после фрагмента {i+1} стр {page_num + 1}.")
                        playback_successful = False
                        break # Break inner loop

                # If inner loop was broken (due to stop or error), break outer loop too
                if not playback_successful:
                    break

            # --- End of Page Loop ---

        except Exception as e:
             # Catch unexpected errors in the thread loop
             print(f"Критическая ошибка в потоке озвучивания: {e}")
             import traceback
             traceback.print_exc()
             self.root.after(0, self.update_status, f"Критическая ошибка озвучивания: {e}")
             playback_successful = False
             # Ensure stop flag is set on unexpected error to trigger correct cleanup
             self.stop_playback_flag.set()

        finally:
            # --- Cleanup and GUI Update (Always runs) ---
            print("Поток озвучки: Блок finally достигнут.")
            # Stop any potentially lingering sounddevice playback, just in case
            sd.stop()
            # Schedule the final GUI update on the main thread
            # Pass whether it completed normally (True) or was stopped/errored (False)
            self.root.after(0, self.on_playback_finished, playback_successful)


    def stop_audio(self):
        """Signals the playback thread to stop and stops sounddevice."""
        if self.playing_thread and self.playing_thread.is_alive():
            print("Сигнал остановки потоку воспроизведения...")
            self.playback_was_stopped = True # Indicate stop was initiated by user/button
            self.stop_playback_flag.set() # Signal the thread
            sd.stop() # Immediately stop current audio output
            self.update_status("Остановка воспроизведения...")
            # Do not join the thread here, let on_playback_finished handle GUI updates
        else:
            # If no thread is running, still ensure sounddevice is stopped
            sd.stop()
            print("Нет активного потока воспроизведения для остановки.")


    def on_playback_finished(self, completed_normally: bool):
        """
        Called from the main thread after the playback thread finishes or is stopped.
        Handles GUI updates and cleanup logic.
        """
        print(f"Обработка завершения озвучивания. Завершено нормально: {completed_normally}, Остановка кнопкой: {self.playback_was_stopped}")

        # Reset playing thread reference
        self.playing_thread = None

        if completed_normally and not self.playback_was_stopped:
            # Playback finished the whole range without stop/error
            self.update_status("Озвучивание диапазона завершено.")
            # Clean up temp files immediately if playback completed fully
            self.cleanup_temp_files()
            self.playback_was_stopped = False # Reset flag
        elif self.playback_was_stopped:
            # Playback was stopped by the user via the stop button
            self.update_status("Воспроизведение остановлено. Временные файлы сохранены для возможного сохранения MP3.")
            # Keep temp files (don't call cleanup_temp_files here)
        else:
            # Playback stopped due to an error or PDF closure
            # Status might have already been updated by the thread for specific errors
            current_status = self.status_label.cget("text") if self.status_label else ""
            if "Ошибка" not in current_status and "Критическая" not in current_status:
                 self.update_status("Воспроизведение прервано.")
            # Clean up temp files on error
            self.cleanup_temp_files()
            self.playback_was_stopped = False # Reset flag

        # Always re-enable controls after playback finishes/stops/errors
        self.enable_controls(playback_running=False)


    def cleanup_temp_files(self, keep_parent_dir: bool = False):
        """
        Cleans up temporary audio files and optionally the parent directory.
        If keep_parent_dir is True, only files are deleted, directory remains (used for saving stopped audio).
        """
        if not self.temp_audio_files and not self.current_temp_dir:
            print("Нет временных файлов или папки для очистки.")
            return

        print(f"Начало очистки временных данных...")
        cleaned_files_count = 0

        # Clean up individual files first
        files_to_remove = list(self.temp_audio_files) # Work on a copy
        self.temp_audio_files.clear() # Clear the main list

        for f_path in files_to_remove:
            try:
                if f_path.exists():
                    f_path.unlink() # Use unlink for files
                    cleaned_files_count += 1
                    # print(f"Удален временный файл: {f_path}") # Verbose logging
                else:
                    print(f"Файл для очистки не найден: {f_path}")
            except OSError as e_clean:
                print(f"Не удалось удалить временный файл {f_path}: {e_clean}")

        print(f"Удалено временных файлов: {cleaned_files_count}.")

        # Clean up the parent directory if requested and known
        if not keep_parent_dir and self.current_temp_dir:
            temp_dir_path = self.current_temp_dir
            self.current_temp_dir = None # Reset tracker
            print(f"Попытка удаления временной папки: {temp_dir_path}")
            try:
                if temp_dir_path.exists() and temp_dir_path.is_dir():
                    shutil.rmtree(temp_dir_path, ignore_errors=True) # Use ignore_errors for robustness
                    print(f"Временная папка удалена: {temp_dir_path}")
                elif not temp_dir_path.exists():
                     print(f"Временная папка уже удалена: {temp_dir_path}")

            except Exception as e_rmdir:
                # shutil.rmtree with ignore_errors=True should suppress most errors
                print(f"Не удалось полностью удалить временную папку {temp_dir_path}: {e_rmdir}")

        # Update button states after cleanup (especially save stopped)
        # Use root.after to ensure it runs after current event processing
        self.root.after(0, self.enable_controls, False)


    def highlight_text(self, full_page_text: str, current_chunk: str):
        """Highlights the current chunk being spoken in the text display."""
        if not self.text_display or not self.text_display.winfo_exists():
            return # Exit if text widget is gone

        try:
            # Ensure text display is enabled for modification
            self.text_display.config(state=tk.NORMAL)

            # Check if the content needs refreshing (e.g., if user navigated away and back)
            # Getting the full text can be slow, so use sparingly or find a better check
            # current_displayed_text = self.text_display.get("1.0", tk.END).strip()
            # if current_displayed_text != full_page_text.strip():
            #    print("Обновление текста перед подсветкой...")
            #    self.text_display.delete('1.0', tk.END)
            #    self.text_display.insert('1.0', full_page_text)

            # Remove previous highlight
            self.text_display.tag_remove("highlight", "1.0", tk.END)

            # Find the chunk in the full text
            # Using text.find might not be robust if chunks are modified slightly (e.g., whitespace)
            # A more robust method might involve tracking character offsets, but `find` is simpler here.
            start_idx = full_page_text.find(current_chunk)

            if start_idx != -1:
                end_idx = start_idx + len(current_chunk)
                # Convert string indices to Tkinter text indices ('line.char')
                start_tk_idx = f"1.0 + {start_idx} chars"
                end_tk_idx = f"1.0 + {end_idx} chars"

                # Apply the tag and make sure it's visible
                self.text_display.tag_add("highlight", start_tk_idx, end_tk_idx)
                self.text_display.see(start_tk_idx) # Scroll to make the start visible
            else:
                print(f"Предупреждение: Не удалось найти фрагмент для подсветки: '{current_chunk[:30]}...'")


        except tk.TclError as e:
            # Handle specific Tkinter errors if the widget state is unexpected
            print(f"Ошибка Tkinter при подсветке текста: {e}")
        except Exception as e_highlight:
            # Catch any other errors during highlighting
            print(f"Неизвестная ошибка при подсветке текста: {e_highlight}")
        finally:
             # Always ensure the text display is disabled after modification
             if self.text_display and self.text_display.winfo_exists():
                 try:
                      self.text_display.config(state=tk.DISABLED)
                 except tk.TclError:
                      pass # Ignore if widget is already destroyed


    def _run_ffmpeg_concat(self, input_wav_files: List[Path], output_mp3_path: Path, temp_dir: Path) -> bool:
        """Runs ffmpeg to concatenate WAV files into a single MP3."""
        if not input_wav_files:
            print("Нет WAV файлов для объединения.")
            return False

        list_file_path = temp_dir / "ffmpeg_concat_list.txt"
        print(f"Создание файла списка для FFmpeg: {list_file_path}")

        try:
            # Create the list file for ffmpeg concat demuxer
            with open(list_file_path, 'w', encoding='utf-8') as f:
                for wav_file in input_wav_files:
                    if wav_file.exists() and wav_file.is_file():
                        # Use absolute path, replace backslashes for compatibility if needed (pathlib handles this mostly)
                        abs_path_str = str(wav_file.resolve()).replace('\\', '/')
                        f.write(f"file '{abs_path_str}'\n")
                    else:
                        print(f"Предупреждение: Пропуск отсутствующего файла в списке FFmpeg: {wav_file}")

            # Check if the list file actually contains anything
            if list_file_path.stat().st_size == 0:
                 print("Файл списка FFmpeg пуст. Нечего объединять.")
                 if list_file_path.exists(): list_file_path.unlink() # Clean up empty list file
                 return False

            # Ensure output directory exists
            output_mp3_path.parent.mkdir(parents=True, exist_ok=True)

            # Construct FFmpeg command
            command = [
                'ffmpeg',
                '-f', 'concat',         # Use the concat demuxer
                '-safe', '0',           # Allow relative/absolute paths in list file (use with caution)
                '-i', str(list_file_path), # Input list file
                '-codec:a', 'libmp3lame', # MP3 codec
                '-b:a', FFMPEG_BITRATE,  # Audio bitrate
                '-y',                   # Overwrite output file without asking
                str(output_mp3_path)    # Output MP3 file path
            ]

            print(f"Запуск FFmpeg: {' '.join(command)}")
            self.update_status("Объединение аудио и конвертация в MP3...") # Update status before running

            # Run FFmpeg process
            process = subprocess.run(
                command,
                capture_output=True,    # Capture stdout and stderr
                text=True,              # Decode output as text
                check=True,             # Raise CalledProcessError on failure (non-zero exit code)
                encoding='utf-8',       # Specify encoding for decoding output
                errors='ignore'         # Ignore decoding errors in ffmpeg output (safer)
            )

            # Log FFmpeg output (useful for debugging)
            print("FFmpeg stdout:\n", process.stdout)
            print("FFmpeg stderr:\n", process.stderr) # Often contains progress/info
            print(f"Файл MP3 успешно создан: {output_mp3_path}")
            return True

        except FileNotFoundError:
            # FFmpeg executable not found in PATH
            error_msg = "ОШИБКА: FFmpeg не найден. Убедитесь, что ffmpeg установлен и прописан в системном PATH."
            print(error_msg)
            self.root.after(0, messagebox.showerror, "Ошибка FFmpeg", error_msg)
            self.root.after(0, self.update_status, "Ошибка: FFmpeg не найден.")
            return False
        except subprocess.CalledProcessError as e:
            # FFmpeg command failed (returned non-zero exit code)
            print(f"Ошибка выполнения FFmpeg (Код возврата: {e.returncode}):")
            print("Команда:", e.cmd)
            # Log the captured output, especially stderr which usually contains the error message
            print("FFmpeg stdout:", e.stdout)
            print("FFmpeg stderr:", e.stderr)
            error_details = e.stderr[:500] if e.stderr else "(нет деталей в stderr)"
            self.root.after(0, messagebox.showerror, "Ошибка FFmpeg", f"Ошибка при конвертации аудио:\n{error_details}...")
            self.root.after(0, self.update_status, "Ошибка FFmpeg при конвертации.")
            return False
        except Exception as e_ffmpeg:
            # Catch any other unexpected errors during the process
            print(f"Неизвестная ошибка при работе с FFmpeg: {e_ffmpeg}")
            import traceback
            traceback.print_exc()
            self.root.after(0, messagebox.showerror, "Ошибка FFmpeg", f"Неизвестная ошибка при конвертации:\n{e_ffmpeg}")
            self.root.after(0, self.update_status, "Неизвестная ошибка FFmpeg.")
            return False
        finally:
            # Clean up the temporary list file regardless of success/failure
            if list_file_path and list_file_path.exists():
                try:
                    list_file_path.unlink()
                    print(f"Удален файл списка FFmpeg: {list_file_path}")
                except OSError as e_clean_list:
                    print(f"Не удалось удалить файл списка {list_file_path}: {e_clean_list}")


    # --- Full MP3 Saving ---

    def save_full_audio_to_mp3(self):
        """Initiates saving the full selected page range to an MP3 file."""
        if not self.tts_ready:
            messagebox.showerror("TTS не готов", "Модель озвучивания не готова.")
            return

        page_range = self._validate_page_range()
        if page_range is None:
            return # Validation failed

        start_page_idx, end_page_idx = page_range

        # Ask for output file path
        output_path_str = filedialog.asksaveasfilename(
            title="Сохранить ПОЛНЫЙ MP3 файл",
            defaultextension=".mp3",
            filetypes=[("MP3 Files", "*.mp3"), ("All Files", "*.*")],
            initialfile=f"{self.pdf_path.stem}_pages_{start_page_idx+1}-{end_page_idx+1}.mp3" if self.pdf_path else "output.mp3"
        )
        if not output_path_str:
            return # User cancelled

        output_path = Path(output_path_str)
        selected_voice = self.voice_combobox.get() if self.voice_combobox else self.voices[0]

        # Disable controls during save
        self.enable_controls(playback_running=True) # Use playback_running to disable most controls
        self.update_status("Генерация полного MP3 файла... Это может занять время.")
        self.root.update_idletasks() # Ensure GUI updates before starting thread

        # Run saving process in a separate thread
        save_thread = threading.Thread(
            target=self._run_full_save_mp3,
            args=(output_path, start_page_idx, end_page_idx, selected_voice),
            daemon=True
        )
        save_thread.start()


    def _run_full_save_mp3(self, output_path: Path, start_page_idx: int, end_page_idx: int, voice: str):
        """Worker thread for generating all audio chunks and concatenating them."""
        print(f"Поток сохранения (полный): Начало для стр {start_page_idx + 1}-{end_page_idx + 1}, Голос: '{voice}', Файл: {output_path}")
        temp_dir: Optional[Path] = None
        temp_wav_files: List[Path] = []
        generation_successful = True
        pdf_doc_local: Optional[fitz.Document] = None # Use a local reference

        try:
            # --- Open PDF locally in this thread ---
            # Avoids potential issues if the main thread closes the document
            if not self.pdf_path or not self.pdf_path.exists():
                 raise ValueError("Путь к PDF файлу недействителен или файл не существует.")
            pdf_doc_local = fitz.open(self.pdf_path)
            if len(pdf_doc_local) != self.total_pages:
                 print("Предупреждение: Количество страниц в повторно открытом PDF отличается.")
                 # Adjust end_page_idx if necessary? Or just proceed carefully.
                 end_page_idx = min(end_page_idx, len(pdf_doc_local) - 1)


            # --- Create Temporary Directory ---
            temp_dir = Path(tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX + "fullsave_"))
            print(f"Временная папка для полного сохранения: {temp_dir}")

            # --- Generate Audio Chunks Page by Page ---
            total_pages_in_range = end_page_idx - start_page_idx + 1
            for i, page_num in enumerate(range(start_page_idx, end_page_idx + 1)):
                current_page_display = page_num + 1
                progress_percent = int(((i + 1) / total_pages_in_range) * 100)
                self.root.after(0, self.update_status, f"Генерация MP3: Обработка стр {current_page_display}/{self.total_pages} ({progress_percent}%)...")

                try:
                    page = pdf_doc_local.load_page(page_num)
                    full_page_text = page.get_text("text", sort=True).strip()
                    page = None # Release page
                except Exception as e:
                    print(f"Ошибка чтения стр {current_page_display} при полном сохранении: {e}. Пропуск.")
                    continue # Skip this page

                if not full_page_text:
                    print(f"Стр. {current_page_display} пуста, пропуск при сохранении.")
                    continue

                text_chunks = split_text_into_chunks(full_page_text)
                num_chunks = len(text_chunks)

                for j, chunk in enumerate(text_chunks):
                    # Note: No stop flag check needed here as it's a dedicated save thread
                    # Generate audio (use speed 1.0 for saving, unless configurable)
                    audio_np = self._generate_audio_chunk(chunk, speed_multiplier=1.0, voice=voice) # Use base speed for saving

                    if audio_np is not None and audio_np.size > 0:
                        temp_wav_path = temp_dir / f"audio_{page_num:04d}_{j:04d}.wav"
                        try:
                            save_audio_to_wav(temp_wav_path, audio_np, self.sample_rate)
                            temp_wav_files.append(temp_wav_path)
                        except Exception as write_e:
                            print(f"Критическая ошибка записи временного WAV {temp_wav_path}: {write_e}")
                            generation_successful = False
                            break # Stop processing chunks on this page
                    elif audio_np is None: # TTS Error
                        print(f"Ошибка генерации TTS для фрагмента на стр. {current_page_display}. Сохранение будет неполным.")
                        generation_successful = False
                        # Decide whether to stop entirely or continue with gaps
                        # For now, let's mark as failed but continue processing other pages/chunks
                        # break # Uncomment to stop on first TTS error
                # If write failed, break outer loop
                if not generation_successful and temp_wav_files: # Check if write error occurred
                     print("Прерывание генерации из-за ошибки записи.")
                     break # Stop processing pages


            # --- Concatenate WAV files using FFmpeg ---
            if not temp_wav_files:
                message = "Не удалось сгенерировать аудио фрагменты для сохранения."
                print(message)
                self.root.after(0, messagebox.showerror, "Ошибка сохранения", message)
                generation_successful = False # Ensure final status reflects this
            elif generation_successful: # Only run ffmpeg if generation seemed ok
                ffmpeg_success = self._run_ffmpeg_concat(temp_wav_files, output_path, temp_dir)
                if ffmpeg_success:
                    final_message = f"Файл MP3 успешно сохранён:\n{output_path}"
                    self.root.after(0, messagebox.showinfo, "Успех", final_message)
                    self.root.after(0, self.update_status, "MP3 файл сохранён.")
                else:
                    # Error message already shown by _run_ffmpeg_concat
                    self.root.after(0, self.update_status, "Ошибка при объединении/конвертации MP3.")
                    generation_successful = False # Mark overall process as failed
            else:
                 # Generation failed earlier (write error or TTS error if break was used)
                 self.root.after(0, self.update_status, "Генерация аудио фрагментов не удалась. MP3 не создан.")


        except Exception as e_save:
            # Catch any other unexpected errors during the saving process
            print(f"Критическая ошибка в потоке полного сохранения: {e_save}")
            import traceback
            traceback.print_exc()
            self.root.after(0, messagebox.showerror, "Ошибка сохранения", f"Непредвиденная ошибка в процессе сохранения:\n{e_save}")
            self.root.after(0, self.update_status, "Критическая ошибка при сохранении MP3.")
            generation_successful = False
        finally:
            # --- Close local PDF reference ---
            if pdf_doc_local:
                try:
                    pdf_doc_local.close()
                except Exception as e_close:
                     print(f"Ошибка при закрытии локального PDF в потоке сохранения: {e_close}")

            # --- Clean up temporary directory ---
            if temp_dir and temp_dir.exists():
                print(f"Очистка временной папки полного сохранения: {temp_dir}")
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    print("Временная папка полного сохранения удалена.")
                except Exception as e_rmdir:
                    print(f"Не удалось полностью удалить временную папку {temp_dir}: {e_rmdir}")

            # --- Re-enable controls in GUI ---
            self.root.after(0, self.enable_controls, False) # Signal that process is finished

            # --- Final status update if needed ---
            if not generation_successful and "Ошибка" not in (self.status_label.cget("text") if self.status_label else ""):
                 self.root.after(0, self.update_status, "Сохранение MP3 завершено с ошибками.")


    # --- Stopped MP3 Saving ---

    def save_stopped_audio_to_mp3(self):
        """Initiates saving the previously generated temporary audio files to MP3."""
        if not self.playback_was_stopped or not self.temp_audio_files or not self.current_temp_dir:
            messagebox.showwarning("Нечего сохранять", "Нет записанных аудио фрагментов после остановки воспроизведения для сохранения.")
            return

        files_to_save = list(self.temp_audio_files) # Copy the list
        temp_dir_to_use = self.current_temp_dir # Get the directory path

        output_path_str = filedialog.asksaveasfilename(
            title="Сохранить ОСТАНОВЛЕННЫЙ MP3 файл",
            defaultextension=".mp3",
            filetypes=[("MP3 Files", "*.mp3"), ("All Files", "*.*")],
            initialfile=f"{self.pdf_path.stem}_stopped.mp3" if self.pdf_path else "stopped_audio.mp3"
        )
        if not output_path_str:
            return # User cancelled

        output_path = Path(output_path_str)

        # Disable controls during save
        self.enable_controls(playback_running=True)
        self.update_status("Объединение остановленных аудио файлов в MP3...")
        self.root.update_idletasks()

        # Run concatenation in a thread
        save_thread = threading.Thread(
            target=self._run_stopped_save_mp3,
            args=(output_path, files_to_save, temp_dir_to_use),
            daemon=True
        )
        save_thread.start()


    def _run_stopped_save_mp3(self, output_path: Path, files_to_concat: List[Path], temp_dir_path: Path):
        """Worker thread for concatenating existing temporary WAV files."""
        print(f"Поток сохранения (остановленный): Начало для {len(files_to_concat)} файлов из {temp_dir_path}, Выход: {output_path}")
        success = False
        try:
            if not files_to_concat:
                self.root.after(0, self.update_status, "Нет аудио файлов для сохранения.")
                return # Exit early

            # Use the common ffmpeg runner function
            success = self._run_ffmpeg_concat(files_to_concat, output_path, temp_dir_path)

            if success:
                final_message = f"Остановленное аудио успешно сохранено:\n{output_path}"
                self.root.after(0, messagebox.showinfo, "Успех", final_message)
                self.root.after(0, self.update_status, "Остановленный MP3 сохранён.")
                # Clear the flag and potentially the temp files now that they are saved
                self.playback_was_stopped = False
                # Decide if temp files should be cleaned immediately after successful save
                # self.root.after(0, self.cleanup_temp_files) # Uncomment to clean after save
            else:
                # Error message handled by _run_ffmpeg_concat
                self.root.after(0, self.update_status, "Ошибка при сохранении остановленного MP3.")

        except Exception as e_stopped_save:
            print(f"Критическая ошибка в потоке сохранения остановленного: {e_stopped_save}")
            import traceback
            traceback.print_exc()
            self.root.after(0, messagebox.showerror, "Ошибка", f"Непредвиденная ошибка при сохранении остановленного аудио:\n{e_stopped_save}")
            self.root.after(0, self.update_status, "Ошибка при сохранении остановленного MP3.")
        finally:
            # Re-enable controls regardless of success/failure
            self.root.after(0, self.enable_controls, False)
            # Decide on cleanup: If saving failed, maybe keep files? If succeeded, maybe clean?
            # Current logic keeps files until next playback starts or app closes, unless cleaned above.
            # Let's ensure the save button is disabled if save succeeded
            if success:
                self.root.after(0, lambda: setattr(self, 'playback_was_stopped', False)) # Reset flag in GUI thread
                self.root.after(0, self.cleanup_temp_files) # Clean up after successful save


    # --- Application Lifecycle ---

    def on_closing(self):
        """Handles the application closing event."""
        print("Запрос на закрытие приложения...")
        # 1. Stop any active playback thread and sound output
        self.stop_audio()

        # 2. Wait briefly for the thread to potentially acknowledge the stop signal
        #    (Not strictly necessary if sd.stop() is effective, but can be safer)
        if self.playing_thread and self.playing_thread.is_alive():
            print("Ожидание завершения потока озвучки (макс 1 сек)...")
            self.playing_thread.join(timeout=1.0)
            if self.playing_thread.is_alive():
                 print("Поток озвучки не завершился вовремя.")

        # 3. Clean up all temporary files and directories unconditionally
        print("Финальная очистка временных файлов...")
        # Ensure we clear the list and the tracked directory before cleanup call
        files_to_clean = list(self.temp_audio_files)
        dir_to_clean = self.current_temp_dir
        self.temp_audio_files.clear()
        self.current_temp_dir = None
        # Call cleanup for files
        for f_path in files_to_clean:
             try:
                 if f_path.exists(): f_path.unlink()
             except OSError as e: print(f"Ошибка при удалении файла {f_path}: {e}")
        # Call cleanup for directory
        if dir_to_clean and dir_to_clean.exists():
             try:
                 shutil.rmtree(dir_to_clean, ignore_errors=True)
                 print(f"Удалена временная папка: {dir_to_clean}")
             except Exception as e: print(f"Ошибка при удалении папки {dir_to_clean}: {e}")


        # 4. Close the PDF document
        if self.pdf_document:
            print("Закрытие PDF документа...")
            try:
                self.pdf_document.close()
            except Exception as e:
                print(f"Ошибка при закрытии PDF: {e}")
            self.pdf_document = None

        # 5. Destroy the main window
        print("Уничтожение окна Tkinter...")
        self.root.destroy()
        print("Приложение закрыто.")


    def update_status(self, message: str):
        """Safely updates the status bar label from any thread."""
        if self.status_label and self.status_label.winfo_exists():
            # Use self.root.after to schedule the update on the main GUI thread
            def _update():
                if self.status_label and self.status_label.winfo_exists():
                    self.status_label.config(text=message)
            # Check if root exists before scheduling
            if self.root and self.root.winfo_exists():
                 self.root.after(0, _update)
        # Always print the status message to the console for logging
        print(f"Status: {message}")


# --- Helper Functions (Outside Class) ---

def check_ffmpeg() -> bool:
    """Checks if the ffmpeg command is available in the system PATH."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        print(f"FFmpeg найден: {ffmpeg_path}")
        return True
    else:
        print("ОШИБКА: FFmpeg не найден в системном PATH.")
        # Show error message immediately if GUI is likely available
        try:
            # Need a temporary root to show messagebox if main loop hasn't started
            root_temp = tk.Tk()
            root_temp.withdraw() # Hide the temp window
            messagebox.showerror(
                "FFmpeg не найден",
                "FFmpeg не установлен или не добавлен в PATH.\n"
                "Сохранение аудио в MP3 будет недоступно.\n\n"
                "Пожалуйста, установите FFmpeg с официального сайта (ffmpeg.org) "
                "и убедитесь, что путь к нему добавлен в переменную окружения PATH."
            )
            root_temp.destroy()
        except tk.TclError:
             print("Не удалось показать messagebox для ошибки FFmpeg (GUI не готов?).")
        return False

# --- Main Execution ---

if __name__ == "__main__":
    # Optional: Check for other heavy dependencies like PyTorch early
    print("Проверка зависимостей...")
    try:
        import PyYAML
        import omegaconf
        # These are often needed by Silero models, check might prevent cryptic errors later
        print("PyYAML и omegaconf найдены.")
    except ImportError:
        print("Предупреждение: PyYAML и/или omegaconf не найдены.")
        print("Рекомендуется установить: pip install PyYAML omegaconf")

    # Check for FFmpeg *before* creating the main window
    ffmpeg_available = check_ffmpeg()
    if not ffmpeg_available:
        print("Продолжение работы без функции сохранения MP3.")

    # Create the main application window and instance
    print("Запуск приложения...")
    main_root = tk.Tk()
    app = PdfReaderApp(main_root)

    # Set the close protocol handler
    main_root.protocol("WM_DELETE_WINDOW", app.on_closing)

    # Start the Tkinter event loop
    try:
         main_root.mainloop()
    except KeyboardInterrupt:
         print("\nПриложение прервано пользователем (Ctrl+C).")
         # Attempt graceful shutdown if interrupted
         app.on_closing()