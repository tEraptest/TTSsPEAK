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
import ffmpeg
import tempfile
import subprocess
import scipy.io.wavfile
import shutil

LANGUAGE = 'ru'
MODEL_ID = 'v4_ru'
VOICES = ["kseniya", "xenia", "oleg", "zahar", "julia", "dmitry"]
SAMPLE_RATE = 48000
DEVICE = torch.device('cpu')

tts_model = None
tts_ready = False

def load_tts_model():
    global tts_model, tts_ready
    print("Загрузка модели Silero TTS...")
    try:
        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                  model='silero_tts',
                                  language=LANGUAGE,
                                  speaker=MODEL_ID,
                                  trust_repo=True)
        model.to(DEVICE)
        tts_model = model
        tts_ready = True
        print(f"Модель Silero TTS ({MODEL_ID}, {VOICES[0]}) загружена успешно.")
        return True
    except Exception as e:
        print(f"Критическая ошибка загрузки Silero TTS: {e}")
        if tk._default_root and tk._default_root.winfo_exists():
             try:
                 messagebox.showerror("Ошибка TTS", f"Не удалось загрузить модель Silero TTS.\nОшибка: {e}")
             except tk.TclError:
                 print("Не удалось показать messagebox, возможно, root уже уничтожен.")
        return False

def reduce_noise(audio_np, noise_threshold=0.02):
    """Reduces background noise by applying a noise gate."""
    return np.where(np.abs(audio_np) > noise_threshold, audio_np, 0)

def normalize_audio(audio_np, target_peak=0.9):
    """Normalizes audio and applies a noise gate."""
    audio_np = reduce_noise(audio_np)  # Apply noise reduction
    peak = np.max(np.abs(audio_np))
    if peak > 0:
        factor = target_peak / peak
        return audio_np * factor
    return audio_np

def generate_audio_chunk(text_to_speak, speed_multiplier=1.0, voice=VOICES[0]):
    if not tts_ready or not tts_model or not text_to_speak:
        print("TTS не готов, модель не загружена или текст пустой.")
        return None

    try:
        cleaned_text = ' '.join(text_to_speak.split())
        if len(cleaned_text) > 1000:
            print("Текст слишком длинный, разбиваем на части.")
            return None

        print(f"Генерация аудио для: '{cleaned_text[:60]}...' с скоростью {speed_multiplier:.1f}x и голосом '{voice}'")

        audio = tts_model.apply_tts(text=cleaned_text,
                                    speaker=voice,
                                    sample_rate=SAMPLE_RATE,
                                    put_accent=True,
                                    put_yo=True)

        if isinstance(audio, torch.Tensor):
            audio_np = audio.numpy()
        else:
            audio_np = np.array(audio)

        # Apply normalization and noise reduction
        audio_np = normalize_audio(audio_np)

        if speed_multiplier != 1.0:
             print(f"Применение скорости {speed_multiplier:.1f}x к аудио")
             indices = np.arange(0, len(audio_np), speed_multiplier)
             resampled_audio_np = np.interp(indices, np.arange(len(audio_np)), audio_np)
             # Нормализуем полученный сигнал после изменения скорости
             return normalize_audio(resampled_audio_np)
        else:
             return audio_np

    except Exception as e:
        print(f"Ошибка во время генерации TTS: {e}")
        return None


def play_audio_chunk(audio_np):
    if audio_np is None or audio_np.size == 0:
        print("Нет аудио данных для воспроизведения.")
        return False
    try:
        print("Воспроизведение...")
        sd.play(audio_np, SAMPLE_RATE)
        sd.wait()
        print("Воспроизведение фрагмента завершено.")
        return True
    except Exception as e:
        print(f"Ошибка воспроизведения аудио: {e}")
        sd.stop()
        return False


def save_audio_to_file(file_path, audio_np):
    try:
        scipy.io.wavfile.write(file_path, SAMPLE_RATE, (audio_np * 32767).astype(np.int16))
        print(f"Аудио сохранено в файл: {file_path}")
    except Exception as e:
        print(f"Ошибка записи WAV файла {file_path}: {e}")


def split_text_into_chunks(text, max_length=450):
    chunks = []
    current_pos = 0
    while current_pos < len(text):
        end_pos = min(current_pos + max_length, len(text))
        split_index = -1
        for sep in ['.', '?', '!']:
            found_index = text.rfind(sep, current_pos, end_pos)
            if found_index != -1:
                split_index = max(split_index, found_index)

        if split_index != -1 and end_pos < len(text):
            chunk = text[current_pos:split_index + 1]
            current_pos = split_index + 1
        elif end_pos < len(text):
            space_index = text.rfind(' ', current_pos, end_pos)
            if (space_index != -1 and space_index > current_pos):
                chunk = text[current_pos:space_index]
                current_pos = space_index + 1
            else:
                chunk = text[current_pos:end_pos]
                current_pos = end_pos
        else:
            chunk = text[current_pos:]
            current_pos = len(text)

        cleaned_chunk = chunk.strip()
        if cleaned_chunk:
            chunks.append(cleaned_chunk)

    return chunks


def generate_full_mp3(pdf_document, output_path, start_page_idx, end_page_idx, voice=VOICES[0]):
    print(f"Начало полной генерации MP3 для страниц {start_page_idx+1}-{end_page_idx+1} с голосом '{voice}'")
    if not tts_ready or not tts_model or not pdf_document:
        print("TTS не готов или PDF документ не загружен.")
        return False

    temp_dir = tempfile.mkdtemp(prefix="pdfspeaker_")
    print(f"Временная папка для аудио: {temp_dir}")
    temp_wav_files = []
    success_flag = True

    try:
        for i, page_num in enumerate(range(start_page_idx, end_page_idx + 1)):
            try:
                page = pdf_document.load_page(page_num)
                full_page_text = page.get_text("text", sort=True).strip()
            except Exception as e:
                print(f"Ошибка чтения страницы {page_num + 1}: {e}. Пропуск.")
                continue

            if not full_page_text:
                print(f"Страница {page_num + 1} пуста, пропуск.")
                continue

            print(f"Генерация аудио для страницы {page_num + 1}...")
            text_chunks = split_text_into_chunks(full_page_text)

            for j, chunk in enumerate(text_chunks):
                audio_np = generate_audio_chunk(chunk, speed_multiplier=1.0, voice=voice)

                if audio_np is not None and audio_np.size > 0:
                    temp_wav_path = os.path.join(temp_dir, f"audio_{i:04d}_{j:04d}.wav")
                    try:
                        save_audio_to_file(temp_wav_path, audio_np)
                        temp_wav_files.append(temp_wav_path)
                    except Exception as write_e:
                        print(f"Ошибка записи временного WAV файла {temp_wav_path}: {write_e}")
                        success_flag = False
                        break
                elif audio_np is None:
                    print(f"Ошибка генерации TTS для фрагмента на стр. {page_num + 1}. Сохранение будет неполным.")
                    success_flag = False
                    pass
            if not success_flag:
                break

        if not temp_wav_files:
            print("Нет аудио для сохранения.")
            messagebox.showerror("Ошибка сохранения", "Не удалось сгенерировать аудио для выбранного диапазона.")
            return False

        print(f"Объединение {len(temp_wav_files)} аудио фрагментов...")
        list_file_path = os.path.join(temp_dir, "ffmpeg_list.txt")
        with open(list_file_path, 'w', encoding='utf-8') as f:
            for wav_file in temp_wav_files:
                abs_path = os.path.abspath(wav_file).replace('\\', '/')
                f.write(f"file '{abs_path}'\n")

        try:
            command = [
                'ffmpeg', '-f', 'concat', '-safe', '0', '-i', list_file_path,
                '-codec:a', 'libmp3lame', '-b:a', '192k', '-y', output_path
            ]
            print(f"Запуск FFmpeg: {' '.join(command)}")
            process = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
            print("FFmpeg stdout:", process.stdout)
            print("FFmpeg stderr:", process.stderr)
            print(f"Файл MP3 сохранён: {output_path}")
            return success_flag

        except FileNotFoundError:
            print("ОШИБКА: FFmpeg не найден. Убедитесь, что ffmpeg установлен и прописан в системном PATH.")
            messagebox.showerror("Ошибка FFmpeg", "FFmpeg не найден. Установите ffmpeg и добавьте его в PATH.")
            return False
        except subprocess.CalledProcessError as e:
            print(f"Ошибка выполнения FFmpeg:")
            print("Команда:", e.cmd)
            print("Код возврата:", e.returncode)
            print("Stdout:", e.stdout)
            print("Stderr:", e.stderr)
            messagebox.showerror("Ошибка FFmpeg", f"Ошибка при конвертации аудио:\n{e.stderr[:500]}...")
            return False
        except Exception as e_ffmpeg:
            print(f"Неизвестная ошибка при работе с FFmpeg: {e_ffmpeg}")
            messagebox.showerror("Ошибка FFmpeg", f"Неизвестная ошибка при конвертации:\n{e_ffmpeg}")
            return False

    finally:
        print(f"Очистка временных файлов из {temp_dir}...")
        if 'list_file_path' in locals() and os.path.exists(list_file_path):
            try:
                os.remove(list_file_path)
            except Exception as e_clean_list:
                print(f"Не удалось удалить файл списка {list_file_path}: {e_clean_list}")

        for f_path in temp_wav_files:
            try:
                if os.path.exists(f_path):
                    os.remove(f_path)
            except Exception as e_clean:
                print(f"Не удалось удалить временный файл {f_path}: {e_clean}")
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            print("Временные файлы очищены.")
        except Exception as e_rmdir:
            print(f"Не удалось удалить временную папку {temp_dir}: {e_rmdir}")


class PdfReaderApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("PDF Reader & Speaker (Русский TTS)")
        self.root.geometry("1000x750")

        self.pdf_document = None
        self.pdf_path = ""
        self.current_page_num = 0
        self.total_pages = 0
        self.playing_thread = None
        self.stop_playback_flag = threading.Event()
        self.speech_speed = 1.0
        self.temp_audio_files = []
        self.playback_was_stopped = False

        self._create_widgets()

        threading.Thread(target=self._initial_tts_load, daemon=True).start()

    def _initial_tts_load(self):
        if load_tts_model():
            self.root.after(0, self.enable_controls)
        else:
            self.root.after(0, self.update_status, "Ошибка загрузки TTS модели. Озвучивание недоступно.")

    def _create_widgets(self):
        top_frame = ttk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        control_frame = ttk.LabelFrame(top_frame, text="Управление", padding="10")
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

        self.btn_save_mp3 = ttk.Button(control_frame, text="💾 Сохранить MP3 (полное)", command=self.save_full_audio_to_mp3, state=tk.DISABLED)
        self.btn_save_mp3.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        self.btn_save_stopped_mp3 = ttk.Button(control_frame, text="💾 Сохранить MP3 (остановленное)", command=self.save_stopped_audio_to_mp3, state=tk.DISABLED)
        self.btn_save_stopped_mp3.grid(row=1, column=3, columnspan=4, padx=5, pady=5, sticky="ew")

        nav_speed_frame = ttk.LabelFrame(top_frame, text="Навигация и скорость", padding="10")
        nav_speed_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        self.btn_prev = ttk.Button(nav_speed_frame, text="< Пред.", command=self.prev_page, state=tk.DISABLED)
        self.btn_prev.grid(row=0, column=0, padx=5, pady=5)

        self.lbl_page_display = ttk.Label(nav_speed_frame, text="Страница: - / -", width=15, anchor="center")
        self.lbl_page_display.grid(row=0, column=1, padx=5, pady=5)

        self.btn_next = ttk.Button(nav_speed_frame, text="След. >", command=self.next_page, state=tk.DISABLED)
        self.btn_next.grid(row=0, column=2, padx=5, pady=5)

        lbl_speed = ttk.Label(nav_speed_frame, text="Скорость:")
        lbl_speed.grid(row=0, column=3, padx=(20, 0), pady=5)

        self.speed_slider = ttk.Scale(nav_speed_frame, from_=0.5, to=2.0, length=150, value=1.0, orient=tk.HORIZONTAL, command=self.update_speed)
        self.speed_slider.grid(row=0, column=4, padx=5, pady=5)

        self.speed_label = ttk.Label(nav_speed_frame, text="1.0x", width=5)
        self.speed_label.grid(row=0, column=5, padx=(5, 0), pady=5)

        lbl_voice = ttk.Label(nav_speed_frame, text="Голос:")
        lbl_voice.grid(row=0, column=6, padx=(20, 0), pady=5)
        self.voice_combobox = ttk.Combobox(nav_speed_frame, values=VOICES, state="readonly", width=10)
        self.voice_combobox.grid(row=0, column=7, padx=5, pady=5)
        self.voice_combobox.set(VOICES[0])

        display_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        display_pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        pdf_frame = ttk.Frame(display_pane, width=500, height=600)
        self.pdf_image_label = ttk.Label(pdf_frame, anchor=tk.CENTER)
        self.pdf_image_label.pack(fill=tk.BOTH, expand=True)
        display_pane.add(pdf_frame, weight=1)

        text_frame = ttk.Frame(display_pane, width=400, height=600)
        self.text_display = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, state=tk.DISABLED, height=20)
        self.text_display.pack(fill=tk.BOTH, expand=True)
        display_pane.add(text_frame, weight=1)

        self.status_label = ttk.Label(self.root, text="Загрузите PDF файл. Идет загрузка TTS модели...", relief=tk.SUNKEN, anchor=tk.W, padding="2 5")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def update_speed(self, value):
        try:
            self.speech_speed = float(value)
            self.speed_label.config(text=f"{self.speech_speed:.1f}x")
        except ValueError:
            pass

    def enable_controls(self, playback_running=False):
        base_enabled = tts_ready and self.pdf_document
        normal_state = tk.NORMAL if base_enabled else tk.DISABLED
        disabled_state = tk.DISABLED

        self.btn_play.config(state=disabled_state if playback_running else normal_state)
        self.btn_stop.config(state=normal_state if playback_running else disabled_state)

        self.entry_start_page.config(state=disabled_state if playback_running else normal_state)
        self.entry_end_page.config(state=disabled_state if playback_running else normal_state)
        self.btn_prev.config(state=disabled_state if playback_running else normal_state)
        self.btn_next.config(state=disabled_state if playback_running else normal_state)
        self.btn_save_mp3.config(state=disabled_state if playback_running else normal_state)
        self.btn_load.config(state=disabled_state if playback_running else tk.NORMAL)

        self.text_display.config(state=tk.NORMAL if self.pdf_document else tk.DISABLED)

        can_save_stopped = base_enabled and not playback_running and self.playback_was_stopped and self.temp_audio_files
        self.btn_save_stopped_mp3.config(state=tk.NORMAL if can_save_stopped else tk.DISABLED)

    def select_pdf(self):
        filepath = filedialog.askopenfilename(
            title="Выберите PDF файл",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")]
        )
        if not filepath:
            return

        self.stop_audio()
        self.cleanup_temp_files()

        try:
            if self.pdf_document:
                self.pdf_document.close()
            self.pdf_document = fitz.open(filepath)
            self.pdf_path = filepath
            self.total_pages = self.pdf_document.page_count
            self.current_page_num = 0

            self.update_status(f"Загружен: {os.path.basename(filepath)}, Страниц: {self.total_pages}")

            self.entry_start_page.config(state=tk.NORMAL)
            self.entry_end_page.config(state=tk.NORMAL)
            self.entry_start_page.delete(0, tk.END)
            self.entry_start_page.insert(0, "1")
            self.entry_end_page.delete(0, tk.END)
            self.entry_end_page.insert(0, str(self.total_pages))

            self.show_page(self.current_page_num)
            self.enable_controls()

        except Exception as e:
            self.pdf_document = None
            self.pdf_path = ""
            self.total_pages = 0
            messagebox.showerror("Ошибка загрузки PDF", f"Не удалось открыть файл:\n{filepath}\n\nОшибка: {e}")
            self.update_status("Ошибка загрузки PDF.")
            self.clear_display()
            self.enable_controls()

    def clear_display(self):
        self.pdf_image_label.config(image='')
        self.pdf_image_label.image = None
        self.text_display.config(state=tk.NORMAL)
        self.text_display.delete('1.0', tk.END)
        self.text_display.config(state=tk.DISABLED)
        self.lbl_page_display.config(text="Страница: - / -")

    def show_page(self, page_index):
        if not self.pdf_document or not (0 <= page_index < self.total_pages):
            return
        self.current_page_num = page_index
        self.update_page_label()
        try:
            page = self.pdf_document.load_page(page_index)
            zoom = 1.5
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_tk = ImageTk.PhotoImage(img)
            self.pdf_image_label.config(image=img_tk)
            self.pdf_image_label.image = img_tk
        except Exception as e:
            print(f"Ошибка рендеринга страницы {page_index + 1}: {e}")
            self.pdf_image_label.config(image='')
            self.pdf_image_label.image = None
            self.update_status(f"Ошибка отображения страницы {page_index + 1}")
        try:
            text = page.get_text("text", sort=True)
            self.text_display.config(state=tk.NORMAL)
            self.text_display.delete('1.0', tk.END)
            self.text_display.insert('1.0', text)
            self.text_display.config(state=tk.DISABLED)
        except Exception as e:
            print(f"Ошибка извлечения текста со страницы {page_index + 1}: {e}")
            self.text_display.config(state=tk.NORMAL)
            self.text_display.delete('1.0', tk.END)
            self.text_display.insert('1.0', f"Ошибка текста стр. {page_index + 1}")
            self.text_display.config(state=tk.DISABLED)

    def update_page_label(self):
        txt = f"Страница: {self.current_page_num + 1} / {self.total_pages}" if self.pdf_document else "Страница: - / -"
        self.lbl_page_display.config(text=txt)

    def prev_page(self):
        if self.pdf_document and self.current_page_num > 0:
            self.show_page(self.current_page_num - 1)

    def next_page(self):
        if self.pdf_document and self.current_page_num < self.total_pages - 1:
            self.show_page(self.current_page_num + 1)

    def play_range(self):
        if not self.pdf_document:
            messagebox.showwarning("Нет PDF", "Сначала загрузите PDF.")
            return
        if not tts_ready:
            messagebox.showerror("TTS не готов", "Модель озвучивания не готова.")
            return
        if self.playing_thread and self.playing_thread.is_alive():
            messagebox.showinfo("Занято", "Воспроизведение уже идет.")
            return

        try:
            start_page = int(self.entry_start_page.get()) - 1
            end_page = int(self.entry_end_page.get()) - 1
            if not (0 <= start_page < self.total_pages and 0 <= end_page < self.total_pages and start_page <= end_page):
                raise ValueError("Некорректный диапазон")
        except ValueError:
            messagebox.showerror("Ошибка диапазона", f"Некорректный диапазон страниц (от 1 до {self.total_pages}).")
            return

        self.stop_playback_flag.clear()
        self.playback_was_stopped = False
        self.cleanup_temp_files()
        self.temp_audio_files = []

        selected_voice = self.voice_combobox.get()
        thread_args = (start_page, end_page, selected_voice)
        self.playing_thread = threading.Thread(target=self._play_audio_thread, args=thread_args, daemon=True)

        self.enable_controls(playback_running=True)
        self.playing_thread.start()

    def _play_audio_thread(self, start_page_idx, end_page_idx, voice):
        print(f"Запуск озвучки со стр {start_page_idx + 1} по {end_page_idx + 1} с голосом '{voice}'")
        temp_dir = tempfile.mkdtemp(prefix="pdfspeaker_play_")
        print(f"Временная папка для воспроизведения: {temp_dir}")
        page_processed_successfully = True

        try:
            for page_num in range(start_page_idx, end_page_idx + 1):
                if self.stop_playback_flag.is_set():
                    print("Остановка (начало цикла стр).")
                    page_processed_successfully = False
                    break
                if not self.pdf_document:
                    print("PDF закрыт (начало цикла стр).")
                    page_processed_successfully = False
                    break

                self.root.after(0, lambda p=page_num: self.show_page(p))
                self.root.after(0, self.update_status, f"Обработка страницы {page_num + 1}...")
                time.sleep(0.1)
                if self.stop_playback_flag.is_set():
                    print("Остановка (после показа стр).")
                    page_processed_successfully = False
                    break

                try:
                    page = self.pdf_document.load_page(page_num)
                    full_page_text = page.get_text("text", sort=True)
                except Exception as e:
                    print(f"Ошибка чтения текста стр {page_num + 1}: {e}")
                    time.sleep(1)
                    continue

                if not full_page_text or full_page_text.isspace():
                    self.root.after(0, self.update_status, f"Стр. {page_num + 1}: нет текста.")
                    time.sleep(0.5)
                    continue

                text_chunks = split_text_into_chunks(full_page_text)

                for i, chunk in enumerate(text_chunks):
                    if self.stop_playback_flag.is_set():
                        print("Остановка (начало цикла фрагментов).")
                        page_processed_successfully = False
                        break
                    if not self.pdf_document:
                        print("PDF закрыт (начало цикла фрагментов).")
                        page_processed_successfully = False
                        break

                    self.root.after(0, self.update_status, f"Озвучивание стр. {page_num + 1} (часть {i+1}/{len(text_chunks)})...")
                    self.root.after(0, lambda ft=full_page_text, c=chunk: self.highlight_text(ft, c))

                    audio_np = generate_audio_chunk(chunk, speed_multiplier=self.speech_speed, voice=voice)

                    if audio_np is not None and audio_np.size > 0:
                        temp_wav_path = os.path.join(temp_dir, f"chunk_{page_num:04d}_{i:04d}.wav")
                        try:
                            save_audio_to_file(temp_wav_path, audio_np)
                            self.temp_audio_files.append(temp_wav_path)
                            print(f"Временный файл сохранен: {temp_wav_path}")
                        except Exception as write_e:
                            print(f"Ошибка записи временного WAV файла {temp_wav_path}: {write_e}")

                        play_success = play_audio_chunk(audio_np)
                        if not play_success:
                            self.root.after(0, self.update_status, f"Ошибка воспроизведения на стр. {page_num + 1}")

                    elif audio_np is None:
                        self.root.after(0, self.update_status, f"Ошибка TTS на стр. {page_num + 1}")
                        self.stop_playback_flag.set()
                        page_processed_successfully = False
                        break

                    if self.stop_playback_flag.is_set():
                        print("Остановка (после фрагмента).")
                        page_processed_successfully = False
                        break

                if not page_processed_successfully:
                    break

        finally:
            print("Блок finally потока воспроизведения достигнут.")
            self.root.after(0, self.on_playback_finished, page_processed_successfully)

    def stop_audio(self):
        if self.playing_thread and self.playing_thread.is_alive():
            print("Сигнал остановки потоку воспроизведения...")
            self.playback_was_stopped = True
            self.stop_playback_flag.set()
            sd.stop()
        else:
            print("Нет активного потока воспроизведения для остановки.")

    def on_playback_finished(self, completed_normally):
        print(f"on_playback_finished вызван. Завершено нормально: {completed_normally}, Остановка кнопкой: {self.playback_was_stopped}")

        if completed_normally and not self.playback_was_stopped:
            self.update_status("Озвучивание диапазона завершено.")
            self.cleanup_temp_files()
            self.playback_was_stopped = False
        elif self.playback_was_stopped:
            self.update_status("Воспроизведение остановлено. Можно сохранить.")
        else:
            current_status = self.status_label.cget("text")
            if "Озвучивание стр" in current_status or "Обработка страницы" in current_status:
                self.update_status("Воспроизведение прервано из-за ошибки.")
            self.cleanup_temp_files()
            self.playback_was_stopped = False

        self.playing_thread = None

        self.enable_controls(playback_running=False)

    def save_full_audio_to_mp3(self):
        if not self.pdf_document:
            messagebox.showwarning("Нет PDF", "Загрузите PDF.")
            return
        if not tts_ready:
            messagebox.showerror("TTS не готов", "Модель не загружена.")
            return

        try:
            start_page = int(self.entry_start_page.get()) - 1
            end_page = int(self.entry_end_page.get()) - 1
            if not (0 <= start_page < self.total_pages and 0 <= end_page < self.total_pages and start_page <= end_page):
                raise ValueError("Некорректный диапазон")
        except ValueError:
            messagebox.showerror("Ошибка диапазона", f"Некорректный диапазон страниц (от 1 до {self.total_pages}).")
            return

        output_path = filedialog.asksaveasfilename(
            title="Сохранить ПОЛНЫЙ MP3 файл",
            defaultextension=".mp3",
            filetypes=[("MP3 Files", "*.mp3"), ("All Files", "*.*")]
        )
        if not output_path:
            return

        self.update_status("Идет полная генерация MP3... Это может занять время.")
        self.root.update_idletasks()

        save_thread = threading.Thread(target=self._run_full_save_mp3, args=(output_path, start_page, end_page), daemon=True)
        save_thread.start()

    def _run_full_save_mp3(self, output_path, start_page, end_page):
        try:
            doc_to_use = fitz.open(self.pdf_path) if self.pdf_path else self.pdf_document
            if not doc_to_use:
                raise ValueError("Не удалось получить доступ к PDF документу для сохранения.")

            selected_voice = self.voice_combobox.get()
            success = generate_full_mp3(doc_to_use, output_path, start_page, end_page, voice=selected_voice)
            if doc_to_use != self.pdf_document:
                doc_to_use.close()

            if success:
                self.root.after(0, messagebox.showinfo, "Успех", f"Файл MP3 успешно сохранён:\n{output_path}")
                self.root.after(0, self.update_status, "MP3 файл сохранён.")
            else:
                self.root.after(0, self.update_status, "Ошибка при сохранении MP3.")
        except Exception as e_save:
            print(f"Критическая ошибка в потоке сохранения: {e_save}")
            self.root.after(0, messagebox.showerror, "Ошибка", f"Ошибка в процессе сохранения:\n{e_save}")
            self.root.after(0, self.update_status, "Ошибка при сохранении MP3.")

    def save_stopped_audio_to_mp3(self):
        if not self.temp_audio_files:
            messagebox.showwarning("Нечего сохранять", "Нет данных для сохранения после остановки.")
            return

        output_path = filedialog.asksaveasfilename(
            title="Сохранить ОСТАНОВЛЕННЫЙ MP3 файл",
            defaultextension=".mp3",
            filetypes=[("MP3 Files", "*.mp3"), ("All Files", "*.*")]
        )
        if not output_path:
            return

        self.update_status("Идет объединение аудио файлов...")
        self.root.update_idletasks()

        files_to_save = list(self.temp_audio_files)
        temp_dir = os.path.dirname(files_to_save[0]) if files_to_save else None

        save_thread = threading.Thread(target=self._run_stopped_save_mp3, args=(output_path, files_to_save, temp_dir), daemon=True)
        save_thread.start()

    def _run_stopped_save_mp3(self, output_path, files_to_concat, temp_dir_path):
        list_file_path = None
        concatenated_ok = False
        try:
            print(f"Начало сохранения остановленного аудио ({len(files_to_concat)} фрагментов)")
            list_file_path = os.path.join(temp_dir_path or ".", "ffmpeg_stopped_list.txt")
            with open(list_file_path, 'w', encoding='utf-8') as f:
                for wav_file in files_to_concat:
                    if os.path.exists(wav_file):
                        abs_path = os.path.abspath(wav_file).replace('\\', '/')
                        f.write(f"file '{abs_path}'\n")

            if not files_to_concat:
                self.root.after(0, self.update_status, "Нет данных для сохранения.")
                return

            command = [
                'ffmpeg', '-f', 'concat', '-safe', '0', '-i', list_file_path,
                '-codec:a', 'libmp3lame', '-b:a', '192k', '-y', output_path
            ]
            print(f"Запуск FFmpeg для остановленного: {' '.join(command)}")
            process = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
            print("FFmpeg stdout (stopped):", process.stdout)
            print("FFmpeg stderr (stopped):", process.stderr)
            print(f"Остановленный файл MP3 сохранён: {output_path}")
            concatenated_ok = True
            self.root.after(0, messagebox.showinfo, "Успех", f"Остановленное аудио сохранено:\n{output_path}")
            self.root.after(0, self.update_status, "Остановленный MP3 сохранён.")

        except FileNotFoundError:
            print("ОШИБКА: FFmpeg не найден (при сохранении остановленного).")
            self.root.after(0, messagebox.showerror, "Ошибка FFmpeg", "FFmpeg не найден. Установите ffmpeg.")
            self.root.after(0, self.update_status, "Ошибка FFmpeg при сохранении.")
        except subprocess.CalledProcessError as e:
            print(f"Ошибка FFmpeg (остановленное): {e.stderr}")
            self.root.after(0, messagebox.showerror, "Ошибка FFmpeg", f"Ошибка при объединении аудио:\n{e.stderr[:500]}...")
            self.root.after(0, self.update_status, "Ошибка FFmpeg при сохранении.")
        except Exception as e_stopped_save:
            print(f"Критическая ошибка в потоке сохранения остановленного: {e_stopped_save}")
            self.root.after(0, messagebox.showerror, "Ошибка", f"Ошибка в процессе сохранения:\n{e_stopped_save}")
            self.root.after(0, self.update_status, "Ошибка при сохранении MP3.")
        finally:
            self.root.after(0, self.cleanup_temp_files, concatenated_ok, temp_dir_path)
            if list_file_path and os.path.exists(list_file_path):
                try:
                    os.remove(list_file_path)
                except Exception as e_clean_list:
                    print(f"Не удалось удалить файл списка {list_file_path}: {e_clean_list}")

    def cleanup_temp_files(self, keep_parent_dir=False, parent_dir_path=None):
        print(f"Начало очистки {len(self.temp_audio_files)} временных файлов...")
        cleaned_count = 0
        dirs_to_check = set()

        files_to_remove = list(self.temp_audio_files)
        if not keep_parent_dir:  # Очищаем основной список только если не сохраняем
            self.temp_audio_files.clear()

        for f_path in files_to_remove:
            try:
                if os.path.exists(f_path):
                    dir_name = os.path.dirname(f_path)
                    if dir_name:
                        dirs_to_check.add(dir_name)
                    os.remove(f_path)
                    cleaned_count += 1
                else:
                    print(f"Файл для очистки не найден: {f_path}")
            except Exception as e_clean:
                print(f"Не удалось удалить временный файл {f_path}: {e_clean}")

        if not keep_parent_dir:
            for dir_path in dirs_to_check:
                # Не удаляем папку, если она совпадает с parent_dir_path (на случай, если она передана)
                if parent_dir_path and os.path.abspath(dir_path) == os.path.abspath(parent_dir_path):
                    continue
                try:
                    if os.path.exists(dir_path) and not os.listdir(dir_path):
                        os.rmdir(dir_path)
                        print(f"Удалена пустая временная папка: {dir_path}")
                    elif os.path.exists(dir_path):
                        print(f"Временная папка не пуста, не удалена: {dir_path}")

                except OSError as e_rmdir:
                    print(f"Не удалось удалить временную папку {dir_path}: {e_rmdir}")

        print(f"Очистка завершена. Удалено файлов: {cleaned_count}.")
        # Обновляем состояние кнопки после очистки
        self.root.after(0, self.enable_controls, False)

    def highlight_text(self, full_text, chunk):
        try:
            if not self.text_display.winfo_exists():
                return
            self.text_display.config(state=tk.NORMAL)
            current_text = self.text_display.get("1.0", tk.END).strip()
            if current_text != full_text.strip():
                self.text_display.delete('1.0', tk.END)
                self.text_display.insert('1.0', full_text)

            start_idx = full_text.find(chunk)
            self.text_display.tag_remove("highlight", "1.0", tk.END)

            if start_idx != -1:
                end_idx = start_idx + len(chunk)
                start_tk_idx = f"1.0 + {start_idx} chars"
                end_tk_idx = f"1.0 + {end_idx} chars"
                self.text_display.tag_add("highlight", start_tk_idx, end_tk_idx)
                self.text_display.tag_configure("highlight", background="yellow", foreground="black")
                self.text_display.see(start_tk_idx)

            self.text_display.config(state=tk.DISABLED)
        except tk.TclError as e:
            print(f"Ошибка Tkinter при подсветке: {e}")
        except Exception as e_highlight:
            print(f"Неизвестная ошибка при подсветке: {e_highlight}")

    def on_closing(self):
        print("Запрос на закрытие приложения...")
        self.stop_audio()
        # При закрытии удаляем все временные файлы
        self.cleanup_temp_files(keep_parent_dir=False)
        if self.pdf_document:
            try:
                self.pdf_document.close()
            except:
                pass
        self.root.destroy()

    def update_status(self, message):
        if self.status_label and self.status_label.winfo_exists():
            # Запускаем обновление статус бара через self.root.after для потокобезопасности
            def _update():
                if self.status_label.winfo_exists():
                    self.status_label.config(text=message)
            if self.root and self.root.winfo_exists():
                self.root.after(0, _update)
        print(message)


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        messagebox.showerror(
            "FFmpeg не найден",
            "FFmpeg не установлен или не добавлен в PATH.\n"
            "Установите FFmpeg и добавьте его в PATH, чтобы сохранение работало.\n"
            "Скачать можно с https://ffmpeg.org/download.html"
        )
        return False
    return True


if __name__ == "__main__":
    try:
        import PyYAML
        import omegaconf
    except ImportError:
        print("Предупреждение: PyYAML и/или omegaconf не найдены. Установите: pip install PyYAML omegaconf")

    if not check_ffmpeg():
        print("FFmpeg не найден. Сохранение файлов будет недоступно.")
        # Не выходим, позволяем программе запуститься без возможности сохранения

    main_root = tk.Tk()
    app = PdfReaderApp(main_root)
    main_root.protocol("WM_DELETE_WINDOW", app.on_closing)
    main_root.mainloop()

try:
    if some_condition:  # Replace with actual condition
        pass  # Replace with actual logic
    else:
        pass  # Replace with actual logic
except Exception as e:
    print(f"An error occurred: {e}")

import threading
import tempfile
import fitz  # PyMuPDF

start_page = 1  # Example value, replace with actual logic
end_page = 10  # Example value, replace with actual logic
tts_ready = False  # Example value, replace with actual logic
tts_model = None  # Example value, replace with actual logic

def split_text_into_chunks(text, chunk_size):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def generate_audio_chunk(text_chunk):
    pass

def generate_full_mp3():
    pass

try:
    pass  # Replace with actual logic
except Exception as e:
    print(f"An error occurred: {e}")

if some_condition:  # Replace with actual condition
    pass  # Replace with actual logic
else:
    pass  # Replace with actual logic

if another_condition:  # Replace with actual condition
    pass  # Replace with actual logic