import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import fitz
import sounddevice as sd
import numpy as np
import threading
import time
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple, Any

from constants import *
import audio_utils
import text_utils
import ffmpeg_utils
from tts_utils import TTSManager

class PdfReaderApp:
    def __init__(self, root_window: tk.Tk):
        self.root = root_window
        self.root.title("PDF Reader & Speaker (Русский TTS)")
        self.root.geometry("1000x750")

        self.tts_manager = TTSManager(language=DEFAULT_LANGUAGE,
                                      model_id=DEFAULT_MODEL_ID,
                                      sample_rate=DEFAULT_SAMPLE_RATE,
                                      device_type=DEFAULT_DEVICE_TYPE)
        self.voices = DEFAULT_VOICES

        self.pdf_document: Optional[fitz.Document] = None
        self.pdf_path: Optional[Path] = None
        self.current_page_num: int = 0
        self.total_pages: int = 0

        self.playing_thread: Optional[threading.Thread] = None
        self.stop_playback_flag = threading.Event()
        self.speech_speed: float = 1.0
        self.temp_audio_files: List[Path] = []
        self.current_temp_dir: Optional[Path] = None
        self.playback_was_stopped: bool = False

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

        threading.Thread(target=self._initial_tts_load_thread, daemon=True).start()

    def _initial_tts_load_thread(self):
        self.update_status("Загрузка модели Silero TTS...")
        success, message = self.tts_manager.load_model()

        def _update_gui_after_load():
            if success:
                self.update_status(message)
                self.enable_controls()
            else:
                self.update_status("Ошибка загрузки TTS. Озвучивание недоступно.")
                if self.root.winfo_exists():
                    try:
                        messagebox.showerror("Ошибка TTS", f"Не удалось загрузить модель Silero TTS.\n{message}")
                    except tk.TclError:
                        print("Не удалось показать messagebox (root уничтожен?).")
            self.enable_controls()

        if self.root and self.root.winfo_exists():
            self.root.after(0, _update_gui_after_load)

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
        self.text_display = scrolledtext.ScrolledText(text_container, wrap=tk.WORD, state=tk.DISABLED, height=10, relief=tk.FLAT, bd=0)
        self.text_display.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.text_display.tag_configure("highlight", background=HIGHLIGHT_BG, foreground=HIGHLIGHT_FG)
        display_pane.add(text_frame, weight=1)

    def _create_status_bar(self):
        self.status_label = ttk.Label(self.root, text="Загрузите PDF файл...", relief=tk.SUNKEN, anchor=tk.W, padding="2 5")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def update_speed(self, value: str):
        try:
            self.speech_speed = round(float(value), 1)
            if self.speed_label:
                self.speed_label.config(text=f"{self.speech_speed:.1f}x")
        except ValueError:
            pass

    def enable_controls(self, playback_running: bool = False):
        if not self.root or not self.root.winfo_exists():
            return

        is_ready_for_action = self.tts_manager.is_ready and self.pdf_document is not None

        normal_state = tk.NORMAL
        disabled_state = tk.DISABLED

        play_state = disabled_state if playback_running else (normal_state if is_ready_for_action else disabled_state)
        stop_state = normal_state if playback_running else disabled_state

        nav_range_state = disabled_state if playback_running else (normal_state if self.pdf_document is not None else disabled_state)

        load_state = disabled_state if playback_running else normal_state

        save_full_state = disabled_state if playback_running else (normal_state if is_ready_for_action else disabled_state)

        can_save_stopped = (is_ready_for_action and
                            not playback_running and
                            self.playback_was_stopped and
                            self.temp_audio_files)
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

        speed_voice_state = normal_state if self.tts_manager.is_ready else disabled_state
        if self.speed_slider: self.speed_slider.config(state=speed_voice_state)
        if self.voice_combobox: self.voice_combobox.config(state=speed_voice_state)

        if self.text_display: self.text_display.config(state=tk.DISABLED)

    def select_pdf(self):
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

            self.pdf_document = fitz.open(filepath, filetype="pdf")
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
        if self.pdf_image_label:
            self.pdf_image_label.config(image='')
            self.pdf_image_label.image = None
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
        if not self.pdf_document or not (0 <= page_index < self.total_pages):
            print(f"Попытка показать некорректную страницу: {page_index+1} / {self.total_pages}")
            return

        self.current_page_num = page_index
        self.update_page_label()

        try:
            page = self.pdf_document.load_page(page_index)
            zoom = 1.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            def _update_image_in_gui():
                if not self.pdf_image_label or not self.pdf_image_label.winfo_exists(): return
                img_tk = ImageTk.PhotoImage(img)
                self.pdf_image_label.config(image=img_tk)
                self.pdf_image_label.image = img_tk

            self.root.after(0, _update_image_in_gui)

        except Exception as e:
            print(f"Ошибка рендеринга страницы {page_index + 1}: {e}")
            def _clear_image_in_gui():
                 if not self.pdf_image_label or not self.pdf_image_label.winfo_exists(): return
                 self.pdf_image_label.config(image='')
                 self.pdf_image_label.image = None
            self.root.after(0, _clear_image_in_gui)
            self.update_status(f"Ошибка отображения страницы {page_index + 1}")

        try:
            text = page.get_text("text", sort=True).strip()

            def _update_text_in_gui():
                if not self.text_display or not self.text_display.winfo_exists(): return
                try:
                    self.text_display.config(state=tk.NORMAL)
                    self.text_display.delete('1.0', tk.END)
                    if text:
                        self.text_display.insert('1.0', text)
                    else:
                        self.text_display.insert('1.0', f"[Страница {page_index + 1} не содержит извлекаемого текста]")
                    self.text_display.config(state=tk.DISABLED)
                    self.text_display.see("1.0")
                    self.text_display.tag_remove("highlight", "1.0", tk.END)
                except tk.TclError as e_tk:
                    print(f"Ошибка Tkinter при обновлении текста: {e_tk}")
                except Exception as e_upd:
                     print(f"Неизвестная ошибка при обновлении текста: {e_upd}")

            self.root.after(0, _update_text_in_gui)

        except Exception as e:
            print(f"Ошибка извлечения текста со страницы {page_index + 1}: {e}")
            def _update_text_error_in_gui():
                 if not self.text_display or not self.text_display.winfo_exists(): return
                 try:
                    self.text_display.config(state=tk.NORMAL)
                    self.text_display.delete('1.0', tk.END)
                    self.text_display.insert('1.0', f"[Ошибка извлечения текста со стр. {page_index + 1}]")
                    self.text_display.config(state=tk.DISABLED)
                 except tk.TclError as e_tk:
                      print(f"Ошибка Tkinter при обновлении текста (ошибка): {e_tk}")
            self.root.after(0, _update_text_error_in_gui)

        page = None

    def update_page_label(self):
        if self.lbl_page_display and self.lbl_page_display.winfo_exists():
            if self.pdf_document and self.total_pages > 0:
                txt = f"Стр: {self.current_page_num + 1} / {self.total_pages}"
            else:
                txt = "Стр: - / -"
            self.lbl_page_display.config(text=txt)

    def prev_page(self):
        if self.pdf_document and self.current_page_num > 0:
            self.show_page(self.current_page_num - 1)

    def next_page(self):
        if self.pdf_document and self.current_page_num < self.total_pages - 1:
            self.show_page(self.current_page_num + 1)

    def _validate_page_range(self) -> Optional[Tuple[int, int]]:
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
        if not self.tts_manager.is_ready:
            messagebox.showerror("TTS не готов", "Модель озвучивания не загружена или не готова.")
            return
        if self.playing_thread and self.playing_thread.is_alive():
            messagebox.showinfo("Занято", "Воспроизведение уже идет. Остановите текущее.")
            return

        page_range = self._validate_page_range()
        if page_range is None:
            return

        start_page_idx, end_page_idx = page_range

        self.stop_playback_flag.clear()
        self.playback_was_stopped = False
        self.cleanup_temp_files()
        self.temp_audio_files = []

        try:
            self.current_temp_dir = Path(tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX + "play_"))
            print(f"Создана временная папка для воспроизведения: {self.current_temp_dir}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось создать временную директорию: {e}")
            return

        selected_voice = self.voice_combobox.get() if self.voice_combobox else self.voices[0]

        thread_args = (start_page_idx, end_page_idx, selected_voice, self.current_temp_dir)
        self.playing_thread = threading.Thread(target=self._play_audio_thread, args=thread_args, daemon=True)

        self.enable_controls(playback_running=True)
        self.update_status(f"Запуск озвучивания страниц {start_page_idx + 1}-{end_page_idx + 1}...")
        self.playing_thread.start()

    def _play_audio_thread(self, start_page_idx: int, end_page_idx: int, voice: str, temp_dir: Path):
        print(f"Поток озвучки: Начало для стр {start_page_idx + 1}-{end_page_idx + 1}, Голос: '{voice}', Папка: {temp_dir}")
        playback_successful = True

        try:
            for page_num in range(start_page_idx, end_page_idx + 1):
                if self.stop_playback_flag.is_set():
                    print(f"Поток озвучки: Остановка обнаружена перед страницей {page_num + 1}.")
                    playback_successful = False
                    break
                if not self.pdf_document:
                    print(f"Поток озвучки: PDF документ закрыт перед страницей {page_num + 1}.")
                    playback_successful = False
                    break

                self.root.after(0, lambda p=page_num: self.show_page(p))
                self.root.after(0, self.update_status, f"Обработка страницы {page_num + 1} / {self.total_pages}...")
                time.sleep(0.05)
                if self.stop_playback_flag.is_set():
                    print(f"Поток озвучки: Остановка обнаружена после обновления GUI для стр {page_num + 1}.")
                    playback_successful = False
                    break

                page_text: Optional[str] = None
                try:
                    page = self.pdf_document.load_page(page_num)
                    try:
                        page_text = page.get_text("text", sort=True).strip()
                    finally:
                        page = None
                except Exception as e:
                    print(f"Поток озвучки: Ошибка чтения текста стр {page_num + 1}: {e}")
                    self.root.after(0, self.update_status, f"Ошибка чтения текста стр {page_num + 1}. Пропуск.")
                    time.sleep(0.5)
                    continue

                if not page_text:
                    self.root.after(0, self.update_status, f"Стр. {page_num + 1}: нет текста для озвучивания. Пропуск.")
                    time.sleep(0.2)
                    continue

                text_chunks = text_utils.split_text_into_chunks(page_text)
                num_chunks = len(text_chunks)
                print(f"Стр. {page_num + 1}: {num_chunks} фрагмент(ов) текста.")

                for i, chunk in enumerate(text_chunks):
                    if self.stop_playback_flag.is_set():
                        print(f"Поток озвучки: Остановка перед фрагментом {i+1} стр {page_num + 1}.")
                        playback_successful = False
                        break
                    if not self.pdf_document:
                        print(f"Поток озвучки: PDF закрыт перед фрагментом {i+1} стр {page_num + 1}.")
                        playback_successful = False
                        break

                    self.root.after(0, self.update_status, f"Стр. {page_num + 1}: Озвучивание фрагмента {i+1}/{num_chunks}...")
                    self.root.after(0, lambda ft=page_text, c=chunk: self.highlight_text(ft, c))

                    audio_np = self.tts_manager.generate_chunk(chunk, voice, self.speech_speed)

                    if audio_np is not None and audio_np.size > 0:
                        temp_wav_path = temp_dir / f"chunk_{page_num:04d}_{i:04d}.wav"
                        try:
                            audio_utils.save_audio_to_wav(temp_wav_path, audio_np, self.tts_manager.sample_rate)
                            self.temp_audio_files.append(temp_wav_path)
                        except Exception as write_e:
                            print(f"Ошибка записи временного WAV файла {temp_wav_path}: {write_e}")

                        play_success = self._play_audio_chunk_sync(audio_np)
                        if not play_success:
                            self.root.after(0, self.update_status, f"Ошибка воспроизведения на стр. {page_num + 1}. Остановка.")
                            self.stop_playback_flag.set()
                            playback_successful = False
                            break

                    elif audio_np is None:
                         self.root.after(0, self.update_status, f"Ошибка генерации TTS на стр. {page_num + 1}. Остановка.")
                         print(f"Ошибка TTS для фрагмента: '{chunk[:60]}...'")
                         self.stop_playback_flag.set()
                         playback_successful = False
                         break

                    if self.stop_playback_flag.is_set():
                        print(f"Поток озвучки: Остановка после фрагмента {i+1} стр {page_num + 1}.")
                        playback_successful = False
                        break

                if not playback_successful:
                    break

        except Exception as e:
             print(f"Критическая ошибка в потоке озвучивания: {e}")
             import traceback
             traceback.print_exc()
             self.root.after(0, self.update_status, f"Критическая ошибка озвучивания: {e}")
             playback_successful = False
             self.stop_playback_flag.set()

        finally:
            print("Поток озвучки: Блок finally достигнут.")
            sd.stop()
            self.root.after(0, self.on_playback_finished, playback_successful)

    def _play_audio_chunk_sync(self, audio_np: Optional[np.ndarray]) -> bool:
        if audio_np is None or audio_np.size == 0:
            print("Нет аудио данных для воспроизведения.")
            return False

        try:
            sd.play(audio_np, self.tts_manager.sample_rate)
            start_time = time.monotonic()
            duration = len(audio_np) / self.tts_manager.sample_rate
            while time.monotonic() - start_time < duration:
                if self.stop_playback_flag.is_set():
                    sd.stop()
                    print("Воспроизведение фрагмента прервано.")
                    return False
                time.sleep(0.02)
            return True
        except Exception as e:
            print(f"Ошибка воспроизведения аудио: {e}")
            sd.stop()
            return False

    def stop_audio(self):
        if self.playing_thread and self.playing_thread.is_alive():
            print("Сигнал остановки потоку воспроизведения...")
            self.playback_was_stopped = True
            self.stop_playback_flag.set()
            sd.stop()
            self.update_status("Остановка воспроизведения...")
        else:
            sd.stop()

    def on_playback_finished(self, completed_normally: bool):
        print(f"Обработка завершения озвучивания. Завершено нормально: {completed_normally}, Остановка кнопкой: {self.playback_was_stopped}")

        self.playing_thread = None

        if completed_normally and not self.playback_was_stopped:
            self.update_status("Озвучивание диапазона завершено.")
            self.cleanup_temp_files()
            self.playback_was_stopped = False
        elif self.playback_was_stopped:
            self.update_status("Воспроизведение остановлено. Временные файлы сохранены для MP3.")
        else:
            current_status = self.status_label.cget("text") if self.status_label else ""
            if "Ошибка" not in current_status and "Критическая" not in current_status:
                 self.update_status("Воспроизведение прервано.")
            self.cleanup_temp_files()
            self.playback_was_stopped = False

        self.enable_controls(playback_running=False)

    def cleanup_temp_files(self):
        files_to_clear = list(self.temp_audio_files)
        dir_to_clear = self.current_temp_dir

        if not files_to_clear and not dir_to_clear:
            return

        print(f"Начало очистки временных данных...")
        cleaned_files_count = 0

        self.temp_audio_files.clear()
        self.current_temp_dir = None

        for f_path in files_to_clear:
            try:
                if f_path.exists():
                    f_path.unlink()
                    cleaned_files_count += 1
            except OSError as e_clean:
                print(f"Не удалось удалить временный файл {f_path}: {e_clean}")

        print(f"Удалено временных файлов: {cleaned_files_count}.")

        if dir_to_clear:
            print(f"Попытка удаления временной папки: {dir_to_clear}")
            try:
                if dir_to_clear.exists() and dir_to_clear.is_dir():
                    shutil.rmtree(dir_to_clear, ignore_errors=True)
                    print(f"Временная папка удалена: {dir_to_clear}")
            except Exception as e_rmdir:
                print(f"Не удалось полностью удалить временную папку {dir_to_clear}: {e_rmdir}")

        self.root.after(0, self.enable_controls, False)

    def highlight_text(self, full_page_text: str, current_chunk: str):
        if not self.text_display or not self.text_display.winfo_exists():
            return

        try:
            self.text_display.config(state=tk.NORMAL)

            self.text_display.tag_remove("highlight", "1.0", tk.END)

            start_index = -1
            try:
                start_index = full_page_text.index(current_chunk)
            except ValueError:
                 print(f"Предупреждение: Не удалось точно найти фрагмент для подсветки: '{current_chunk[:30]}...'")

            if start_index != -1:
                end_index = start_index + len(current_chunk)
                start_tk_idx = f"1.0 + {start_index} chars"
                end_tk_idx = f"1.0 + {end_index} chars"

                self.text_display.tag_add("highlight", start_tk_idx, end_tk_idx)
                self.text_display.see(start_tk_idx)

        except tk.TclError as e:
            print(f"Ошибка Tkinter при подсветке текста: {e}")
        except Exception as e_highlight:
            print(f"Неизвестная ошибка при подсветке текста: {e_highlight}")
        finally:
             if self.text_display and self.text_display.winfo_exists():
                 try:
                      self.text_display.config(state=tk.DISABLED)
                 except tk.TclError:
                      pass

    def save_full_audio_to_mp3(self):
        if not self.tts_manager.is_ready:
            messagebox.showerror("TTS не готов", "Модель озвучивания не готова.")
            return

        page_range = self._validate_page_range()
        if page_range is None:
            return

        start_page_idx, end_page_idx = page_range

        default_filename = f"{self.pdf_path.stem}_pages_{start_page_idx+1}-{end_page_idx+1}.mp3" if self.pdf_path else "output.mp3"
        output_path_str = filedialog.asksaveasfilename(
            title="Сохранить ПОЛНЫЙ MP3 файл",
            defaultextension=".mp3",
            filetypes=[("MP3 Files", "*.mp3"), ("All Files", "*.*")],
            initialfile=default_filename
        )
        if not output_path_str:
            return

        output_path = Path(output_path_str)
        selected_voice = self.voice_combobox.get() if self.voice_combobox else self.voices[0]

        self.enable_controls(playback_running=True)
        self.update_status("Генерация полного MP3... Это может занять время.")
        self.root.update_idletasks()

        save_thread = threading.Thread(
            target=self._run_full_save_mp3_thread,
            args=(output_path, start_page_idx, end_page_idx, selected_voice),
            daemon=True
        )
        save_thread.start()

    def _run_full_save_mp3_thread(self, output_path: Path, start_page_idx: int, end_page_idx: int, voice: str):
        print(f"Поток сохранения (полный): Начало стр {start_page_idx + 1}-{end_page_idx + 1}, Голос: '{voice}', Файл: {output_path}")
        temp_dir_save: Optional[Path] = None
        temp_wav_files_save: List[Path] = []
        generation_successful = True
        pdf_doc_local: Optional[fitz.Document] = None

        try:
            if not self.pdf_path or not self.pdf_path.exists():
                 raise ValueError("Путь к PDF файлу недействителен или файл не существует.")
            pdf_doc_local = fitz.open(self.pdf_path, filetype="pdf")
            if len(pdf_doc_local) != self.total_pages:
                 print("Предупреждение: Количество страниц в повторно открытом PDF отличается.")
                 end_page_idx = min(end_page_idx, len(pdf_doc_local) - 1)
                 self.total_pages = len(pdf_doc_local)

            temp_dir_save = Path(tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX + "fullsave_"))
            print(f"Временная папка для полного сохранения: {temp_dir_save}")

            total_pages_in_range = end_page_idx - start_page_idx + 1
            for i, page_num in enumerate(range(start_page_idx, end_page_idx + 1)):
                current_page_display = page_num + 1
                progress_percent = int(((i + 1) / total_pages_in_range) * 100) if total_pages_in_range > 0 else 100
                self.root.after(0, self.update_status, f"Генерация MP3: Обработка стр {current_page_display}/{self.total_pages} ({progress_percent}%)...")

                page_text_save: Optional[str] = None
                try:
                    page = pdf_doc_local.load_page(page_num)
                    try:
                        page_text_save = page.get_text("text", sort=True).strip()
                    finally:
                        page = None
                except Exception as e:
                    print(f"Ошибка чтения стр {current_page_display} при полном сохранении: {e}. Пропуск.")
                    continue

                if not page_text_save:
                    print(f"Стр. {current_page_display} пуста, пропуск при сохранении.")
                    continue

                text_chunks_save = text_utils.split_text_into_chunks(page_text_save)
                for j, chunk in enumerate(text_chunks_save):
                    audio_np_save = self.tts_manager.generate_chunk(chunk, voice, speed_multiplier=1.0)

                    if audio_np_save is not None and audio_np_save.size > 0:
                        temp_wav_path = temp_dir_save / f"audio_{page_num:04d}_{j:04d}.wav"
                        try:
                            audio_utils.save_audio_to_wav(temp_wav_path, audio_np_save, self.tts_manager.sample_rate)
                            temp_wav_files_save.append(temp_wav_path)
                        except Exception as write_e:
                            print(f"Критическая ошибка записи временного WAV {temp_wav_path}: {write_e}")
                            generation_successful = False
                            break
                    elif audio_np_save is None:
                        print(f"Ошибка генерации TTS для фрагмента на стр. {current_page_display}. Сохранение будет неполным.")
                        generation_successful = False

                if not generation_successful and temp_wav_files_save:
                     print("Прерывание генерации из-за ошибки записи WAV.")
                     break

            ffmpeg_success, ffmpeg_error_msg = ffmpeg_utils.run_ffmpeg_concat(temp_wav_files_save, output_path, temp_dir_save)
            if not temp_wav_files_save:
                message = "Не удалось сгенерировать аудио фрагменты для сохранения."
                print(message)
                self.root.after(0, messagebox.showerror, "Ошибка сохранения", message)
                generation_successful = False
            elif generation_successful:
                self.root.after(0, self.update_status, "Объединение аудио и конвертация в MP3...")
                ffmpeg_success, ffmpeg_error_msg = ffmpeg_utils.run_ffmpeg_concat(temp_wav_files_save, output_path, temp_dir_save)
                if ffmpeg_success:
                    final_message = f"Файл MP3 успешно сохранён:\n{output_path}"
                    self.root.after(0, messagebox.showinfo, "Успех", final_message)
                    self.root.after(0, self.update_status, "MP3 файл сохранён.")
                else:
                    error_display = ffmpeg_error_msg or "Неизвестная ошибка FFmpeg."
                    self.root.after(0, messagebox.showerror, "Ошибка FFmpeg", f"Ошибка при конвертации аудио:\n{error_display}")
                    self.root.after(0, self.update_status, "Ошибка FFmpeg при конвертации.")
                    generation_successful = False
            else:
                 self.root.after(0, self.update_status, "Генерация аудио не удалась. MP3 не создан.")

        except Exception as e_save:
            print(f"Критическая ошибка в потоке полного сохранения: {e_save}")
            import traceback
            traceback.print_exc()
            self.root.after(0, messagebox.showerror, "Ошибка сохранения", f"Непредвиденная ошибка:\n{e_save}")
            self.root.after(0, self.update_status, "Критическая ошибка при сохранении MP3.")
            generation_successful = False
        finally:
            if pdf_doc_local:
                try:
                    pdf_doc_local.close()
                except Exception as e_close:
                     print(f"Ошибка при закрытии локального PDF в потоке сохранения: {e_close}")

            if temp_dir_save and temp_dir_save.exists():
                print(f"Очистка временной папки полного сохранения: {temp_dir_save}")
                try:
                    shutil.rmtree(temp_dir_save, ignore_errors=True)
                    print("Временная папка полного сохранения удалена.")
                except Exception as e_rmdir:
                    print(f"Не удалось полностью удалить временную папку {temp_dir_save}: {e_rmdir}")

            self.root.after(0, self.enable_controls, False)

            if not generation_successful:
                 current_status = self.status_label.cget("text") if self.status_label else ""
                 if "Ошибка" not in current_status and "MP3 не создан" not in current_status:
                    self.root.after(0, self.update_status, "Сохранение MP3 завершено с ошибками.")

    def save_stopped_audio_to_mp3(self):
        if not self.playback_was_stopped or not self.temp_audio_files or not self.current_temp_dir:
            messagebox.showwarning("Нечего сохранять", "Нет записанных аудио фрагментов после остановки.")
            return

        files_to_save = list(self.temp_audio_files)
        temp_dir_to_use = self.current_temp_dir

        default_filename = f"{self.pdf_path.stem}_stopped.mp3" if self.pdf_path else "stopped_audio.mp3"
        output_path_str = filedialog.asksaveasfilename(
            title="Сохранить ОСТАНОВЛЕННЫЙ MP3 файл",
            defaultextension=".mp3",
            filetypes=[("MP3 Files", "*.mp3"), ("All Files", "*.*")],
            initialfile=default_filename
        )
        if not output_path_str:
            return

        output_path = Path(output_path_str)

        self.enable_controls(playback_running=True)
        self.update_status("Объединение остановленных аудио в MP3...")
        self.root.update_idletasks()

        save_thread = threading.Thread(
            target=self._run_stopped_save_mp3_thread,
            args=(output_path, files_to_save, temp_dir_to_use),
            daemon=True
        )
        save_thread.start()

    def _run_stopped_save_mp3_thread(self, output_path: Path, files_to_concat: List[Path], temp_dir_path: Path):
        print(f"Поток сохранения (остановленный): Начало для {len(files_to_concat)} файлов из {temp_dir_path}, Выход: {output_path}")
        success = False
        error_message = None
        try:
            if not files_to_concat:
                self.root.after(0, self.update_status, "Нет аудио файлов для сохранения.")
                return

            success, error_message = ffmpeg_utils.run_ffmpeg_concat(files_to_concat, output_path, temp_dir_path)

            if success:
                final_message = f"Остановленное аудио успешно сохранено:\n{output_path}"
                self.root.after(0, messagebox.showinfo, "Успех", final_message)
                self.root.after(0, self.update_status, "Остановленный MP3 сохранён.")
                def _finalize_stopped_save():
                    self.playback_was_stopped = False
                    self.cleanup_temp_files()
                    self.enable_controls(playback_running=False)
                self.root.after(0, _finalize_stopped_save)

            else:
                error_display = error_message or "Неизвестная ошибка FFmpeg."
                self.root.after(0, messagebox.showerror, "Ошибка FFmpeg", f"Ошибка при сохранении остановленного аудио:\n{error_display}")
                self.root.after(0, self.update_status, "Ошибка при сохранении остановленного MP3.")

        except Exception as e_stopped_save:
            print(f"Критическая ошибка в потоке сохранения остановленного: {e_stopped_save}")
            import traceback
            traceback.print_exc()
            self.root.after(0, messagebox.showerror, "Ошибка", f"Непредвиденная ошибка:\n{e_stopped_save}")
            self.root.after(0, self.update_status, "Ошибка при сохранении остановленного MP3.")
        finally:
            if not success:
                self.root.after(0, self.enable_controls, False)

    def on_closing(self):
        print("Запрос на закрытие приложения...")
        self.stop_audio()

        if self.playing_thread and self.playing_thread.is_alive():
            print("Ожидание завершения потока озвучки (макс 1 сек)...")
            self.playing_thread.join(timeout=1.0)
            if self.playing_thread.is_alive():
                 print("Предупреждение: Поток озвучки не завершился вовремя.")

        print("Финальная очистка временных файлов...")
        self.cleanup_temp_files()

        if self.pdf_document:
            print("Закрытие PDF документа...")
            try:
                self.pdf_document.close()
            except Exception as e:
                print(f"Ошибка при закрытии PDF: {e}")
            self.pdf_document = None

        print("Уничтожение окна Tkinter...")
        self.root.destroy()
        print("Приложение закрыто.")

    def update_status(self, message: str):
        print(f"Status: {message}")
        if self.status_label and self.status_label.winfo_exists():
            def _update_gui():
                if self.status_label and self.status_label.winfo_exists():
                    self.status_label.config(text=message)
            if self.root and self.root.winfo_exists():
                 self.root.after(0, _update_gui)