import subprocess
import shutil
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from typing import List, Optional, Tuple
from constants import FFMPEG_BITRATE

def check_ffmpeg() -> bool:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        print(f"FFmpeg найден: {ffmpeg_path}")
        return True
    else:
        print("ОШИБКА: FFmpeg не найден в системном PATH.")
        try:
            root_temp = tk.Tk()
            root_temp.withdraw()
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

def run_ffmpeg_concat(input_wav_files: List[Path], output_mp3_path: Path, temp_dir: Path) -> Tuple[bool, Optional[str]]:
    if not input_wav_files:
        return False, "Нет WAV файлов для объединения."

    list_file_path = temp_dir / "ffmpeg_concat_list.txt"
    print(f"Создание файла списка для FFmpeg: {list_file_path}")

    try:
        valid_files_count = 0
        with open(list_file_path, 'w', encoding='utf-8') as f:
            for wav_file in input_wav_files:
                if wav_file.exists() and wav_file.is_file() and wav_file.stat().st_size > 0:
                    abs_path_str = str(wav_file.resolve()).replace('\\', '/')
                    f.write(f"file '{abs_path_str}'\n")
                    valid_files_count += 1
                else:
                    print(f"Предупреждение: Пропуск невалидного/пустого файла в списке FFmpeg: {wav_file}")

        if valid_files_count == 0:
             msg = "Файл списка FFmpeg пуст или не содержит валидных файлов. Нечего объединять."
             print(msg)
             if list_file_path.exists(): list_file_path.unlink()
             return False, msg

        output_mp3_path.parent.mkdir(parents=True, exist_ok=True)

        command = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(list_file_path),
            '-codec:a', 'libmp3lame',
            '-b:a', FFMPEG_BITRATE,
            '-y',
            str(output_mp3_path)
        ]

        print(f"Запуск FFmpeg: {' '.join(command)}")

        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            encoding='utf-8',
            errors='ignore'
        )

        print("FFmpeg stderr:\n", process.stderr)

        if process.returncode != 0:
            error_msg = f"Ошибка выполнения FFmpeg (Код возврата: {process.returncode}):\n{process.stderr[:500]}..."
            print(error_msg)
            return False, error_msg
        else:
            print(f"Файл MP3 успешно создан: {output_mp3_path}")
            return True, None

    except FileNotFoundError:
        error_msg = "ОШИБКА: FFmpeg не найден. Установите ffmpeg и добавьте его в PATH."
        print(error_msg)
        return False, error_msg
    except Exception as e_ffmpeg:
        error_msg = f"Неизвестная ошибка при работе с FFmpeg: {e_ffmpeg}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return False, error_msg
    finally:
        if list_file_path and list_file_path.exists():
            try:
                list_file_path.unlink()
                print(f"Удален файл списка FFmpeg: {list_file_path}")
            except OSError as e_clean_list:
                print(f"Не удалось удалить файл списка {list_file_path}: {e_clean_list}")