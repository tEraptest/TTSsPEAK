import tkinter as tk
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from app import PdfReaderApp
    from ffmpeg_utils import check_ffmpeg
except ImportError as e:
    print(f"Ошибка импорта модулей: {e}")
    print("Убедитесь, что все файлы (app.py, constants.py, ...) находятся в той же директории, что и pdf_speaker.py")
    sys.exit(1)

def check_heavy_deps():
    print("Проверка дополнительных зависимостей...")
    missing = []
    try:
        import PyYAML
        print("- PyYAML найден.")
    except ImportError:
        missing.append("PyYAML")
    try:
        import omegaconf
        print("- omegaconf найден.")
    except ImportError:
        missing.append("omegaconf")
    try:
        import sounddevice
        print("- sounddevice найден.")
    except ImportError:
         missing.append("sounddevice")
    try:
        import PIL
        print("- Pillow (PIL) найден.")
    except ImportError:
         missing.append("Pillow")
    try:
        import fitz
        print("- PyMuPDF (fitz) найден.")
    except ImportError:
         missing.append("PyMuPDF")
    try:
        import torch
        print("- PyTorch найден.")
    except ImportError:
        missing.append("torch")
    try:
        import numpy
        print("- NumPy найден.")
    except ImportError:
        missing.append("numpy")
    try:
        import scipy
        print("- SciPy найден.")
    except ImportError:
        missing.append("scipy")

    if missing:
        print("\nПредупреждение: Отсутствуют некоторые рекомендованные/необходимые зависимости:")
        for lib in missing:
            print(f"  - {lib}")
        print("Пожалуйста, установите их с помощью pip:")
        print(f"  pip install {' '.join(missing)}")
    print("Все основные зависимости найдены.")
    return True

if __name__ == "__main__":
    print("Запуск PDF Speaker...")
    if not check_heavy_deps():
        print("Установите недостающие зависимости и повторите попытку.")
    ffmpeg_available = check_ffmpeg()
    if not ffmpeg_available:
        print("Предупреждение: FFmpeg не найден. Функция сохранения MP3 будет недоступна.")
    print("Создание GUI...")
    main_root = tk.Tk()
    try:
        app = PdfReaderApp(main_root)
    except Exception as e:
        print(f"\nКритическая ошибка при инициализации приложения: {e}")
        import traceback
        traceback.print_exc()
        try:
            messagebox.showerror("Критическая ошибка", f"Не удалось инициализировать приложение:\n{e}")
        except Exception:
            pass
        sys.exit(1)
    main_root.protocol("WM_DELETE_WINDOW", app.on_closing)
    print("Запуск главного цикла приложения...")
    try:
         main_root.mainloop()
    except KeyboardInterrupt:
         print("\nПриложение прервано пользователем (Ctrl+C). Завершение работы...")
         app.on_closing()
    except Exception as main_loop_error:
        print(f"\nНеожиданная ошибка в главном цикле: {main_loop_error}")
        import traceback
        traceback.print_exc()
        try:
            app.on_closing()
        except Exception as closing_error:
            print(f"Ошибка во время принудительного закрытия: {closing_error}")
    print("Выход из pdf_speaker.py")