import time
import threading
import requests
import pyautogui
import os
import re
from PIL import ImageStat, Image, ImageEnhance
import pyperclip

import cv2
import numpy as np
import io
import logging

import base64
from mistralai import Mistral
#from mistralai.models.chat import TextChunk # ВОССТАНОВЛЕНО: УБРАН КОММЕНТАРИЙ (В соответствии с обсуждением)
# Mistral v1.5.2 and above automatically handle ContentChunk/str. No explicit import needed.

from typing import Optional, Tuple, List, Any, Dict

import pytesseract

# --- НАЧАЛО ИСПРАВЛЕНИЯ: Перемещаем telegram_log сюда ---
def telegram_log(message, is_debug_message: bool = False):
    debug_logging_actual = globals().get("DEBUG_LOGGING_ENABLED", True)
    if is_debug_message and not debug_logging_actual:
        return
    message_to_send = f"[DEBUG] {message}" if is_debug_message and debug_logging_actual else message
    current_base_url = globals().get("BASE_URL")
    current_subscribers = globals().get("subscribers")
    if not isinstance(current_subscribers, set):
        current_subscribers = set()
    if not current_subscribers or not current_base_url:
        print(f"[CONSOLE_LOG_ONLY"
              f"{' (NO_SUBSCRIBERS)' if not current_subscribers else ''}"
              f"{' (NO_BASE_URL)' if not current_base_url else ''}"
              f"] {message_to_send}")
        return
    subscribers_to_iterate = set(current_subscribers)
    for chat_id_local in subscribers_to_iterate:
        send_message(chat_id_local, message_to_send)
# --- КОНЕЦ ИСПРАВЛЕНИЯ ---

# --- НАСТРОЙКИ OCR И ОТЛАДКИ ---
OCR_PROVIDER = "mistral" # Или "tesseract" для принудительного использования Tesseract
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

DEBUG_LOGGING_ENABLED = True # Оставляем True для отладки
# ------------------------

# Глобальные переменные для Telegram-бота и настроек
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    print("[ERROR] TELEGRAM_BOT_TOKEN не найден в переменных окружения!")

BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
SUBSCRIBERS_FILE = "subscribers.txt"
subscribers = set() # Инициализируем пустой набор подписчиков

DEBUG_SCREENSHOT = True # Отправлять ли отладочные скриншоты в Telegram

# Инициализация логгирования
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

# Инициализация Mistral AI клиента
mistral_client = None
if OCR_PROVIDER == "mistral":
    if MISTRAL_API_KEY:
        try:
            mistral_client = Mistral(api_key=MISTRAL_API_KEY)
            telegram_log("[INFO] Mistral AI client initialized.", is_debug_message=True)
            print("[INFO] Mistral AI client initialized.")
        except Exception as e_mistral_init:
            telegram_log(f"[CRITICAL_ERROR] Не удалось инициализировать Mistral AI client: {e_mistral_init}. Бот завершает работу.", is_debug_message=False)
            print(f"[CRITICAL_ERROR] Не удалось инициализировать Mistral AI client: {e_mistral_init}. Бот завершает работу.")
            import sys
            sys.exit(1)
    else:
        telegram_log("[CRITICAL_ERROR] MISTRAL_API_KEY не установлен, бот завершает работу.", is_debug_message=False)
        print("[CRITICAL_ERROR] MISTRAL_API_KEY не установлен, бот завершает работу.")
        import sys
        sys.exit(1)

if OCR_PROVIDER == "mistral" and not mistral_client:
    print(f"[CRITICAL_ERROR] OCR_PROVIDER is 'mistral', but Mistral OCR client is not initialized. OCR will not work.")
    telegram_log("[CRITICAL_ERROR] OCR (Mistral) недоступен: клиент не инициализирован.", is_debug_message=False)


# ОБНОВЛЕННЫЕ ЛУЧШИЕ ПАРАМЕТРЫ ДЛЯ TESSERACT (после тюнинга)
BEST_TESSERACT_PARAMS = {
    'scale_factor': 4,
    'contrast_enhance': 2.5,
    'sharpness_enhance': 2.5,
    'adaptive_method': cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    'adaptive_block_size': 19,
    'adaptive_C': 5,
    'median_blur_kernel': 3,
    'clahe_clip_limit': 2.0,
    'clahe_tile_grid_size': (8, 8),
    'denoise_h': 5.0,
    'denoise_template_window_size': 7,
    'denoise_search_window_size': 21
}

# ОБНОВЛЕННЫЙ OCR_CHAR_MAP для более строгих паттернов
OCR_CHAR_MAP = {
    '0': '[0O]',     # Allow 'O' for zero
    '1': '[1li]',    # Allow 'l' or 'i' for one
    '2': '[2Zz]',    # Allow 'Z' or 'z' for two
    '3': '[3]',
    '4': '[4]',
    '5': '[5S]',     # Allow 'S' for five
    '6': '[6G]',     # Allow 'G' for six
    '7': '[7]',
    '8': '[8]',
    '9': '[9]',
    '-': '[-—]',    # Hyphen or em-dash (removed 'a' as it was too broad based on previous log analysis)
    '.': '[\.,]',    # Dot or comma
    '(': '[\(]',    # Escaped parenthesis
    ')': '[\)]',    # Escaped parenthesis
    '+': '[\+]',    # Escaped plus
}


def preprocess_for_tesseract(pil_image: Image.Image,
                             scale_factor: int = 3,
                             contrast_enhance: float = 1.8,
                             sharpness_enhance: float = 2.0,
                             adaptive_method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             adaptive_block_size: int = 21,
                             adaptive_C: int = 5,
                             median_blur_kernel: int = 1,
                             clahe_clip_limit: float = 2.0,
                             clahe_tile_grid_size: Tuple[int, int] = (8, 8),
                             denoise_h: float = 0.0,
                             denoise_template_window_size: int = 7,
                             denoise_search_window_size: int = 21
                            ) -> Image.Image:
    """
    Предобработка изображения для Tesseract OCR с настраиваемыми параметрами.
    Включены CLAHE и FastNlMeansDenoising.
    """

    new_width = int(pil_image.width * scale_factor)
    new_height = int(pil_image.height * scale_factor)
    pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(contrast_enhance)
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(sharpness_enhance)

    open_cv_image = np.array(pil_image.convert('RGB'))
    open_cv_image = open_cv_image[:, :, ::-1].copy() # Convert RGB to BGR

    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    # Адаптивная нормализация контраста (CLAHE)
    try:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
        gray = clahe.apply(gray)
        telegram_log(f"[DEBUG_PREPROCESS] Выполнена адаптивная нормализация контраста (CLAHE, clipLimit={clahe_clip_limit}, tileGridSize={clahe_tile_grid_size}).", is_debug_message=True)
    except Exception as e:
        telegram_log(f"[ERROR_PREPROCESS] Ошибка при применении CLAHE: {e}", is_debug_message=True)

    # Денойзинг (FastNlMeansDenoising)
    if denoise_h > 0:
        try:
            # fastNlMeansDenoising требует 8-битного одноканального изображения
            gray = cv2.fastNlMeansDenoising(gray, None, h=denoise_h,
                                             templateWindowSize=denoise_template_window_size,
                                             searchWindowSize=denoise_search_window_size)
            telegram_log(f"[DEBUG_PREPROCESS] Выполнен денойзинг (FastNlMeansDenoising, h={denoise_h}, templateWS={denoise_template_window_size}, searchWS={denoise_search_window_size}).", is_debug_message=True)
        except Exception as e:
            telegram_log(f"[ERROR_PREPROCESS] Ошибка при применении FastNlMeansDenoising: {e}", is_debug_message=True)

    thresh = cv2.adaptiveThreshold(
        gray, 255, adaptive_method,
        cv2.THRESH_BINARY, adaptive_block_size, adaptive_C
    )
    telegram_log(f"[DEBUG_PREPROCESS] Выполнена адаптивная бинаризация (method={adaptive_method}, blockSize={adaptive_block_size}, C={adaptive_C}).", is_debug_message=True)

    if median_blur_kernel > 1:
        thresh = cv2.medianBlur(thresh, median_blur_kernel)

    return Image.fromarray(thresh)

def create_flexible_pattern(input_string: str) -> str:
    """
    Creates a regex pattern from a string, allowing for *minimal* common OCR errors
    and optional spaces.
    """
    pattern_chars = []
    for char in input_string:
        if char in OCR_CHAR_MAP:
            pattern_chars.append(OCR_CHAR_MAP[char])
        else:
            pattern_chars.append(re.escape(char.lower())) # For other chars, just escape and lower for a general match
    return '(?:\s*' + ''.join(pattern_chars) + '\s*)' # Allow optional spaces around the full pattern


def extract_text_mistral_ocr(pil_image: Image.Image) -> Tuple[str, List[List[Any]]]:
    """
    Использует vision через chat completion (mistralai==1.5.2) для распознавания текста с изображения.
    Возвращает полный текст и пустой список блоков (bounding box не поддерживается).
    Добавлена логика повторных попыток для сетевых ошибок.
    """
    if not mistral_client:
        telegram_log("[ERROR_MISTRAL_OCR] Mistral client is not initialized. OCR call skipped.", is_debug_message=True)
        return "", []

    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    image_data_url = f"data:image/png;base64,{img_base64}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract all text from this image. Preserve line breaks."},
                {"type": "image_url", "image_url": image_data_url}
            ]
        }
    ]

    max_retries = 3
    retry_delay_seconds = 5

    for attempt in range(max_retries):
        try:
            telegram_log(f"[DEBUG_MISTRAL_API] Попытка {attempt + 1}/{max_retries}: Вызов Mistral OCR (chat vision)...", is_debug_message=True)
            response = mistral_client.chat.complete(
                model="mistral-small-latest",
                messages=messages,
                temperature=0.0,
            )

            full_text = ""
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                content_result = response.choices[0].message.content
                # Mistralai==1.5.2 and above directly returns str or list of str/ContentChunk,
                # we handle both for robustness.
                if isinstance(content_result, list):
                    all_text_chunks = []
                    for chunk in content_result:
                        if hasattr(chunk, 'text'): # For ContentChunk objects
                            all_text_chunks.append(chunk.text.strip())
                        elif isinstance(chunk, str): # For raw string chunks
                            all_text_chunks.append(chunk.strip())
                    full_text = "\n".join(all_text_chunks)
                elif isinstance(content_result, str):
                    full_text = content_result.strip()

            if full_text:
                telegram_log(f"[DEBUG_MISTRAL_API] Mistral OCR успешно (Попытка {attempt + 1}). Длина текста: {len(full_text)}", is_debug_message=True)
                return full_text, []
            else:
                telegram_log(f"[ERROR_MISTRAL_API] Mistral OCR не вернул контент (Попытка {attempt + 1}).", is_debug_message=True)
                if attempt < max_retries - 1:
                    telegram_log(f"[DEBUG_MISTRAL_API] Повторная попытка после пустого контента через {retry_delay_seconds} секунд...", is_debug_message=True)
                    time.sleep(retry_delay_seconds)
                else:
                    telegram_log(f"[ERROR_MISTRAL_API] Все {max_retries} попыток не смогли получить контент от Mistral OCR.", is_debug_message=False)
                    return "", []

        except requests.exceptions.RequestException as req_e:
            error_msg = f"Ошибка сети/запроса в extract_text_mistral_ocr (chat vision) (Попытка {attempt + 1}): {req_e}"
            logging.warning(error_msg)
            telegram_log(error_msg, is_debug_message=True)
            if attempt < max_retries - 1:
                telegram_log(f"[DEBUG_MISTRAL_API] Повторная попытка через {retry_delay_seconds} секунд...", is_debug_message=True)
                time.sleep(retry_delay_seconds)
            else:
                telegram_log(f"[ERROR_MISTRAL_API] Все {max_retries} попыток не удались из-за ошибки сети/запроса.", is_debug_message=False)
                return "", []
        except Exception as e:
            error_msg = f"Необработанное исключение в extract_text_mistral_ocr (chat vision) (Попытка {attempt + 1}): {e}"
            logging.exception(error_msg)
            telegram_log(error_msg, is_debug_message=False)
            return "", []

    return "", []

def extract_text_tesseract(pil_image: Image.Image, **kwargs) -> Tuple[str, List[List[Any]]]:
    """
    Использует Tesseract OCR для распознавания текста с изображения.
    Принимает параметры для предобработки через kwargs.
    Возвращает координаты, масштабированные обратно к исходному размеру скриншота.
    """
    try:
        local_kwargs = kwargs.copy()

        if 'adaptive_block_size' in local_kwargs and local_kwargs['adaptive_block_size'] % 2 == 0:
            telegram_log(f"[WARNING] Tesseract: adaptive_block_size ({local_kwargs['adaptive_block_size']}) должно быть нечетным. Увеличиваю на 1.", is_debug_message=True)
            local_kwargs['adaptive_block_size'] += 1
        if 'adaptive_block_size' in local_kwargs and local_kwargs['adaptive_block_size'] <= 1:
            telegram_log(f"[WARNING] Tesseract: adaptive_block_size ({local_kwargs['adaptive_block_size']}) должно быть больше 1. Устанавливаю 3.", is_debug_message=True)
            local_kwargs['adaptive_block_size'] = 3

        # Получаем scale_factor для обратного масштабирования
        scale_factor = local_kwargs.get('scale_factor', 1)
        if scale_factor <= 0:
            telegram_log(f"[ERROR] Tesseract: Некорректный scale_factor ({scale_factor}). Устанавливаю 1.", is_debug_message=True)
            scale_factor = 1

        processed_img = preprocess_for_tesseract(pil_image, **local_kwargs)

        custom_config = r'--oem 1 --psm 3'

        full_text = pytesseract.image_to_string(
            processed_img,
            lang='rus+eng',
            config=custom_config
        )

        data = pytesseract.image_to_data(
            processed_img,
            lang='rus+eng',
            output_type=pytesseract.Output.DICT,
            config=custom_config
        )

        results = []
        n_boxes = len(data['level'])
        for i in range(n_boxes):
            try:
                conf = float(data['conf'][i])
            except ValueError:
                conf = 0.0
            text = data['text'][i]

            # Получаем координаты из Tesseract (они масштабированы)
            (x_scaled, y_scaled, w_scaled, h_scaled) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])

            # ОБРАТНОЕ МАСШТАБИРОВАНИЕ КООРДИНАТ К ИСХОДНОМУ РАЗМЕРУ
            x_original = int(x_scaled / scale_factor)
            y_original = int(y_scaled / scale_factor)
            w_original = int(w_scaled / scale_factor)
            h_original = int(h_scaled / scale_factor)

            vertices = [(x_original, y_original), (x_original + w_original, y_original), (x_original + w_original, y_original + h_original), (x_original, y_original + h_original)]
            results.append([vertices, text, conf])

        telegram_log(f"[DEBUG_TESSERACT_COORDS] Tesseract: Обратное масштабирование координат выполнено с scale_factor={scale_factor}. Пример первой координаты: (x_scaled={data['left'][0] if data['left'] else 'N/A'}, y_scaled={data['top'][0] if data['top'] else 'N/A'}) -> (x_original={results[0][0][0][0] if results else 'N/A'}, y_original={results[0][0][0][1] if results else 'N/A'})", is_debug_message=True)

        return full_text, results
    except Exception as e:
        telegram_log(f"[ERROR_TESSERACT] {e}")
        return "", []

def get_ocr_results(pil_image: Image.Image) -> Tuple[str, List[List[Any]]]:
    """
    Обертка для вызова выбранного OCR провайдера.
    Теперь поддерживает: mistral, tesseract.
    """
    if OCR_PROVIDER == "mistral" and mistral_client:
        print("[INFO_OCR] Using Mistral OCR.")
        return extract_text_mistral_ocr(pil_image)
    elif OCR_PROVIDER == "tesseract":
        print("[INFO_OCR] Using Tesseract OCR.")
        return extract_text_tesseract(pil_image, **BEST_TESSERACT_PARAMS)
    elif OCR_PROVIDER == "mistral" and not mistral_client:
        print("[WARNING_OCR] OCR_PROVIDER set to mistral but client not available, falling back to Tesseract.")
        telegram_log("[WARNING_OCR] OCR_PROVIDER set to mistral but client not available, falling back to Tesseract.", is_debug_message=True)
        return extract_text_tesseract(pil_image, **BEST_TESSERACT_PARAMS)
    else:
        print("[ERROR_OCR] No OCR provider is available or initialized correctly.")
        telegram_log("[ERROR_OCR] No OCR provider is available or initialized correctly.", is_debug_message=False)
        return "", []

# NOTE: tune_tesseract_preprocessing is a utility for offline tuning and is not used in runtime.
# It is kept here as a reference but could be moved to a separate utility script.
# def tune_tesseract_preprocessing(...)

def load_subscribers():
    subscribers_file_path = globals().get("SUBSCRIBERS_FILE", "subscribers.txt")
    if os.path.exists(subscribers_file_path):
        try:
            loaded_subs = set()
            with open(subscribers_file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        loaded_subs.add(line)
                globals()["subscribers"] = loaded_subs
                print(f"[INFO] Загружено подписчиков: {loaded_subs}")
        except Exception as e:
            print(f"[ERROR] Ошибка при загрузке подписчиков из {subscribers_file_path}: {e}")
    else:
        if "subscribers" not in globals() or not isinstance(globals().get("subscribers"), set):
            globals()["subscribers"] = set()

def save_subscribers():
    subscribers_global = globals().get("subscribers", set())
    subscribers_file_global = globals().get("SUBSCRIBERS_FILE", "subscribers.txt")
    try:
        with open(subscribers_file_global, "w") as f:
            for sub in subscribers_global:
                f.write(f"{sub}\n")
    except Exception as e:
        print(f"[ERROR] Ошибка при сохранении подписчиков в {subscribers_file_global}: {e}")

def send_message(chat_id, message):
    _current_base_url = globals().get("BASE_URL")
    if not _current_base_url:
        print(f"[SEND_MSG_FAIL_NO_BASE_URL_YET] Сообщение для {chat_id} не отправлено (BASE_URL не инициализирован): {message}")
        return
    url = f"{_current_base_url}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        response = requests.post(url, data=payload, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при отправке сообщения для {chat_id}: {e}")
    except Exception as e_general:
        print(f"Неожиданная ошибка при отправке сообщения для {chat_id}: {e_general}")

def send_photo(chat_id, photo_path, caption=""):
    _current_base_url = globals().get("BASE_URL")
    if not _current_base_url:
        print(f"[SEND_PHOTO_FAIL_NO_BASE_URL_YET] Фото для {chat_id} не отправлено (BASE_URL не инициализирован): {photo_path}")
        return
    url = f"{_current_base_url}/sendPhoto"
    try:
        with open(photo_path, "rb") as photo_file:
            files = {"photo": photo_file}
            data = {"chat_id": chat_id, "caption": caption}
            response = requests.post(url, data=data, files=files, timeout=30)
            response.raise_for_status()
    except FileNotFoundError:
        print(f"Ошибка при отправке фото: файл {photo_path} не найден.")
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при отправке фото для {chat_id} ({photo_path}): {e}")
    except Exception as e_general:
        print(f"Неожиданная ошибка при отправке фото для {chat_id} ({photo_path}): {e_general}")


def poll_updates():
    try:
        initial_response = requests.get(f"{BASE_URL}/getUpdates", params={"timeout": 1}, timeout=5)
        initial_data = initial_response.json()
        initial_updates = initial_data.get("result", [])
        if initial_updates:
            offset = initial_updates[-1]["update_id"] + 1
            print(f"[INFO] Пропущены старые сообщения, начинаем с offset: {offset}")
        else:
            offset = None
    except Exception as e:
        print("[ERROR] Ошибка при инициализации offset:", e)
        offset = None

    while True:
        params = {'timeout': 10, 'offset': offset}
        try:
            response = requests.get(f"{BASE_URL}/getUpdates", params=params, timeout=15)
            data = response.json()
            for update in data.get("result", []):
                offset = update["update_id"] + 1
                message = update.get("message")
                if not message:
                    continue
                chat_id = str(message["chat"]["id"])
                text = message.get("text", "")

                if text.lower() == "/start":
                    if chat_id not in subscribers:
                        subscribers.add(chat_id)
                        save_subscribers()
                        send_message(chat_id, "Вы подписались на логи бота!")
                        print(f"[INFO] Новый подписчик: {chat_id}")
                else:
                    print(f"[INFO] Получены данные ставки от пользователя {chat_id}: {text}")
                    telegram_log(f"Получены данные ставки от пользователя {chat_id}: {text}")
                    parts = [part.strip() for part in text.split(",")]
                    if len(parts) != 4:
                        print("[ERROR] Неверный формат ставки! Ожидается: матч, Исход, кэф, размер ставки")
                        telegram_log("Неверный формат ставки!")
                        continue
                    match_name, outcome, coef_condition, bet_amount_str = parts
                    try:
                        bet_amount = float(bet_amount_str.replace(",", "."))
                    except ValueError:
                        print("[ERROR] Размер ставки не является числом!")
                        telegram_log("Ошибка: Размер ставки не является числом!")
                        continue
                    find_match(match_name)
                    time.sleep(1)
                    result = find_outcome(match_name, outcome, coef_condition, bet_amount)
                    if result:
                        print("[INFO] Ставка успешно обработана!")
                    else:
                        print("[INFO] Ставка не обработана, требуется повторная попытка.")
                        telegram_log("Ставка не обработана, требуется повторная попытка.")
        except Exception as e:
            print("Ошибка при получении обновлений:", e)
        time.sleep(1)

def open_browser_and_navigate():
    pyautogui.click(413, 363)
    telegram_log("[STEP 0] Клик по координатам (413, 363) для ввода proxy")
    pyautogui.write("otwyn7rnye-mobile-country-RU-state-524894-city-524901-hold-session-session-68234428f20a1", interval=0.05)
    time.sleep(1)
    telegram_log("[STEP 0] Введена proxy-строка")
    pyautogui.click(410, 396)
    telegram_log("[STEP 0] Клик по координатам (410, 396) для ввода данных авторизации")
    pyautogui.write("kVgpz87hTSt7wsF6")
    telegram_log("[STEP 0] Введён пароль")
    pyautogui.press("enter")
    telegram_log("[STEP 0] Нажат Enter для отправки данных авторизации")
    time.sleep(2)
    pyautogui.hotkey('ctrl', 'l')
    telegram_log("[STEP 1] Открыта адресная строка (ctrl+l)")
    time.sleep(1)
    pyautogui.write("https://www.marathonbet.ru/", interval=0.05)
    telegram_log("[STEP 1] Введён URL: https://www.marathonbet.ru/")
    pyautogui.press("enter")
    telegram_log("[STEP 1] Нажат Enter для открытия сайта")

def wait_for_site_ready_color(target_color, color_tolerance=10, check_region=(604, 119, 5, 5)):
    """
    Ждёт, пока в зоне check_region (5x5 пикселей) цвет не станет близким к target_color (± color_tolerance).
    Если цвет не совпадает, ждёт 10 секунд и пробует снова.
    """
    while True:
        screenshot_candidate = pyautogui.screenshot(region=check_region)
        stat = ImageStat.Stat(screenshot_candidate)
        avg_color = tuple(int(c) for c in stat.mean)
        telegram_log(f"[DEBUG] Checking site color at {check_region}: {avg_color}", is_debug_message=True)

        if all(abs(avg_color[i] - target_color[i]) <= color_tolerance for i in range(3)):
            telegram_log("[INFO] Site color matched, proceeding to login.", is_debug_message=True)
            break
        else:
            telegram_log("[INFO] Site color not matched, waiting 10 seconds before retry.", is_debug_message=True)
            time.sleep(10)

def check_for_text(expected_text, top_left, bottom_right, timeout=15):
    """
    Ожидает появления строки expected_text в области (top_left => bottom_right) не дольше timeout секунд.
    Для распознавания текста используется OCR.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right
    region = (x1, y1, x2 - x1, y2 - y1)
    start = time.time()
    screenshot_sent = False
    while time.time() - start < timeout:
        screenshot = pyautogui.screenshot(region=region)
        time.sleep(1)
        if DEBUG_SCREENSHOT and not screenshot_sent:
            debug_path = "debug_screenshot.png"
            screenshot.save(debug_path)
            for chat_id in subscribers:
                send_photo(chat_id, debug_path, caption="Тестовый скриншот")
            screenshot_sent = True
        full_text, _ = get_ocr_results(screenshot)
        print(f"[DEBUG] OCR-текст в зоне {region}: {full_text}")
        if expected_text.lower() in full_text.lower():
            if DEBUG_SCREENSHOT:
                telegram_log("Распознанный текст: " + full_text, is_debug_message=True)
            return True
        time.sleep(1)
    return False

def do_login():
    print("[INFO] Выполняется логин...")
    telegram_log("[STEP 2] Логин")
    time.sleep(10)
    telegram_log("[DEBUG] Клик по (1138, 134)")
    pyautogui.click(1138, 134)
    time.sleep(5)
    telegram_log("[DEBUG] Клик по (1138, 182)")
    pyautogui.click(1138, 182)
    time.sleep(5)
    telegram_log("[DEBUG] Клик по (600, 377) и ввод логина")
    pyautogui.click(600, 377, clicks=2)
    pyautogui.write("9214111699", interval=0.1)
    time.sleep(1)
    telegram_log("[DEBUG] Клик по (600, 429) и ввод пароля")
    pyautogui.click(600, 429, clicks=2)
    pyautogui.write("Gamma1488", interval=0.1)
    time.sleep(1)
    pyautogui.click(600, 516, clicks=1)
    telegram_log("[DEBUG] Клик по (600, 516)")
    time.sleep(15)
    telegram_log("[DEBUG] Клик по (874, 668) для убирания баннера")
    pyautogui.click(874, 668, clicks=2)
    time.sleep(3)
    telegram_log("[DEBUG] Клик по (976, 294) для убирания баннера")
    pyautogui.click(976, 294, clicks=2)
    time.sleep(3)

    pyautogui.click(874, 668, clicks=2)
    time.sleep(3)
    pyautogui.click(976, 294, clicks=2)
    time.sleep(3)


def find_match(match_name):
    """
    Переходит в лайв и вводит название матча match_name в поиске.
    Использует буфер обмена для корректного ввода русских названий.
    ОСТАЕТСЯ НА СТРАНИЦЕ РЕЗУЛЬТАТОВ ПОИСКА.
    """
    pyautogui.click(1243, 193)
    time.sleep(1)

    try:
        pyperclip.copy(match_name)
        pyautogui.hotkey('shift', 'insert')
        telegram_log(f"Имя матча введено через буфер обмена: {match_name}", is_debug_message=True)
        pyautogui.press("enter")
    except Exception as e:
        telegram_log(f"Ошибка при вводе имени матча: {e}")

    time.sleep(10)

    live_coords = (284, 359)

    pyautogui.click(live_coords[0], live_coords[1])
    telegram_log("[INFO] Клик по вкладке/фильтру 'Live'.", is_debug_message=True)
    time.sleep(10)

    telegram_log("[DEBUG][INFO] Поиск матча выполнен. Остаемся на странице результатов поиска для обработки исходов.", is_debug_message=True)


def is_fuzzy_match(target_name: str, ocr_text: str) -> bool:
    """
    Проверяет, соответствует ли OCR-текст целевому имени, используя нечеткое совпадение.
    Учитывает пробелы, дефисы и частичные совпадения слов.
    """
    target_lower = target_name.lower().replace('-', ' ').strip()
    ocr_lower = ocr_text.lower().replace('-', ' ').strip()

    if target_lower == ocr_lower:
        return True

    if target_lower in ocr_lower:
        return True

    target_words = [w for w in target_lower.split() if w]
    ocr_words = [w for w in ocr_lower.split() if w]

    if len(target_words) > 0:
        matched_words_count = 0
        for t_word in target_words:
            if t_word in ocr_words: # Basic word presence
                matched_words_count += 1

        # Consider a match if at least half of the target words are present,
        # or if it's a single word and it's a substring of an OCR word (or vice-versa).
        if matched_words_count >= len(target_words) / 2:
            return True

        if len(target_words) == 1 and len(target_words[0]) > 3: # For single, longer words
            for o_word in ocr_words:
                if target_words[0] in o_word or o_word in target_words[0]:
                    return True

    return False

def _click_handicap_from_blocks(target_outcome: str, ocr_blocks: List[List[Any]], team_name_target: Optional[str], region_offset_x: int = 0, region_offset_y: int = 0) -> bool:
    """
    Находит и кликает по исходу форы в данном наборе OCR-блоков.
    Координаты ocr_blocks уже масштабированы обратно к исходному размеру скриншота,
    они относительны к переданному региону.
    region_offset_x/y - абсолютные координаты верхнего левого угла региона, из которого взяты ocr_blocks.
    """

    handicap_match = re.search(r'(\(([-+]?\d+(?:\.\d+)?)\))', target_outcome)
    if not handicap_match:
        telegram_log(f"[HANDICAP_CLICK_HELPER] Не удалось извлечь значение форы из target_outcome: {target_outcome}", is_debug_message=True)
        return False

    handicap_display_str = handicap_match.group(1).strip() # e.g., "(-1.0)"
    handicap_value_float_str = handicap_match.group(2).strip() # e.g., "-1.0"

    telegram_log(f"[HANDICAP_CLICK_HELPER] Tesseract ищет: Display='{handicap_display_str}' (Value='{handicap_value_float_str}')", is_debug_message=True)

    Y_LINE_TOLERANCE_COEF = 15
    X_COEF_SEARCH_RANGE = 300
    TEAM_COLUMN_SEARCH_Y_DIFF = 150 # Minimum Y difference to look for a column header above (Increased)
    TEAM_COLUMN_X_TOLERANCE = 70 # X-tolerance for a handicap block to fall under a team column header (Increased)


    # Create flexible patterns for precise matching of the handicap string/value
    # Use re.escape for the value part and allow optional spaces around it
    # Modified to allow broader search within blocks, not just full match
    flexible_display_pattern = create_flexible_pattern(handicap_display_str)
    flexible_numeric_value_only_pattern = create_flexible_pattern(handicap_value_float_str)


    potential_handicap_blocks = []
    for idx, block in enumerate(ocr_blocks):
        block_text = block[1] # Original text from Tesseract
        block_text_cleaned_for_match = block_text.lower().replace(' ', '').replace(',', '.').replace('—', '-')

        # Check if the handicap display string (e.g., "(-1.0)") is present in the block
        # Using re.search instead of re.fullmatch for more flexibility within blocks
        if re.search(flexible_display_pattern, block_text_cleaned_for_match):
            potential_handicap_blocks.append({"block": block, "idx": idx, "priority": 1, "match_type": "display_contained"})
            telegram_log(f"[DEBUG_CLICK_HELPER] Tesseract found DISPLAY pattern (P1): '{block_text}' (cleaned:'{block_text_cleaned_for_match}') contains '{handicap_display_str}'", is_debug_message=True)
            continue

        # Check if just the numeric value (e.g., "-1.0") is present in the block
        if re.search(flexible_numeric_value_only_pattern, block_text_cleaned_for_match):
            potential_handicap_blocks.append({"block": block, "idx": idx, "priority": 2, "match_type": "value_contained"})
            telegram_log(f"[DEBUG_CLICK_HELPER] Tesseract found NUMERIC VALUE pattern (P2): '{block_text}' (cleaned:'{block_text_cleaned_for_match}') contains '{handicap_value_float_str}'", is_debug_message=True)
            continue

        # --- ИЗМЕНЕНИЕ: Убираем избыточное логирование для "NOT match" ---
        # telegram_log(f"[DEBUG_CLICK_HELPER] Tesseract did NOT match: '{block_text}' (cleaned:'{block_text_cleaned_for_match}') with any handicap full-match patterns for '{handicap_display_str}'", is_debug_message=True)

    if not potential_handicap_blocks:
        telegram_log(f"[HANDICAP_CLICK_HELPER] Потенциальные блоки форы для '{handicap_display_str}' не найдены Tesseract'ом.", is_debug_message=True)
        return False

    # Sort by priority (P1 preferred over P2), then by Y-coordinate, then by X-coordinate
    potential_handicap_blocks.sort(key=lambda item: (item["priority"], item["block"][0][0][1], item["block"][0][0][0]))

    # Now, from potential handicap blocks, find the corresponding coefficient and click
    for item in potential_handicap_blocks:
        block_handicap_display = item["block"]
        idx_handicap_block = item["idx"] # Fix: Was using `item["idx"]` here previously for the loop instead of `handicap_item["idx"]`
        x_block_handicap_display, y_block_handicap_display = block_handicap_display[0][0][0], block_handicap_display[0][0][1]
        block_width = block_handicap_display[0][1][0] - block_handicap_display[0][0][0]
        block_height = block_handicap_display[0][2][1] - block_handicap_display[0][0][1]

        telegram_log(f"[HANDICAP_CLICK_HELPER] Обработка потенциального блока форы: '{block_handicap_display[1]}'. Ищем коэффициент и команду. Match Type: {item['match_type']}.", is_debug_message=True)

        # --- ИЗМЕНЕНИЕ: Новая логика поиска команды-заголовка столбца ---
        if team_name_target:
            team_found_for_this_handicap = False
            
            # Identify the horizontal range of the handicap block for column alignment
            handicap_x_center = x_block_handicap_display + block_width / 2
            
            # Search upwards for team column headers in the entire OCR block list (not just before current block)
            # We search from the beginning to find headers which are usually at the top
            for i_team_check, check_block_team in enumerate(ocr_blocks):
                y_check_block_team = check_block_team[0][0][1]
                x_check_block_team_min = check_block_team[0][0][0]
                x_check_block_team_max = check_block_team[0][1][0]
                team_block_text = check_block_team[1]

                # Check if block is significantly above and within column X-range
                if y_check_block_team < y_block_handicap_display - TEAM_COLUMN_SEARCH_Y_DIFF:
                    if x_check_block_team_min - TEAM_COLUMN_X_TOLERANCE <= handicap_x_center <= x_check_block_team_max + TEAM_COLUMN_X_TOLERANCE:
                        # Check if this block fuzzy matches the target team name
                        if is_fuzzy_match(team_name_target, team_block_text):
                            # Exclude blocks that are pure numbers, e.g., '1' or '2' or '3.20'
                            if not re.fullmatch(r'^\d+(\.\d+)?$', team_block_text.strip().replace(',', '.')):
                                team_found_for_this_handicap = True
                                telegram_log(f"[HANDICAP_CLICK_HELPER] Команда '{team_name_target}' найдена ('{team_block_text}') как заголовок столбца для форы '{block_handicap_display[1]}'.", is_debug_message=True)
                                break # Found the column header, no need to search further up
                            else:
                                telegram_log(f"[HANDICAP_CLICK_HELPER] Блок '{team_block_text}' является числом, а не именем команды. Пропускаем.", is_debug_message=True)
                        else:
                            telegram_log(f"[HANDICAP_CLICK_HELPER] Блок '{team_block_text}' находится в колонке, но не соответствует команде '{team_name_target}'.", is_debug_message=True)
                # If we've passed the Y-range where headers might be, stop searching.
                elif y_check_block_team > y_block_handicap_display: # We're below the handicap block, no headers here.
                    break


            if not team_found_for_this_handicap:
                telegram_log(f"[HANDICAP_CLICK_HELPER] Команда '{team_name_target}' НЕ найдена как заголовок столбца для форы '{block_handicap_display[1]}'. Пропускаем эту форы.", is_debug_message=True)
                continue # Try next potential handicap block

        # 2. Extract Coefficient from the block (if it contains both handicap and coef)
        block_text_for_coef_extraction = block_handicap_display[1].replace(' ', '').replace(',', '.')
        
        # Try to find the handicap display string followed by a coefficient in the same block
        # e.g., "(-1.0)1.615"
        handicap_and_coef_pattern = create_flexible_pattern(handicap_display_str).strip('?') + r'(\d+(?:\.\d+)?)$' # Remove trailing '?' from flexible pattern for stricter search
        coef_match_inline = re.search(handicap_and_coef_pattern, block_text_for_coef_extraction)

        if coef_match_inline:
            coef_text = coef_match_inline.group(1) # Use group(1) for the captured coefficient
            # Use the center of the found handicap block for clicking
            block_center_x = x_block_handicap_display + block_width / 2
            block_center_y = y_block_handicap_display + block_height / 2

            pyautogui.click(int(block_center_x) + region_offset_x, int(block_center_y) + region_offset_y)
            telegram_log(f"[КЛИК][HANDICAP_CLICK_HELPER][INLINE_COEF] Кликнута фора '{block_handicap_display[1]}' со встроенным коэф. '{coef_text}' по ({int(block_center_x) + region_offset_x}, {int(block_center_y) + region_offset_y}).", is_debug_message=True)
            time.sleep(0.5)
            return True

        # If not inline, look for the coefficient in adjacent blocks
        telegram_log(f"[HANDICAP_CLICK_HELPER][DEBUG] Ищем соседний коэффициент для блока: '{block_handicap_display[1]}'.", is_debug_message=True)
        found_coef_adjacent = False
        for j in range(idx_handicap_block + 1, len(ocr_blocks)):
            next_block = ocr_blocks[j]
            x_next_block, y_next_block = next_block[0][0][0], next_block[0][0][1]

            if abs(y_next_block - y_block_handicap_display) < Y_LINE_TOLERANCE_COEF and \
               x_next_block > x_block_handicap_display and \
               (x_next_block - (x_block_handicap_display + block_width)) < X_COEF_SEARCH_RANGE:

                if re.match(r'^\d+(?:\.\d+)?$', next_block[1].replace(',', '.')):
                    coef_block_width = next_block[0][1][0] - next_block[0][0][0]
                    coef_block_height = next_block[0][2][1] - next_block[0][0][1]

                    coef_block_center_x = x_next_block + coef_block_width / 2
                    coef_block_center_y = y_next_block + coef_block_height / 2

                    pyautogui.click(int(coef_block_center_x) + region_offset_x, int(coef_block_center_y) + region_offset_y)
                    telegram_log(f"[КЛИК][HANDICAP_CLICK_HELPER][ADJ_COEF] Кликнута фора '{block_handicap_display[1]}' с соседним коэф. '{next_block[1]}' по ({int(coef_block_center_x) + region_offset_x}, {int(coef_block_center_y) + region_offset_y}).", is_debug_message=True)
                    time.sleep(0.5)
                    return True
                # else: # Removed this debug log for less verbosity
            elif y_next_block - y_block_handicap_display > Y_LINE_TOLERANCE_COEF * 2:
                break # Not on the same logical line anymore
            elif x_next_block - (x_block_handicap_display + block_width) > X_COEF_SEARCH_RANGE:
                break # Too far horizontally

    telegram_log(f"[HANDICAP_CLICK_HELPER] Не удалось найти и кликнуть по исходу для '{target_outcome}' даже после обработки потенциальных блоков. Действующий коэффициент не найден.", is_debug_message=True)
    return False


def optimized_search_for_outcome(expected_text, outcome_search_region, max_scroll_iterations=10, difference_threshold=30, match_name=None):
    """
    Новая версия функции поиска исхода, которая объединяет OCR-блоки.
    Координаты, возвращаемые OCR, уже масштабированы обратно к исходному размеру скриншота.
    """
    x1, y1, x2, y2 = outcome_search_region
    region_width = x2 - x1
    region_height = y2 - y1
    expected = expected_text.lower().strip()

    def get_combined_top_left(results_list, start_idx, length):
        xs = []
        ys = []
        for offset in range(length):
            block_vertices = results_list[start_idx + offset][0]
            for (vx, vy) in block_vertices:
                xs.append(vx)
                ys.append(vy)
        return (x1 + min(xs), y1 + min(ys))

    last_screenshot_path = None
    last_full_text = None
    last_results = None

    for iteration in range(max_scroll_iterations):
        current_screenshot_pil = pyautogui.screenshot(region=(x1, y1, region_width, region_height))
        time.sleep(1)

        # --- ИЗМЕНЕНИЕ: Добавляем проверку DEBUG_SCREENSHOT перед отправкой ---
        if DEBUG_SCREENSHOT:
            debug_outcome_path = f"debug_outcome_screenshot_{iteration+1}.png"
            current_screenshot_pil.save(debug_outcome_path)
            for chat_id in subscribers:
                send_photo(chat_id, debug_outcome_path, caption=f"Скриншот поиска исхода, итерация {iteration+1}")

        full_text, results = get_ocr_results(current_screenshot_pil)
        # --- ИЗМЕНЕНИЕ: Ограничиваем количество блоков для логирования ---
        ocr_debug = f"[OCR] Итерация {iteration+1}\nFull text:\n{full_text}\nBlocks (first 10):\n" + "\n".join([str(r) for r in results[:10]])
        if len(results) > 10:
            ocr_debug += f"\n... (ещё {len(results)-10} блоков)"
        for chat_id in subscribers:
            send_message(chat_id, ocr_debug[:4000])

        last_screenshot_path = debug_outcome_path
        last_full_text = full_text
        last_results = results

        n = len(results)
        for i in range(n):
            candidate = results[i][1].strip().lower()
            if not expected.startswith(candidate):
                continue

            current_combined = candidate
            if current_combined == expected:
                coords = get_combined_top_left(results, i, 1)
                matched_text = results[i][1].strip()
                telegram_log(f"[DEBUG] Найден исход в одном блоке: '{current_combined}'. Координаты: {coords}", is_debug_message=True)
                return coords, matched_text

            for j in range(i + 1, min(i + 4, n)):
                next_block = results[j][1].strip().lower()
                potential = current_combined + " " + next_block
                if expected.startswith(potential):
                    current_combined = potential
                    if current_combined == expected:
                        coords = get_combined_top_left(results, i, j - i + 1)
                        original_text_fragments = [
                            results[k][1].strip() for k in range(i, j + 1)
                        ]
                        matched_text = " ".join(original_text_fragments)
                        telegram_log(f"[DEBUG] Найден исход путём объединения блоков {i}-{j}: '{current_combined}'. Координаты: {coords}", is_debug_message=True)
                        return coords, matched_text
                else:
                    break

        pyautogui.scroll(-4)
        time.sleep(1)

    telegram_log("[ERROR] Не удалось найти исход после прокрутки. Отправляю скриншот и сырые OCR-данные для диагностики.")
    # --- ИЗМЕНЕНИЕ: Добавляем проверку DEBUG_SCREENSHOT перед отправкой ---
    if DEBUG_SCREENSHOT and last_screenshot_path:
        for chat_id in subscribers:
            send_photo(chat_id, last_screenshot_path, caption="Скриншот последней попытки поиска исхода")
    if last_full_text is not None and last_results is not None:
        ocr_debug = f"[OCR] Последняя попытка\nFull text:\n{last_full_text}\nBlocks (first 10):\n" + "\n".join([str(r) for r in last_results[:10]])
        if len(last_results) > 10:
            ocr_debug += f"\n... (ещё {len(last_results)-10} блоков)"
        for chat_id in subscribers:
            send_message(chat_id, ocr_debug[:4000])
    return None, None

# ======================= Координаты и константы для ставок =======================

# Удалены неиспользуемые BET_INPUT_CANDIDATES_SET1/2

TARGET_COLOR = (218, 218, 218)
COLOR_TOLERANCE = 4

COEFFICIENT_SCREENSHOT_SHIFT_Y = 150
COEFFICIENT_SCREENSHOT_PADDING_X = 250
COEFFICIENT_SCREENSHOT_PADDING_BOTTOM = 80

FIRST_CLICK_COEF_REGION = (1000, 400, 300, 100)

def check_yellow_in_region(region, r_tolerance=15, g_tolerance=15, b_tolerance=15):
    """
    Проверяет, является ли ХОТЯ БЫ ОДИН из нескольких пикселей (центр, углы, середины сторон)
    указанного региона жёлтым.
    Ожидаемые значения RGB для желтого ~ (255, 207, 0).
    """
    try:
        x, y, width, height = region
        if width < 5 or height < 5:
             telegram_log(f"[WARN_YELLOW_REGION] Регион {region} слишком мал для множественной проверки.", is_debug_message=True)
             return False

        screenshot = pyautogui.screenshot(region=region)

        offset = 3
        points_to_check = [
            (width // 2, height // 2),
            (offset, offset),
            (width - 1 - offset, offset),
            (offset, height - 1 - offset),
            (width - 1 - offset, height - 1 - offset),
            (offset, height // 2),
            (width - 1 - offset, height // 2),
        ]

        target_r, target_g, target_b = 255, 207, 0
        found_yellow = False

        for px, py in points_to_check:
            try:
                pixel = screenshot.getpixel((px, py))
                r, g, b = pixel[:3]
                telegram_log(f"[DEBUG_YELLOW_REGION] Проверка точки ({px},{py}) в регионе {region}: RGB({r},{g},{b})", is_debug_message=True)

                is_yellow_here = (
                    abs(r - target_r) <= r_tolerance and
                    abs(g - target_g) <= g_tolerance and
                    abs(b - target_b) <= b_tolerance
                )

                if is_yellow_here:
                    telegram_log(f"[DEBUG_YELLOW_REGION] Желтый цвет ОБНАРУЖЕН в точке ({px},{py}). Регион считается желтым.")
                    found_yellow = True
                    break
            except IndexError:
                 telegram_log(f"[WARN_YELLOW_REGION] Не удалось получить пиксель в точке ({px},{py}) региона {region}. Пропускаем точку.")
            except Exception as point_e:
                 telegram_log(f"[ERROR_YELLOW_REGION] Неожиданная ошибка при проверке точки ({px},{py}): {point_e}")
                 continue

        if not found_yellow:
            telegram_log(f"[DEBUG_YELLOW_REGION] Желтый цвет НЕ обнаружен ни в одной из проверенных точек региона {region}.", is_debug_message=True)

        return found_yellow

    except Exception as e:
        telegram_log(f"[ERROR] Ошибка при проверке желтого цвета в регионе {region}: {e}")
        return False

# Удалена неиспользуемая функция check_yellow_pixel

def check_coefficient_condition(found_coef, condition_str):
    """
    Проверяет, удовлетворяет ли найденный коэффициент (found_coef) условиям,
    заданным в строке condition_str. Пример условия: ">1.1", "<3", ">1.1 <4" или просто "1.5".
    """
    tokens = condition_str.split()
    valid = True
    for token in tokens:
        token = token.strip()
        if token.startswith(">"):
            try:
                threshold = float(token[1:])
                if not (found_coef >= threshold):
                    valid = False
            except ValueError: # Changed to ValueError for float conversion
                valid = False
        elif token.startswith("<"):
            try:
                threshold = float(token[1:])
                if not (found_coef <= threshold):
                    valid = False
            except ValueError: # Changed to ValueError for float conversion
                valid = False
        else:
            try:
                exact_value = float(token)
                if not (found_coef == exact_value):
                    valid = False
            except ValueError: # Changed to ValueError for float conversion
                valid = False
    return valid

def extract_coefficient_from_region(region, max_retries=3, retry_delay=1):
    """
    Делает скриншот заданной области и через OCR извлекает число.
    Используется для финальной проверки коэффициента в купоне.
    Добавлена логика повторных попыток при ошибках API.
    """
    screenshot = pyautogui.screenshot(region=region)

    image_to_process = screenshot

    # --- ИЗМЕНЕНИЕ: Добавляем проверку DEBUG_SCREENSHOT перед отправкой ---
    if DEBUG_SCREENSHOT:
        debug_coef_path = "debug_coef_screenshot.png"
        image_to_process.save(debug_coef_path)
        for chat_id in subscribers:
            send_photo(chat_id, debug_coef_path, caption=f"Финальный скрин коэффициента (область {region})")

    time.sleep(1)

    full_text = ""
    for attempt in range(max_retries):
        telegram_log(f"[DEBUG] Попытка {attempt + 1}/{max_retries} вызова OCR API для области {region}...", is_debug_message=True)
        try:
            full_text, _ = get_ocr_results(image_to_process)
            telegram_log(f"[DEBUG] OCR API успешно вернул результат для области {region} (попытка {attempt + 1}).", is_debug_message=True)
            matches = re.findall(r"\b\d+(?:\.\d+)?\b", full_text)
            if matches:
                coef_str = matches[0].replace(",", ".")
                try:
                    coefficient = float(coef_str)
                    return coefficient
                except ValueError: # Changed to ValueError for float conversion
                    telegram_log(f"[ERROR] Ошибка преобразования OCR результата '{coef_str}' в число: {e}", is_debug_message=True)
                    return None
            else:
                telegram_log(f"[WARNING] Коэффициент не найден в OCR тексте для области {region}. Full text: '{full_text}'", is_debug_message=True)
                if attempt < max_retries -1:
                    telegram_log(f"[DEBUG] Ожидание {retry_delay} сек перед следующей попыткой...", is_debug_message=True)
                    time.sleep(retry_delay)
                else:
                    telegram_log(f"[ERROR] Все {max_retries} попыток вызова OCR API не удались, коэффициент не найден.", is_debug_message=False)
                    return None
        except Exception as e:
            telegram_log(f"[ERROR] Попытка {attempt + 1}/{max_retries} вызова OCR API для области {region} не удалась: {e}", is_debug_message=True)
            if attempt < max_retries - 1:
                telegram_log(f"[DEBUG] Ожидание {retry_delay} сек перед следующей попыткой...", is_debug_message=True)
                time.sleep(retry_delay)
            else:
                telegram_log(f"[ERROR] Все {max_retries} попыток вызова OCR API не удались.", is_debug_message=False)
                return None

    return None

def parse_coefficient_from_text(text):
    """
    Извлекает первое число вида XX или XX.XX из строки text (например, "Barcelona 28.3").
    Возвращает float или None.
    """
    matches = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    if matches:
        coef_str = matches[0].replace(",", ".")
        try:
            return float(coef_str)
        except ValueError: # Changed to ValueError for float conversion
            return None
    return None

def find_bet_input_coords(timeout=15, color_tolerance_ready=20, color_tolerance_white=5): # УВЕЛИЧЕН ДОПУСК color_tolerance_ready до 20
    """
    Ищет координаты поля для ввода суммы ставки ПОСЛЕ перехода на главную.
    1. Ждет появления индикатора готовности (зеленый цвет).
    2. Если индикатор найден, проверяет 2 точки на наличие белого цвета.
    Возвращает кортеж (координаты, тип_кандидата) или (None, None).
    Тип кандидата: "primary" или "secondary".
    """
    # --- ИЗМЕНЕНИЕ: Требуется актуализация по новому скриншоту купона! ---
    # Эти координаты, вероятно, неверны, так как цвет не совпал.
    # QA, пожалуйста, предоставьте скриншот полного экрана в момент появления купона.
    READY_CHECK_COORDS = (1250, 249) # Цвет: (6, 136, 69)
    TARGET_READY_COLOR = (6, 136, 69)
    CHECK_REGION_SIZE = 5

    BET_INPUT_PRIMARY = (1199, 319)
    BET_INPUT_SECONDARY = (1199, 336)
    TARGET_WHITE_COLOR = (255, 255, 255)

    start_time = time.time()
    telegram_log(f"[DEBUG] Ожидание индикатора готовности поля ввода (цвет {TARGET_READY_COLOR} в {READY_CHECK_COORDS})...", is_debug_message=True)

    ready_indicator_found = False
    debug_ss_counter = 0
    while time.time() - start_time < timeout:
        region_ready = (READY_CHECK_COORDS[0], READY_CHECK_COORDS[1], CHECK_REGION_SIZE, CHECK_REGION_SIZE)
        try:
            screenshot_ready = pyautogui.screenshot(region=region_ready)
            stat_ready = ImageStat.Stat(screenshot_ready)
            avg_color_ready = tuple(int(c) for c in stat_ready.mean)

            telegram_log(f"[DEBUG] Проверка индикатора готовности: {READY_CHECK_COORDS}, текущий цвет {avg_color_ready}", is_debug_message=True)

            # --- ИЗМЕНЕНИЕ: Добавляем проверку DEBUG_SCREENSHOT перед отправкой ---
            if DEBUG_SCREENSHOT:
                debug_ss_counter += 1
                debug_path_ready = f"debug_ready_check_region_{debug_ss_counter}.png"
                screenshot_ready.save(debug_path_ready)
                for chat_id in subscribers:
                    send_photo(chat_id, debug_path_ready, caption=f"Debug: Ready check region {READY_CHECK_COORDS}, Color: {avg_color_ready}, Iter: {debug_ss_counter}")

            if all(abs(avg_color_ready[i] - TARGET_READY_COLOR[i]) <= color_tolerance_ready for i in range(3)):
                telegram_log(f"[DEBUG] Индикатор готовности найден ({avg_color_ready}). Проверяю белые пиксели...", is_debug_message=True)
                ready_indicator_found = True
                break
        except Exception as e:
             telegram_log(f"[ERROR] Ошибка при проверке цвета индикатора готовности: {e}")
        time.sleep(1)

    if not ready_indicator_found:
        telegram_log(f"[ERROR] Таймаут ожидания индикатора готовности (цвет {TARGET_READY_COLOR} не появился в {READY_CHECK_COORDS} за {timeout} сек).")
        return None, None

    def check_white_pixel(coords_to_check, label):
        region_white = (coords_to_check[0], coords_to_check[1], CHECK_REGION_SIZE, CHECK_REGION_SIZE)
        try:
            screenshot_white = pyautogui.screenshot(region=region_white)
            stat_white = ImageStat.Stat(screenshot_white)
            avg_color_white = tuple(int(c) for c in stat_white.mean)
            telegram_log(f"[DEBUG] Проверка белого пикселя ({label}): {coords_to_check}, текущий цвет {avg_color_white}", is_debug_message=True)
            if all(abs(avg_color_white[i] - TARGET_WHITE_COLOR[i]) <= color_tolerance_white for i in range(3)):
                telegram_log(f"[DEBUG] Белый пиксель найден у {label} ({coords_to_check}).", is_debug_message=True)
                return True
        except Exception as e:
            telegram_log(f"[ERROR] Ошибка при проверке белого пикселя у {label} ({coords_to_check}): {e}")
        return False

    if check_white_pixel(BET_INPUT_PRIMARY, "primary"):
        return BET_INPUT_PRIMARY, "primary"

    time.sleep(0.5)
    if check_white_pixel(BET_INPUT_SECONDARY, "secondary"):
        return BET_INPUT_SECONDARY, "secondary"

    telegram_log(f"[ERROR] Белый пиксель не найден ни у primary ({BET_INPUT_PRIMARY}), ни у secondary ({BET_INPUT_SECONDARY}).")
    return None, None

def parse_halftime_handicap_outcome_new(outcome_str: str, match_name: Optional[str] = None) -> Optional[dict]:
    """
    Парсит детали ставки на фору в тайме.
    Извлекает: идентификатор тайма, имя команды, значение форы.
    Примеры: "Таймы Победа с учетом форы 1-й тайм Берое (+1.0)"
             "Таймы Фора 2-й тайм Реал Мадрид (-0.5)"
    """
    outcome_lower = outcome_str.lower()
    details = {
        "half_identifier": None,
        "team_name": None,
        "handicap_value": None,
        "handicap_display": None,
        "original_outcome": outcome_str,
        "base_type": "фора"
    }

    match_specific_half = re.search(r'(\d+)-(?:й|ого|го|му|м)\s+тайм(?:а|у|е|ом)?', outcome_lower)
    if match_specific_half:
        half_num = match_specific_half.group(1)
        details["half_identifier"] = f"{half_num}-й тайм"
    elif "тайм" in outcome_lower:
        details["half_identifier"] = "тайм"
    else:
        telegram_log(f"[PARSE_HT_HANDICAP_NEW] Идентификатор тайма не найден в: {outcome_str}", is_debug_message=True)
        return None

    match_handicap = re.search(r'(\(([-+]?\d+(?:\.\d+)?)\))', outcome_str)
    if match_handicap:
        details["handicap_display"] = match_handicap.group(1)
        details["handicap_value"] = match_handicap.group(2)
    else:
        telegram_log(f"[PARSE_HT_HANDICAP_NEW] Значение форы не найдено в: {outcome_str}", is_debug_message=True)
        return None

    if details["half_identifier"]:
        half_id_pattern_for_split = re.escape(details["half_identifier"].split('-')[0] if '-' in details["half_identifier"] else details["half_identifier"]) + r"-(?:й|ого|го|му|м)\s+тайм(?:а|у|е|ом)?"
        if details["half_identifier"] == "тайм" and not match_specific_half :
             half_id_pattern_for_split = r"тайм"

        m_half_loc = re.search(half_id_pattern_for_split, outcome_str, re.IGNORECASE)
        idx_half_end = -1
        if m_half_loc:
            idx_half_end = m_half_loc.end()

        idx_handicap_start = outcome_str.find(details["handicap_display"])

        if idx_half_end != -1 and idx_handicap_start != -1 and idx_half_end < idx_handicap_start:
            text_between = outcome_str[idx_half_end:idx_handicap_start].strip()
            text_between_cleaned = text_between
            if "победа с учетом форы" in text_between_cleaned.lower():
                 text_between_cleaned = text_between_cleaned.lower().replace("победа с учетом форы", "").strip() # Fixed typo
            if "фора" in text_between_cleaned.lower():
                text_between_cleaned = text_between_cleaned.lower().replace("фора", "").strip()

            if text_between_cleaned:
                details["team_name"] = ' '.join(word.capitalize() for word in text_between_cleaned.split())

    if not details["team_name"] and match_name:
        teams_in_match = [t.strip().lower() for t in match_name.split('-')]
        found_teams_in_outcome = []
        for team_in_match in teams_in_match:
            if team_in_match in outcome_lower:
                found_teams_in_outcome.append(team_in_match)

        if len(found_teams_in_outcome) == 1:
            details["team_name"] = ' '.join(word.capitalize() for word in found_teams_in_outcome[0].split())
        elif len(found_teams_in_outcome) > 1:
            telegram_log(f"[PARSE_HT_HANDICAP_NEW] Найдено несколько команд из матча в исходе: {found_teams_in_outcome}. Невозможно однозначно определить команду.", is_debug_message=True)

    if not details["team_name"]:
        telegram_log(f"[PARSE_HT_HANDICAP_NEW] Имя команды не удалось извлечь для: {outcome_str}. Поиск будет без явного указания команды.", is_debug_message=True)

    if "победа с учетом форы" in outcome_lower:
        details["base_type"] = "победа с учетом форы"
    elif "фора" in outcome_lower:
        details["base_type"] = "фора"

    telegram_log(f"[PARSE_HT_HANDICAP_NEW] Распарсено: {details}", is_debug_message=True)
    return details

def find_halftime_handicap_and_click_new(outcome_str: str, match_name: Optional[str] = None, max_scrolls: int = 7):
    """
    Находит и кликает по исходу форы в тайме.
    1. Парсит outcome_str для извлечения деталей (тайм, команда, фора, тип).
    2. В цикле со скроллом:
        a. Делает скриншот и OCR.
        b. Ищет основной заголовок "Таймы".
        c. Под ним ищет подзаголовок конкретного тайма (например, "1-й тайм").
        d. В секции этого тайма ищет строку с командой (если указана) и нужной форой.
        e. Ищет коэффициент рядом с форой (в том же блоке или в соседнем).
        f. Если найдено - кликает и возвращает True.
    3. Если не найдено после всех скроллов - возвращает False.
    """
    telegram_log(f"[HT_HANDICAP_NEW] Попытка обработки: {outcome_str}, Матч: {match_name}", is_debug_message=True)
    parsed_details = parse_halftime_handicap_outcome_new(outcome_str, match_name)

    if not parsed_details:
        telegram_log(f"[HT_HANDICAP_NEW] Не удалось распарсить детали для: {outcome_str}", is_debug_message=True)
        return False

    half_identifier_search = parsed_details["half_identifier"]
    team_name_search = parsed_details["team_name"]
    handicap_display_search = parsed_details["handicap_display"]

    Y_LINE_TOLERANCE_COEF = 20

    OUTCOME_SEARCH_REGION = (206, 151, 958, 641)
    x1_region, y1_region, x2_region, y2_region = OUTCOME_SEARCH_REGION
    region_width = x2_region - x1_region
    region_height = y2_region - y1_region

    for scroll_iter in range(max_scrolls):
        screenshot = pyautogui.screenshot(region=(x1_region, y1_region, region_width, region_height))
        time.sleep(0.5)

        # --- ИЗМЕНЕНИЕ: Добавляем проверку DEBUG_SCREENSHOT перед отправкой ---
        if DEBUG_SCREENSHOT:
            debug_screenshot_path = f"debug_halftime_handicap_scroll_{scroll_iter+1}.png"
            screenshot.save(debug_screenshot_path)
            for chat_id in subscribers:
                send_photo(chat_id, debug_screenshot_path,
                             caption=f"HT Handicap Scroll {scroll_iter+1} for: {outcome_str}")

        try:
            full_text, ocr_blocks = get_ocr_results(screenshot)
            if not full_text and not ocr_blocks and scroll_iter < max_scrolls -1 :
                 telegram_log(f"[ERROR][HT_TOTAL] OCR вернул пустые значения. Скролл {scroll_iter+1}. Пропускаем к следующему скроллу.", is_debug_message=True)
                 pyautogui.scroll(-4)
                 time.sleep(1)
                 continue

            # --- ИЗМЕНЕНИЕ: Ограничиваем количество блоков для логирования ---
            ocr_debug_msg = (f"[OCR][HT_HANDICAP][SCROLL {scroll_iter+1}] Ищем: {outcome_str}\n"
                           f"Parsed: Half='{half_identifier_search}', Team='{team_name_search}', Handicap='{handicap_display_search}'\n"
                           f"Blocks (first 10): \n" + "\n".join([f'{b[1]} @ {b[0]}' for b in ocr_blocks[:10]]))
            if len(ocr_blocks) > 10:
                ocr_debug_msg += "\n..."
            for chat_id in subscribers:
                send_message(chat_id, ocr_debug_msg[:4000])
        except Exception as e_ocr_section:
            telegram_log(f"[ERROR][HT_HANDICAP] Исключение во время OCR или отправки OCR лога для скролла {scroll_iter+1}: {e_ocr_section}", is_debug_message=True)
            import traceback
            tb_str = traceback.format_exc()
            telegram_log(f"[TRACEBACK_OCR_HT_HANDICAP]:\n{tb_str[:3500]}", is_debug_message=True)
            if scroll_iter < max_scrolls -1 :
                pyautogui.scroll(-4)
                time.sleep(1)
                continue
            else:
                telegram_log(f"[FAIL][HT_HANDICAP] Ошибка OCR на последнем скролле. Прерывание поиска для {outcome_str}.", is_debug_message=True)
                return False

        active_half_section_indicator_block = None
        y_after_active_half_indicator = -1

        half_num_pattern = ''
        if '1' in half_identifier_search:
            half_num_pattern = r'1(?:-?й|st|nd|rd|th)?\s+тайм'
        elif '2' in half_identifier_search:
            half_num_pattern = r'2(?:-?й|st|nd|rd|th)?\s+тайм'

        candidate_section_headers = []
        if half_num_pattern:
            for block in ocr_blocks:
                block_text_lower = block[1].lower()
                if re.search(half_num_pattern, block_text_lower):
                    if "тотал" in block_text_lower:
                        candidate_section_headers.append({"block": block, "priority": 1, "y": block[0][0][1]})
                        telegram_log(f"[DEBUG] Кандидат заголовка (приоритет 1): '{block[1]}' @ {block[0]}", is_debug_message=True)
                    else:
                        candidate_section_headers.append({"block": block, "priority": 2, "y": block[0][0][1]})
                        telegram_log(f"[DEBUG] Кандидат заголовка (приоритет 2): '{block[1]}' @ {block[0]}", is_debug_message=True)

        if candidate_section_headers:
            candidate_section_headers.sort(key=lambda b: b["y"])
            active_half_section_indicator_block = candidate_section_headers[0]["block"]
            telegram_log(f"Выбран индикатор секции тайма: '{active_half_section_indicator_block[1]}' @ {active_half_section_indicator_block[0]}", is_debug_message=True)
        elif half_identifier_search == "тайм":
            for block in ocr_blocks:
                if block[1].lower().strip() == "таймы":
                    active_half_section_indicator_block = block
                    telegram_log(f"Найден общий заголовок 'Таймы' для half_identifier_search='тайм': '{block[1]}' @ {block[0]}", is_debug_message=True)
                    break

        if active_half_section_indicator_block:
             y_after_active_half_indicator = active_half_section_indicator_block[0][2][1]
        else:
             telegram_log(f"[HT_HANDICAP_NEW] Не удалось определить начальную Y координату секции. Скролл {scroll_iter+1}.", is_debug_message=True)
             pyautogui.scroll(-4)
             time.sleep(1)
             continue

        potential_handicap_blocks = []
        for idx, block in enumerate(ocr_blocks):
            if block[0][0][1] >= y_after_active_half_indicator:
                # Use the strict flexible patterns defined globally
                block_text_cleaned_for_match = block[1].lower().replace(' ', '').replace(',', '.').replace('—', '-')
                if re.fullmatch(create_flexible_pattern(handicap_display_search), block_text_cleaned_for_match) or \
                   re.fullmatch(create_flexible_pattern(handicap_display_search.replace('(', '').replace(')', '')), block_text_cleaned_for_match):
                    potential_handicap_blocks.append({"block": block, "idx": idx, "y": block[0][0][1]})

        if not potential_handicap_blocks:
            telegram_log(f"[HT_HANDICAP_NEW] Блоки с форой '{handicap_display_search}' не найдены ниже Y={y_after_active_half_indicator}. Скролл {scroll_iter+1}.", is_debug_message=True)
            pyautogui.scroll(-4)
            time.sleep(1)
            continue

        potential_handicap_blocks.sort(key=lambda item: item["y"])

        for handicap_item in potential_handicap_blocks:
            block_handicap_display = handicap_item["block"]
            idx_handicap_block = handicap_item["idx"]
            y_handicap_block = handicap_item["y"]
            x_handicap_block = block_handicap_display[0][0][0]

            if team_name_search:
                team_found_for_this_handicap = False
                # Search upwards for the team name in the entire OCR block list (not just before current block)
                for i_team_check, check_block_team in enumerate(ocr_blocks):
                    y_check_block_team = check_block_team[0][0][1]
                    x_check_block_team_min = check_block_team[0][0][0]
                    x_check_block_team_max = check_block_team[0][1][0]
                    
                    # Check if block is significantly above and within column X-range
                    if y_check_block_team < y_handicap_block - TEAM_COLUMN_SEARCH_Y_DIFF and \
                       x_check_block_team_min - TEAM_COLUMN_X_TOLERANCE <= (x_handicap_block + block_width/2) <= x_check_block_team_max + TEAM_COLUMN_X_TOLERANCE:
                        
                        if is_fuzzy_match(team_name_search, check_block_team[1]):
                            # Exclude blocks that are pure numbers
                            if not re.fullmatch(r'^\d+(\.\d+)?$', check_block_team[1].strip().replace(',', '.')):
                                team_found_for_this_handicap = True
                                telegram_log(f"[HT_HANDICAP_NEW] Команда '{team_name_search}' найдена ('{check_block_team[1]}') как заголовок столбца для форы '{block_handicap_display[1]}'.", is_debug_message=True)
                                break
                            else:
                                telegram_log(f"[HT_HANDICAP_NEW] Блок '{check_block_team[1]}' является числом, а не именем команды. Пропускаем.", is_debug_message=True)
                        else:
                            telegram_log(f"[HT_HANDICAP_NEW] Блок '{check_block_team[1]}' находится в колонке, но не соответствует команде '{team_name_search}'.", is_debug_message=True)
                # If we've passed the Y-range where headers might be, stop searching.
                elif y_check_block_team > y_handicap_block: # We're below the handicap block, no headers here.
                    break

                # No need to check for i_team_check >= idx_handicap_block, as we are iterating from beginning
                # and checking y_check_block_team < y_handicap_block already filters out blocks below/on the same line.

            if not team_found_for_this_handicap:
                telegram_log(f"[HT_HANDICAP_NEW] Команда '{team_name_search}' НЕ найдена как заголовок столбца для форы '{block_handicap_display[1]}'. Пропускаем эту форы.", is_debug_message=True)
                continue # Try next potential handicap block

            text_in_handicap_block = block_handicap_display[1].replace(" ", "")
            handicap_text_no_space = handicap_display_search.replace(" ", "")

            # If the block contains the handicap string and potentially the coefficient (inline)
            # Use re.search instead of re.fullmatch and extract coefficient from the rest of the string
            # Also, add an escaped version of handicap_display_str in case of Tesseract adding spaces inside.
            handicap_pattern_in_block = re.search(create_flexible_pattern(handicap_display_search), text_in_handicap_block)
            
            if handicap_pattern_in_block:
                remaining_text_after_handicap = text_in_handicap_block[handicap_pattern_in_block.end():].strip()
                coef_match_inline = re.match(r'^\d+(?:\.\d+)?$', remaining_text_after_handicap.replace(',', '.'))
                
                if coef_match_inline:
                    coef_text = coef_match_inline.group(0) # Use group(0) for the whole match
                    block_center_x = x_block_handicap_display + block_width / 2
                    block_center_y = y_block_handicap_display + block_height / 2

                    pyautogui.click(int(block_center_x) + region_offset_x, int(block_center_y) + region_offset_y)
                    telegram_log(f"[КЛИК][HT_HANDICAP_NEW] Кликнута фора '{block_handicap_display[1]}' со встроенным коэф. '{coef_text}' по ({int(block_center_x) + region_offset_x}, {int(block_center_y) + region_offset_y}).", is_debug_message=True)
                    time.sleep(0.5)
                    return True

            # If not inline, look for the coefficient in adjacent blocks
            telegram_log(f"[HT_HANDICAP_NEW][DEBUG] Ищем соседний коэффициент для блока: '{block_handicap_display[1]}'.", is_debug_message=True)
            found_coef_adjacent = False
            for j in range(idx_handicap_block + 1, len(ocr_blocks)):
                next_block = ocr_blocks[j]
                x_next_block, y_next_block = next_block[0][0][0], next_block[0][0][1]

                if abs(y_next_block - y_handicap_block) < Y_LINE_TOLERANCE_COEF and \
                   x_next_block > x_handicap_block and \
                   (x_next_block - (x_handicap_block + block_width)) < X_COEF_SEARCH_RANGE:

                    if re.match(r'^\d+(?:\.\d+)?$', next_block[1].replace(',', '.')):
                        coef_block_width = next_block[0][1][0] - next_block[0][0][0]
                        coef_block_height = next_block[0][2][1] - next_block[0][0][1]

                        coef_block_center_x = x_next_block + coef_block_width / 2
                        coef_block_center_y = y_next_block + coef_block_height / 2

                        pyautogui.click(int(coef_block_center_x) + region_offset_x, int(coef_block_center_y) + region_offset_y)
                        telegram_log(f"[КЛИК][HT_HANDICAP_NEW] Кликнута фора '{block_handicap_display[1]}' с соседним коэф. '{next_block[1]}' по ({int(coef_block_center_x) + region_offset_x}, {int(coef_block_center_y) + region_offset_y}).", is_debug_message=True)
                        time.sleep(0.5)
                        return True
                    # else: # Removed this debug log for less verbosity
                elif y_next_block - y_handicap_block > Y_LINE_TOLERANCE_COEF * 2:
                    break # Not on the same logical line anymore
                elif x_next_block - (x_handicap_block + block_width) > X_COEF_SEARCH_RANGE:
                    break # Too far horizontally

    telegram_log(f"[HT_HANDICAP_NEW][FAIL] Не удалось найти исход: {outcome_str} после {max_scrolls} скроллов.", is_debug_message=True)
    return False

def find_outcome(match_name: str, outcome: str, coef_condition: str, bet_amount: float):
    global chosen_candidate
    PREDEFINED_OUTCOME_COORDS = {
        "1": (483, 570),
        "X": (543, 570),
        "2": (606, 570)
    }
    OUTCOME_SEARCH_REGION = (206, 151, 958, 641) # Global search region for OCR
    FINISH_COORDS = (1160, 597) # Assuming this is still correct for "Place Bet" button
    RETRY_COORDS = (1254, 363) # Assuming this is still correct for a "clear bet" or "retry" button

    outcome_successfully_clicked = False

    # Переход на страницу матча для сложных исходов
    if outcome.strip().lower() not in ["1", "x", "2"]:
        telegram_log(f"[DEBUG][INFO] Сложный исход '{outcome}'. Требуется переход на страницу матча.", is_debug_message=True)
        match_page_click_coords = (277, 526)
        telegram_log(f"[DEBUG][INFO] Клик по {match_page_click_coords} для перехода на страницу матча.", is_debug_message=True)
        try:
            pyautogui.click(match_page_click_coords[0], match_page_click_coords[1])
            time.sleep(5)
        except Exception as e_click_match_page:
            telegram_log(f"[ERROR] Ошибка при клике для перехода на страницу матча: {e_click_match_page}")
            telegram_log("[NAVIGATE] Ставка не обработана (ошибка перехода на стр. матча). Переход на главную страницу...")
            home_coords = (70, 142)
            pyautogui.click(home_coords[0], home_coords[1])
            time.sleep(3)
            return False
    else:
        telegram_log(f"[DEBUG][INFO] Простой исход '{outcome}'. Обработка на текущей странице результатов поиска.", is_debug_message=True)


    if outcome in PREDEFINED_OUTCOME_COORDS:
        coords = PREDEFINED_OUTCOME_COORDS[outcome]
        telegram_log(f"Предопределённый исход '{outcome}' найден. Координаты для клика: {coords}", is_debug_message=True)
        pyautogui.click(coords[0], coords[1])
        time.sleep(2)
        outcome_successfully_clicked = True
    elif "тайм" in outcome.lower() and ("фора" in outcome.lower() or "победа с учетом форы" in outcome.lower()):
        telegram_log(f"Обнаружен исход halftime handicap: '{outcome}'. Использую find_halftime_handicap_and_click_new.", is_debug_message=True)
        if find_halftime_handicap_and_click_new(outcome, match_name=match_name):
            time.sleep(2)
            outcome_successfully_clicked = True
        else:
            telegram_log(f"[ERROR] Halftime handicap исход '{outcome}' не найден или не обработан!")
    # TODO: Implement find_halftime_total_and_click_new
    # elif "тайм" in outcome.lower() and "тотал" in outcome.lower():
    #     telegram_log(f"Обнаружен исход halftime total: '{outcome}'. Использую find_halftime_total_and_click_new.", is_debug_message=True)
    #     if find_halftime_total_and_click_new(outcome):
    #         time.sleep(2)
    #         outcome_successfully_clicked = True
    #     else:
    #         telegram_log(f"[ERROR] Halftime total исход '{outcome}' не найден!")
    # TODO: Implement find_total_and_click_coef_team_new
    # elif "тотал голов (" in outcome.lower():
    #     telegram_log(f"Обнаружен исход team total: '{outcome}'. Использую find_total_and_click_coef_team_new.", is_debug_message=True)
    #     if find_total_and_click_coef_team_new(outcome):
    #         time.sleep(2)
    #         outcome_successfully_clicked = True
    #     else:
    #         telegram_log("[ERROR] Тотал по команде не найден!")
    # TODO: Implement find_total_and_click_coef_new
    # elif "тотал" in outcome.lower():
    #     telegram_log(f"Обнаружен общий исход total: '{outcome}'. Использую find_total_and_click_coef_new.", is_debug_message=True)
    #     if find_total_and_click_coef_new(outcome):
    #         time.sleep(2)
    #         outcome_successfully_clicked = True
    #     else:
    #         telegram_log("[ERROR] Тотал не найден!")
    elif ("фора" in outcome.lower() or "победа с учетом форы" in outcome.lower()):
        telegram_log(f"[HYBRID_FOR] Использую гибридный режим (Mistral+Tesseract) для исхода: '{outcome}'", is_debug_message=True)
        search_text = outcome.strip()
        if find_handicap_hybrid_click_new(search_text=search_text, match_name=match_name, max_scrolls=7, OUTCOME_SEARCH_REGION=OUTCOME_SEARCH_REGION):
            time.sleep(2)
            outcome_successfully_clicked = True
        else:
            telegram_log("[HYBRID_FOR][ERROR] Фора не найдена гибридным методом!")
    else:
        outcome = outcome.strip()
        found_coords, recognized_outcome_text = optimized_search_for_outcome(
            outcome,
            OUTCOME_SEARCH_REGION,
            max_scroll_iterations=10,
            difference_threshold=30,
            match_name=match_name
        )
        if found_coords is not None:
            if found_coords == (0, 0): # This case might occur if some previous logic wrongly sets 0,0 for "already clicked"
                telegram_log(f"[DEBUG] Клик по исходу уже был совершен, пропускаю повторный клик.", is_debug_message=True)
            else:
                telegram_log(f"[DEBUG] Исход '{outcome}' найден по координатам: {found_coords}", is_debug_message=True)
                pyautogui.click(found_coords[0], found_coords[1])
                time.sleep(2)
            outcome_successfully_clicked = True
        else:
            telegram_log("[ERROR] Исход не найден!")

    if not outcome_successfully_clicked:
        telegram_log("[NAVIGATE] Ставка не обработана (исход не найден или не удалось кликнуть). Переход на главную страницу...")
        home_coords = (70, 142)
        pyautogui.click(home_coords[0], home_coords[1])
        time.sleep(3)
        return False

    # --- Логика размещения ставки (теперь выполняется после выбора исхода) ---

    chosen_candidate_coords, candidate_type = find_bet_input_coords()

    if chosen_candidate_coords is None:
        telegram_log("[ERROR] Не найдено место для ввода ставки (после проверки индикатора и белых пикселей).")
        telegram_log("[NAVIGATE] Ставка не обработана (Место ввода не найдено). Переход на главную страницу...")
        home_coords = (70, 142)
        pyautogui.click(home_coords[0], home_coords[1])
        time.sleep(3)
        return False

    chosen_candidate = chosen_candidate_coords
    telegram_log(f"[DEBUG] Выбраны координаты поля ввода: {chosen_candidate_coords} (тип: {candidate_type})", is_debug_message=True)

    pyautogui.click(chosen_candidate_coords[0], chosen_candidate_coords[1], clicks=2)
    pyautogui.write(str(bet_amount), interval=0.1)
    time.sleep(2)
    time.sleep(0.5)

    coef_region_x_base = 1085
    coef_region_y_base = 310
    coef_region_width_val = 35
    coef_region_height_val = 20
    y_offset = 17 if candidate_type == "secondary" else 0

    final_coef_region = (
        coef_region_x_base,
        coef_region_y_base + y_offset,
        coef_region_width_val,
        coef_region_height_val
    )
    telegram_log(f"[DEBUG] Область для финальной проверки коэффициента (тип: {candidate_type}, смещение Y: {y_offset}): {final_coef_region}", is_debug_message=True)

    try:
        screenshot_coef = pyautogui.screenshot(region=final_coef_region)
        # --- ИЗМЕНЕНИЕ: Добавляем проверку DEBUG_SCREENSHOT перед отправкой ---
        if DEBUG_SCREENSHOT:
            debug_coef_path = "debug_coef_screenshot.png"
            screenshot_coef.save(debug_coef_path)
            for chat_id in subscribers:
                send_photo(chat_id, debug_coef_path, caption=f"Финальный скрин коэффициента (область {final_coef_region})")
    except Exception as e:
        telegram_log(f"[ERROR] Ошибка при создании скриншота коэффициента: {e}")
        pyautogui.scroll(300)
        time.sleep(0.5)
        pyautogui.click(RETRY_COORDS[0], RETRY_COORDS[1])
        telegram_log("Ошибка скриншота коэффициента. Ожидание новой ставки.")
        telegram_log("[NAVIGATE] Ставка не обработана (Ошибка скриншота кэфа). Переход на главную страницу...")
        home_coords = (70, 142)
        pyautogui.click(home_coords[0], home_coords[1])
        time.sleep(3)
        return False

    time.sleep(1)
    found_coef = extract_coefficient_from_region(final_coef_region)

    if found_coef is not None:
        telegram_log(f"[DEBUG] Извлечённый коэффициент (финальная проверка): {found_coef}", is_debug_message=True)
        if not check_coefficient_condition(found_coef, coef_condition):
            telegram_log(f"[ERROR] Коэффициент {found_coef} НЕ соответствует условию '{coef_condition}'. Отмена ставки.")
            pyautogui.scroll(300)
            time.sleep(0.5)
            pyautogui.click(RETRY_COORDS[0], RETRY_COORDS[1])
            telegram_log("Коэффициент (финальная проверка) не соответствует условиям. Ожидание новой ставки.")
            telegram_log("[NAVIGATE] Ставка не обработана (Кэф не соответствует). Переход на главную страницу...")
            home_coords = (70, 142)
            pyautogui.click(home_coords[0], home_coords[1])
            time.sleep(3)
            return False
        else:
            telegram_log(f"[DEBUG] Коэффициент {found_coef} соответствует условию '{coef_condition}'. Выполняем Tab/Enter с проверкой желтого пикселя.")
    else:
        telegram_log("[ERROR] Не удалось распознать коэффициент (финальная проверка) после всех попыток. Отмена ставки.")
        time.sleep(0.5)
        pyautogui.click(RETRY_COORDS[0], RETRY_COORDS[1])
        telegram_log("Не удалось распознать коэффициент (финальная проверка). Ожидание новой ставки.")
        telegram_log("[NAVIGATE] Ставка не обработана (OCR кэфа не удался). Переход на главную страницу...")
        home_coords = (70, 142)
        pyautogui.click(home_coords[0], home_coords[1])
        time.sleep(3)
        return False

    try:
        telegram_log(f"[DEBUG_CONFIRM_TAB] Проверка желтого цвета в регионе коэффициента {final_coef_region} для определения Tab-последовательности.", is_debug_message=True)

        if not chosen_candidate_coords:
            telegram_log("[ERROR_CONFIRM_TAB] chosen_candidate_coords не определены перед Tab/Enter!")
            raise ValueError("chosen_candidate_coords is None before Tab/Enter")

        if check_yellow_in_region(final_coef_region):
            telegram_log("[DEBUG_CONFIRM_TAB] Желтый цвет ОБНАРУЖЕН в регионе коэффициента (КЭФ изменился).", is_debug_message=True)
            telegram_log(f"[DEBUG_CONFIRM_TAB] Шаг 1: Клик в поле ввода {chosen_candidate_coords}")
            pyautogui.click(chosen_candidate_coords[0], chosen_candidate_coords[1])
            time.sleep(0.3)
            telegram_log("[DEBUG_CONFIRM_TAB] Шаг 2: Tab (1 раз) -> Enter", is_debug_message=True)
            pyautogui.press('tab')
            time.sleep(0.3)
            pyautogui.press('enter')
            time.sleep(1)

            telegram_log(f"[DEBUG_CONFIRM_TAB] Шаг 3: Клик в поле ввода {chosen_candidate_coords}")
            pyautogui.click(chosen_candidate_coords[0], chosen_candidate_coords[1])
            time.sleep(0.3)
            telegram_log("[DEBUG_CONFIRM_TAB] Шаг 4: Tab (2 раза) -> Enter", is_debug_message=True)
            pyautogui.press('tab')
            time.sleep(0.2)
            pyautogui.press('tab')
            time.sleep(0.3)
            pyautogui.press('enter')
        else:
            telegram_log("[DEBUG_CONFIRM_TAB] Желтый цвет НЕ ОБНАРУЖЕН в регионе коэффициента (КЭФ не изменился).")
            telegram_log(f"[DEBUG_CONFIRM_TAB] Шаг 1: Клик в поле ввода {chosen_candidate_coords}")
            pyautogui.click(chosen_candidate_coords[0], chosen_candidate_coords[1])
            time.sleep(0.3)
            telegram_log("[DEBUG_CONFIRM_TAB] Шаг 2: Tab (2 раза) -> Enter")
            pyautogui.press('tab')
            time.sleep(0.2)
            pyautogui.press('tab')
            time.sleep(0.3)
            pyautogui.press('enter')

        telegram_log(f"Ставка подтверждена (условная Tab/Enter sequence): Исход={outcome}, Сумма={bet_amount}.")

    except Exception as e:
        telegram_log(f"[ERROR_CONFIRM_TAB] Ошибка во время условной Tab/Enter последовательности: {e}")
        time.sleep(0.5)
        pyautogui.click(RETRY_COORDS[0], RETRY_COORDS[1])
        telegram_log("Ошибка при подтверждении ставки (условный Tab/Enter). Ожидание новой ставки.")
        telegram_log("[NAVIGATE] Ставка не обработана (Ошибка Tab/Enter). Переход на главную страницу...")
        home_coords = (70, 142)
        pyautogui.click(home_coords[0], home_coords[1])
        time.sleep(3)
        return False

    max_attempts = 10
    attempt = 0
    region_to_check = (549, 422, 2, 2)
    target_color = (7, 151, 77)
    tolerance = int(255 * 0.02)
    color_found = False
    while attempt < max_attempts:
        region_screenshot = pyautogui.screenshot(region=region_to_check)
        stat = ImageStat.Stat(region_screenshot)
        avg_color = tuple(int(c) for c in stat.mean)
        telegram_log(f"[DEBUG] Проверка региона {region_to_check} (попытка {attempt+1}/{max_attempts}): {avg_color}", is_debug_message=True)
        if all(abs(avg_color[i] - target_color[i]) <= tolerance for i in range(3)):
            telegram_log("Пари принято спасибо")
            color_found = True

            confirmation_screenshot_region = (398, 336, 835 - 398, 494 - 336)
            telegram_log(f"[DEBUG] Делаем подтверждающий скриншот области {confirmation_screenshot_region}...", is_debug_message=True)
            try:
                # --- ИЗМЕНЕНИЕ: Добавляем проверку DEBUG_SCREENSHOT перед отправкой ---
                if DEBUG_SCREENSHOT:
                    conf_ss = pyautogui.screenshot(region=confirmation_screenshot_region)
                    conf_ss_path = "bet_final_confirmation_screenshot.png"
                    conf_ss.save(conf_ss_path)
                    telegram_log(f"[DEBUG] Подтверждающий скриншот сохранен: {conf_ss_path}", is_debug_message=True)
                    for chat_id in subscribers:
                        send_photo(chat_id, conf_ss_path, caption="Пари принято (подтверждающий скриншот)")

                telegram_log("[DEBUG] Клик по кнопке закрытия окна подтверждения (549, 422).", is_debug_message=True)
                pyautogui.click(549, 422)
                time.sleep(0.5)
                telegram_log("Ставка успешно обработана!")

            except Exception as ss_err:
                telegram_log(f"[ERROR] Не удалось сделать/отправить подтверждающий скриншот или кликнуть после: {ss_err}")
            break
        time.sleep(0.5)
        attempt += 1

    if color_found:
        telegram_log("[NAVIGATE] Ставка успешно обработана. Переход на главную страницу...")
        home_coords = (70, 142)
        pyautogui.click(home_coords[0], home_coords[1])
        time.sleep(3)
        return True
    else:
        telegram_log(f"[ERROR] ПОДТВЕРЖДЕНИЕ НЕ ПОЛУЧЕНО: Целевой цвет {target_color} не найден в {region_to_check} после {max_attempts} попыток. Ставка считается НЕУСПЕШНОЙ.")

        cancel_target_color = (203, 33, 42)
        cancel_color_tolerance = 15

        first_cancel_coords = (423, 459)
        telegram_log(f"[CANCEL_CHECK_1] Проверка первой кнопки отмены: координаты {first_cancel_coords}, цвет {cancel_target_color}", is_debug_message=True)
        try:
            screenshot_first_cancel = pyautogui.screenshot(region=(first_cancel_coords[0] - 1, first_cancel_coords[1] - 1, 3, 3))
            pixel_first_cancel = screenshot_first_cancel.getpixel((1, 1))
            r1, g1, b1 = pixel_first_cancel[:3]
            telegram_log(f"[CANCEL_CHECK_1] Цвет в {first_cancel_coords}: RGB({r1},{g1},{b1})", is_debug_message=True)

            if (abs(r1 - cancel_target_color[0]) <= cancel_color_tolerance and
                abs(g1 - cancel_target_color[1]) <= cancel_color_tolerance and
                abs(b1 - cancel_target_color[2]) <= cancel_color_tolerance):

                telegram_log(f"[CANCEL_ACTION_1] Обнаружен цвет {cancel_target_color} в {first_cancel_coords}. Клик для предварительной отмены.", is_debug_message=True)
                pyautogui.click(first_cancel_coords[0], first_cancel_coords[1])
                time.sleep(1)
                telegram_log("Выполнен клик по предварительной кнопке отмены.")
            else:
                telegram_log(f"[CANCEL_CHECK_1] Цвет {cancel_target_color} не найден в {first_cancel_coords}. Предварительная отмена не выполнена.", is_debug_message=True)
        except Exception as e_first_cancel:
            telegram_log(f"[CANCEL_CHECK_1][ERROR] Ошибка при проверке/клике по первой кнопке отмены: {e_first_cancel}", is_debug_message=True)

        second_cancel_coords = (1017, 491)
        telegram_log(f"[CANCEL_CHECK_2] Проверка второй кнопки отмены: координаты {second_cancel_coords}, цвет {cancel_target_color}", is_debug_message=True)

        try:
            screenshot_second_cancel = pyautogui.screenshot(region=(second_cancel_coords[0] - 1, second_cancel_coords[1] - 1, 3, 3))
            pixel_second_cancel = screenshot_second_cancel.getpixel((1, 1))
            r2, g2, b2 = pixel_second_cancel[:3]
            telegram_log(f"[CANCEL_CHECK_2] Цвет в {second_cancel_coords}: RGB({r2},{g2},{b2})", is_debug_message=True)

            if (abs(r2 - cancel_target_color[0]) <= cancel_color_tolerance and
                abs(g2 - cancel_target_color[1]) <= cancel_color_tolerance and
                abs(b2 - cancel_target_color[2]) <= cancel_color_tolerance):

                telegram_log(f"[CANCEL_ACTION_2] Обнаружен цвет {cancel_target_color} в {second_cancel_coords}. Клик для удаления ставки.", is_debug_message=True)
                pyautogui.click(second_cancel_coords[0], second_cancel_coords[1])
                time.sleep(1)
                telegram_log("Ставка удалена (клик по кнопке отмены).")
            else:
                telegram_log(f"[CANCEL_CHECK_2] Цвет {cancel_target_color} не найден в {second_cancel_coords}. Автоматическое удаление ставки не выполнено.", is_debug_message=True)
        except Exception as e_second_cancel:
            telegram_log(f"[CANCEL_CHECK_2][ERROR] Ошибка при проверке/удалении ставки: {e_second_cancel}", is_debug_message=True)

        new_cancel_coords = (1024, 463)
        telegram_log(f"[CANCEL_CHECK_3] Проверка третьей кнопки отмены: координаты {new_cancel_coords}, цвет {cancel_target_color}", is_debug_message=True)
        try:
            screenshot_new_cancel = pyautogui.screenshot(region=(new_cancel_coords[0] - 1, new_cancel_coords[1] - 1, 3, 3))
            pixel_new_cancel = screenshot_new_cancel.getpixel((1, 1))
            r3, g3, b3 = pixel_new_cancel[:3]
            telegram_log(f"[CANCEL_CHECK_3] Цвет в {new_cancel_coords}: RGB({r3},{g3},{b3})", is_debug_message=True)

            if (abs(r3 - cancel_target_color[0]) <= cancel_color_tolerance and
                abs(g3 - cancel_target_color[1]) <= cancel_color_tolerance and
                abs(b3 - cancel_target_color[2]) <= cancel_color_tolerance):

                telegram_log(f"[CANCEL_ACTION_3] Обнаружен цвет {cancel_target_color} в {new_cancel_coords}. Клик для удаления ставки.", is_debug_message=True)
                pyautogui.click(new_cancel_coords[0], new_cancel_coords[1])
                time.sleep(1)
                telegram_log("Ставка удалена (клик по третьей кнопке отмены).")
            else:
                telegram_log(f"[CANCEL_CHECK_3] Цвет {cancel_target_color} не найден в {new_cancel_coords}. Удаление ставки (3) не выполнено.", is_debug_message=True)
        except Exception as e_new_cancel:
            telegram_log(f"[CANCEL_CHECK_3][ERROR] Ошибка при проверке/клике по третьей кнопке отмены: {e_new_cancel}", is_debug_message=True)

        telegram_log("[NAVIGATE] Ставка не обработана (не подтверждена). Переход на главную страницу...")
        home_coords = (70, 142)
        pyautogui.click(home_coords[0], home_coords[1])
        time.sleep(3)
        return False

# Removed finalize_bet_and_navigate as its logic is now embedded in find_outcome's success path.

def send_instructions():
    """
    Отправляет инструкцию по использованию бота всем подписчикам
    """
    instructions = """
🎯 *Инструкция по использованию бота*

Для отправки ставки используйте следующий формат:
`Название матча, Исход, Коэффициент, Сумма`

Где:
• *Название матча* - точное название матча из Live раздела
• *Исход* - тип ставки (например: "1", "X", "2" или полное название исхода)
• *Коэффициент* - условие для коэффициента:
  - Точное значение: "1.5"
  - Больше: ">1.5"
  - Меньше: "<2.0"
  - Диапазон: ">1.5 <2.0"
• *Сумма* - сумма ставки (число)

📝 *Примеры запросов:*
`Barcelona - Real Madrid, 1, >1.5, 100`
`Juventus - Milan, Тотал больше 2.5, >1.8, 50`
`Liverpool - Arsenal, X, 3.2, 75`
`Команда А - Команда Б, Фора Команда А (-1.5), >1.9, 100`
`Команда В - Команда Г, Таймы Фора 1-й тайм Команда В (+0.5), <2.1, 50`

⚠️ *Важно:*
- Разделяйте параметры запятыми
- Указывайте точное название матча как в Live
- Проверяйте правильность формата перед отправкой
"""
    for chat_id in subscribers:
        send_message(chat_id, instructions)

def parse_total_outcome_new(outcome, _): # Removed unused ocr_blocks argument
    """
    Парсит исход для тотала:
    - Извлекает значение тотала
    - Определяет тип (больше/меньше)
    Возвращает (base_type, total_value) или (None, None)
    """
    m = re.search(r'\(([-+]?\d+(?:\.\d+)?)\)', outcome)
    if not m:
        telegram_log(f"[TOTAL_NEW] Не найдено значение тотала в исходе: {outcome}")
        return None, None

    total_value = m.group(1)

    outcome_lower = outcome.lower()
    if "больше" in outcome_lower:
        base_type = "больше"
    elif "меньше" in outcome_lower:
        base_type = "меньше"
    else:
        telegram_log(f"[TOTAL_NEW] Не определен тип тотала (больше/меньше) в исходе: {outcome}")
        return None, None

    return base_type, total_value

# find_total_outcome_table_new, find_total_coef_candidates_new,
# find_total_and_click_coef_new, parse_total_team_from_outcome_new,
# find_total_and_click_coef_team_new are currently commented out in find_outcome,
# so their implementations are not critical for this specific fix.
# They should be reviewed for similar OCR precision issues if uncommented later.
# For now, keeping them as-is.

def send_ocr_diagnostics_telegram(
    scroll_iter: int,
    search_type: str,
    search_desc: str,
    screenshot_path: str,
    full_text: str,
    ocr_blocks: list,
    candidates: Optional[List[str]] = None,
    found: bool = False,
    extra: str = ""
):
    """
    Отправляет расширенную диагностику поиска исхода в Telegram:
    - скриншот
    - описание поиска
    - полный текст OCR
    - список блоков
    - найденные кандидаты (если есть)
    - статус (найдено/не найдено)
    """
    header = f"[DIAG][{search_type.upper()}][SCROLL {scroll_iter + 1}] {search_desc}"
    # --- ИЗМЕНЕНИЕ: Добавляем проверку DEBUG_SCREENSHOT перед отправкой фото ---
    if DEBUG_SCREENSHOT:
        for chat_id in subscribers:
            send_photo(chat_id, screenshot_path, caption=header)
    msg = header + f"\nFull OCR text:\n{full_text}\n\nOCR blocks (first 10):\n" # --- ИЗМЕНЕНИЕ: Ограничиваем количество блоков ---
    msg += "\n".join([str(b) for b in ocr_blocks[:10]])
    if len(ocr_blocks) > 10: # --- ИЗМЕНЕНИЕ: Ограничиваем количество блоков ---
        msg += f"\n... (ещё {len(ocr_blocks)-10} блоков)"
    if candidates is not None:
        msg += f"\n\nКандидаты: {candidates}"
    if extra:
        msg += f"\n{extra}"
    msg += f"\nСтатус: {'НАЙДЕНО' if found else 'НЕ НАЙДЕНО'}"
    for chat_id in subscribers:
        send_message(chat_id, msg[:4000])


def find_handicap_hybrid_click_new(search_text: str, match_name: str, max_scrolls: int, OUTCOME_SEARCH_REGION: Tuple[int, int, int, int]) -> bool:
    x1_region, y1_region, x2_region, y2_region = OUTCOME_SEARCH_REGION
    region_width = x2_region - x1_region
    region_height = y2_region - y1_region

    handicap_match = re.search(r'(\(([-+]?\d+(?:\.\d+)?)\))', search_text)
    if not handicap_match:
        telegram_log(f"[HYBRID_FOR][ERROR] Не удалось извлечь значение форы из search_text: {search_text}", is_debug_message=True)
        return False
    handicap_display_from_input = handicap_match.group(1).strip() # e.g., "(-1.0)"
    handicap_value_from_input = handicap_match.group(2).strip()  # e.g., "-1.0"

    team_name_target = None
    match_teams = [t.strip().lower() for t in match_name.split('-')]
    for team_part in match_teams:
        if team_part in search_text.lower():
            team_name_target = team_part
            break

    # Prepare patterns for matching handicap value from Mistral's output
    # (Mistral part is mostly for confirmation, Tesseract for bbox)
    abs_handicap_value_str = str(abs(float(handicap_value_from_input)))
    # Pattern for (X.X) or X.X or -X.X or +X.X allowing optional spaces inside/around
    loose_handicap_value_pattern_mistral = re.compile(
        r'[\(\s]*[-+]?\s*' + re.escape(abs_handicap_value_str.replace('.', r'\.')) + r'\s*[\)\s]*'
    )
    handicap_display_normalized_for_search_mistral = handicap_display_from_input.lower().replace(" ", "").replace(",", ".")

    for scroll_iter in range(max_scrolls):
        telegram_log(f"[HYBRID_FOR][SCROLL {scroll_iter+1}] Делаем скриншот и выполняем OCR.", is_debug_message=True)
        current_full_screenshot = pyautogui.screenshot(region=(x1_region, y1_region, region_width, region_height))

        # --- Mistral OCR for broad check and section finding ---
        full_text_mistral, _ = extract_text_mistral_ocr(current_full_screenshot)
        full_text_mistral_lower = full_text_mistral.lower()
        full_text_mistral_cleaned_for_match = full_text_mistral_lower.replace(" ", "").replace(",", ".").replace("—", "-")

        handicap_type_present_mistral = "форы" in full_text_mistral_cleaned_for_match or \
                                        "победасучетомфоры" in full_text_mistral_cleaned_for_match or \
                                        "гандикап" in full_text_mistral_cleaned_for_match or \
                                        "handicap" in full_text_mistral_cleaned_for_match

        handicap_value_present_mistral = \
            handicap_display_normalized_for_search_mistral in full_text_mistral_cleaned_for_match or \
            loose_handicap_value_pattern_mistral.search(full_text_mistral_cleaned_for_match) is not None

        team_present_mistral = False
        team_name_for_mistral_check = "Н/Д"
        if team_name_target:
            if is_fuzzy_match(team_name_target, full_text_mistral_lower):
                team_present_mistral = True
                team_name_for_mistral_check = team_name_target

        telegram_log(f"[HYBRID_FOR][SCROLL {scroll_iter+1}] Результат Mistral OCR: Тип форы: {handicap_type_present_mistral}, Команда: {team_present_mistral} ('{team_name_for_mistral_check}'), Значение форы: {handicap_value_present_mistral} ('{handicap_display_from_input}').", is_debug_message=True)

        # Condition for running Tesseract: Mistral must broadly confirm the presence.
        if handicap_value_present_mistral and (not team_name_target or team_present_mistral) and handicap_type_present_mistral:
            telegram_log(f"[HYBRID_FOR][SCROLL {scroll_iter+1}] Mistral conditions met. Preparing Tesseract for precise coordinates.", is_debug_message=True)

            # --- ИЗМЕНЕНИЕ: Временно отключаем обрезку по заголовку для отладки таблиц ---
            # Run Tesseract on the *full* current screenshot
            full_text_tesseract, tesseract_blocks = extract_text_tesseract(current_full_screenshot, **BEST_TESSERACT_PARAMS)
            
            # Since we're not cropping based on header for now, these offsets are just the main region offsets
            tesseract_region_offset_x = x1_region
            tesseract_region_offset_y = y1_region

            # --- ИЗМЕНЕНИЕ: Убираем логику обработки found_header_block, т.к. не обрезаем ---
            # This block is now unused, its logic for cropping is temporarily skipped.
            # However, we still want to log the "full" tesseract blocks and try to click.
            # This part will be revisited later if we re-enable smarter cropping.

            send_ocr_diagnostics_telegram(
                scroll_iter,
                "HYBRID_TESSERACT_BBOX",
                f"Tesseract BBox for: '{search_text}' (scroll {scroll_iter+1})",
                f"debug_hybrid_handicap_scroll_{scroll_iter+1}.png", # Still reference the full screenshot
                full_text_tesseract,
                tesseract_blocks, # Use all blocks for analysis in _click_handicap_from_blocks
                found=False,
                extra=f"OCR_PROVIDER: {OCR_PROVIDER}. Match Name: {match_name}. Parsed Team: {team_name_target}. Parsed Handicap: {handicap_display_from_input}"
            )

            # Now, call the clicking helper with the blocks and the correct offsets
            if _click_handicap_from_blocks(search_text, tesseract_blocks, team_name_target, tesseract_region_offset_x, tesseract_region_offset_y):
                telegram_log(f"[HYBRID_FOR][SCROLL {scroll_iter+1}] Исход '{search_text}' успешно найден и кликнут Tesseract'ом.", is_debug_message=True)
                return True
            else:
                telegram_log(f"[HYBRID_FOR][SCROLL {scroll_iter+1}] Исход '{search_text}' НЕ найден Tesseract'ом на этом экране, несмотря на подтверждение Mistral. Пробуем следующий скролл.", is_debug_message=True)
        else:
            telegram_log(f"[HYBRID_FOR][SCROLL {scroll_iter+1}] Условие Mistral не пройдено (Значение: {handicap_value_present_mistral}, Тип: {handicap_type_present_mistral}, Команда: {not team_name_target or team_present_mistral}). Tesseract не запускается. Продолжаем прокрутку.", is_debug_message=True)

        pyautogui.scroll(-4)
        time.sleep(1)

    telegram_log(f"[HYBRID_FOR][FAIL] Превышено максимальное количество прокруток ({max_scrolls}). Исход '{search_text}' не найден.", is_debug_message=True)
    return False


def main():
    load_subscribers()

    updater = threading.Thread(target=poll_updates, daemon=True)
    updater.start()

    telegram_log("🤖 Бот запущен! Отправьте /start, чтобы получать логи бота.")
    send_instructions()

    time.sleep(5)
    open_browser_and_navigate()
    time.sleep(10)

    SITE_READY_COLOR = (255, 255, 255)
    wait_for_site_ready_color(SITE_READY_COLOR, 10, (83, 652, 5, 5))

    do_login()
    time.sleep(5)

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()