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
#from mistralai.models.chat import TextChunk # ВОССТАНОВЛЕНО: УБРАН КОММЕНТАРИЙ

from typing import Optional, Tuple, List, Any, Dict

import pytesseract


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

# --- НАСТРОЙКИ OCR И ОТЛАДКИ ---
OCR_PROVIDER = "mistral"
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

DEBUG_LOGGING_ENABLED = True
# ------------------------

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

# ---------------------- Настройки Telegram-бота ----------------------
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    print("[ERROR] TELEGRAM_BOT_TOKEN не найден в переменных окружения!")

BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
SUBSCRIBERS_FILE = "subscribers.txt"
subscribers = set()

DEBUG_SCREENSHOT = True

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

BEST_TESSERACT_PARAMS = {
    'scale_factor': 3,
    'contrast_enhance': 1.8,
    'sharpness_enhance': 2.0,
    'adaptive_method': cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    'adaptive_block_size': 19,
    'adaptive_C': 5,
    'median_blur_kernel': 3
}


def preprocess_for_tesseract(pil_image: Image.Image,
                             scale_factor: int = 3,
                             contrast_enhance: float = 1.8,
                             sharpness_enhance: float = 2.0,
                             adaptive_method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             adaptive_block_size: int = 21,
                             adaptive_C: int = 5,
                             median_blur_kernel: int = 1) -> Image.Image:
    """
    Предобработка изображения для Tesseract OCR с настраиваемыми параметрами.
    """

    new_width = int(pil_image.width * scale_factor)
    new_height = int(pil_image.height * scale_factor)
    pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(contrast_enhance)
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(sharpness_enhance)

    open_cv_image = np.array(pil_image.convert('RGB'))
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 255, adaptive_method,
        cv2.THRESH_BINARY, adaptive_block_size, adaptive_C
    )
    telegram_log(f"[DEBUG_PREPROCESS] Выполнена адаптивная бинаризация (method={adaptive_method}, blockSize={adaptive_block_size}, C={adaptive_C}).", is_debug_message=True)

    if median_blur_kernel > 1:
        thresh = cv2.medianBlur(thresh, median_blur_kernel)

    return Image.fromarray(thresh)

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
                # >>> ИСПРАВЛЕНИЕ: Правильная обработка списка ContentChunk для объединения всех текстовых фрагментов
                if isinstance(content_result, list) and content_result:
                    all_text_chunks = []
                    for chunk in content_result:
                        # Ensure TextChunk is imported for this check
                        try:
                            # Safely import TextChunk only when needed, to avoid breaking if it's not present
                            # or if mistralai version is older.
                            from mistralai.models.chat import TextChunk
                            if isinstance(chunk, TextChunk): # Проверяем, является ли блок текстовым
                                all_text_chunks.append(chunk.text.strip())
                            elif isinstance(chunk, str): # Fallback for older versions or different content types
                                all_text_chunks.append(chunk.strip())
                        except ImportError: # If TextChunk not imported, treat as string
                            if isinstance(chunk, str):
                                all_text_chunks.append(chunk.strip())
                    full_text = "\n".join(all_text_chunks) # Объединяем все текстовые фрагменты
                elif isinstance(content_result, str):
                    full_text = content_result.strip()

            if full_text:
                telegram_log(f"[DEBUG_MISTRAL_API] Mistral OCR успешно (Попытка 1). Длина текста: {len(full_text)}", is_debug_message=True)
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


        # custom_config = r'--oem 1 --psm 3 -c tessedit_char_whitelist=0123456789().,-+АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъьэюяABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        custom_config = r'--oem 1 --psm 3' # ИЗМЕНЕНИЕ: Убран tessedit_char_whitelist для более гибкого распознавания

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

            vertices = [(x_original, y_original), (x_original + w_original, y_original), (x_original + w_original, y_original + h_original), (x_original, y_original + h_original)] # Исправлена последняя вершина
            results.append([vertices, text, conf])

        telegram_log(f"[DEBUG_TESSERACT_COORDS] Tesseract: Обратное масштабирование координат выполнено с scale_factor={scale_factor}. Пример первой координаты: (x_scaled={data['left'][0]}, y_scaled={data['top'][0]}) -> (x_original={results[0][0][0][0]}, y_original={results[0][0][0][1]})", is_debug_message=True)

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
        # Передаем найденные параметры в extract_text_tesseract
        return extract_text_tesseract(pil_image, **BEST_TESSERACT_PARAMS)
    # Если OCR_PROVIDER установлен в mistral, но клиент не инициализирован, откатываемся к tesseract
    elif OCR_PROVIDER == "mistral" and not mistral_client:
        print("[WARNING_OCR] OCR_PROVIDER set to mistral but client not available, falling back to Tesseract.")
        telegram_log("[WARNING_OCR] OCR_PROVIDER set to mistral but client not available, falling back to Tesseract.", is_debug_message=True)
        return extract_text_tesseract(pil_image, **BEST_TESSERACT_PARAMS)
    else:
        # Если ни один из провайдеров не выбран или не настроен корректно
        print("[ERROR_OCR] No OCR provider is available or initialized correctly.")
        telegram_log("[ERROR_OCR] No OCR provider is available or initialized correctly.", is_debug_message=False)
        return "", []

def score_ocr_result(full_text_ocr: str, ocr_blocks: List[List[Any]],
                     target_handicap_display: str, target_coef_value: str) -> float:
    """
    Оценивает качество OCR распознавания для заданного целевого текста.
    Возвращает числовой балл, чем больше, тем лучше.
    """
    score = 0.0
    full_text_lower = full_text_ocr.lower().replace(" ", "").replace(",", ".")

    target_handicap_lower = target_handicap_display.lower().replace(" ", "")
    if target_handicap_lower in full_text_lower:
        score += 1.0

    target_coef_lower = target_coef_value.lower().replace(",", ".")
    if target_coef_lower != "нет_коэф" and target_coef_lower in full_text_lower:
        score += 1.0

    for block_vertices, block_text, block_conf in ocr_blocks:
        block_text_lower = block_text.lower().replace(" ", "").replace(",", ".")
        if target_handicap_lower in block_text_lower:
            if target_coef_lower != "нет_коэф" and target_coef_lower in block_text_lower:
                score += 2.0
                score += block_conf / 100.0
                break

    found_handicap_block_conf = 0.0
    for block_vertices, block_text, block_conf in ocr_blocks:
        block_text_cleaned = block_text.replace(" ", "").replace(",", ".")
        if target_handicap_lower in block_text_cleaned:
            found_handicap_block_conf = max(found_handicap_block_conf, block_conf)
    score += found_handicap_block_conf / 200.0

    return score


def tune_tesseract_preprocessing(
    target_pil_image: Image.Image,
    target_ocr_text_samples: List[str],
    max_combinations: int = 5000
) -> Optional[Dict[str, Any]]:
    """
    Автоматически подбирает лучшие параметры предобработки для Tesseract OCR
    на основе распознавания целевых текстов.

    Возвращает словарь с лучшими параметрами или None, если ничего не найдено.
    """

    param_ranges = {
        'scale_factor': [3, 4],
        'contrast_enhance': [1.8, 2.5],
        'sharpness_enhance': [2.0, 2.5],
        'adaptive_method': [cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.ADAPTIVE_THRESH_MEAN_C],
        'adaptive_block_size': [5, 7, 9, 11, 13, 15, 17, 19, 21],
        'adaptive_C': [-10, -5, -3, 0, 3, 5, 7, 9, 10],
        'median_blur_kernel': [1, 3]
    }

    best_score = -1.0
    best_params = None

    import itertools
    all_combinations = list(itertools.product(*param_ranges.values()))

    telegram_log(f"[TUNER] Начинаю перебор {len(all_combinations)} комбинаций параметров. Это может занять время!", is_debug_message=False)

    for i, params_tuple in enumerate(all_combinations):
        if i >= max_combinations:
            telegram_log(f"[TUNER] Достигнуто максимальное количество комбинаций ({max_combinations}). Завершаю перебор.", is_debug_message=False)
            break

        current_params = {
            'scale_factor': params_tuple[0],
            'contrast_enhance': params_tuple[1],
            'sharpness_enhance': params_tuple[2],
            'adaptive_method': params_tuple[3],
            'adaptive_block_size': params_tuple[4],
            'adaptive_C': params_tuple[5],
            'median_blur_kernel': params_tuple[6]
        }

        if current_params['adaptive_block_size'] % 2 == 0 or current_params['adaptive_block_size'] <= 1:
            current_params['adaptive_block_size'] = current_params['adaptive_block_size'] + 1 if current_params['adaptive_block_size'] > 1 else 3

        try:
            processed_img = preprocess_for_tesseract(target_pil_image.copy(), **current_params)
            full_text, ocr_blocks = extract_text_tesseract(processed_img, **current_params)

            current_total_score_for_params = 0.0
            for target_sample_text in target_ocr_text_samples:
                handicap_match = re.search(r'(\(([-+]?\d+(?:\.\d+)?)\))', target_sample_text)
                if not handicap_match:
                    continue

                target_handicap_display_sample = handicap_match.group(1).strip()
                target_coef_value_sample = "НЕТ_КОЭФ"
                coef_search = re.search(r'([0-9]+(?:[.,][0-9]+)?)$', target_sample_text)
                if coef_search:
                    target_coef_value_sample = coef_search.group(1).replace(",", ".")

                current_total_score_for_params += score_ocr_result(full_text, ocr_blocks,
                                                                   target_handicap_display_sample, target_coef_value_sample)

            if current_total_score_for_params > best_score:
                best_score = current_total_score_for_params
                best_params = current_params
                telegram_log(f"[TUNER_BEST] Новые лучшие параметры! Total Score: {best_score:.2f}, Params: {best_params}", is_debug_message=True)

        except Exception as e:
            telegram_log(f"[TUNER_ERROR] Ошибка при тестировании параметров {current_params}: {e}", is_debug_message=True)
            continue

    if best_params:
        telegram_log(f"[TUNER_FINAL] Автоматическая настройка завершена. Лучший общий балл: {best_score:.2f}", is_debug_message=False)
        telegram_log(f"[TUNER_FINAL] Найдены лучшие параметры предобработки: {best_params}", is_debug_message=False)
    else:
        telegram_log(f"[TUNER_FINAL] Автоматическая настройка не нашла подходящих параметров.", is_debug_message=False)

    return best_params

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
            if t_word in ocr_words:
                matched_words_count += 1

        if matched_words_count >= len(target_words) / 2:
            return True

        if len(target_words) == 1 and len(target_words[0]) > 3:
            for o_word in ocr_words:
                if target_words[0] in o_word or o_word in target_words[0]:
                    return True

    return False

def _click_handicap_from_blocks(target_outcome: str, ocr_blocks: List[List[Any]], team_name_target: Optional[str], region_offset_x: int = 0, region_offset_y: int = 0) -> bool:
    """
    Находит и кликает по исходу форы в данном наборе OCR-блоков.
    Tesseract здесь фокусируется только на поиске конкретного значения форы и коэффициента.
    Координаты ocr_blocks уже масштабированы обратно к исходному размеру скриншота.
    # Удалено из docstring: Добавлена логика привязки к имени команды.
    """

    handicap_match = re.search(r'(\(([-+]?\d+(?:\.\d+)?)\))', target_outcome)
    if not handicap_match:
        telegram_log(f"[HANDICAP_CLICK_HELPER] Не удалось извлечь значение форы из target_outcome: {target_outcome}", is_debug_message=True)
        return False

    handicap_display_str = handicap_match.group(1).strip() # e.g., "(-1.0)"
    handicap_value_float = float(handicap_match.group(2)) # e.g., -1.0
    handicap_value_float_str = handicap_match.group(2).strip() # e.g., "-1.0"
    handicap_value_abs_str = str(abs(handicap_value_float)) # e.g., "1.0"

    telegram_log(f"[HANDICAP_CLICK_HELPER] Tesseract ищет: Value='{handicap_value_float}' (Display='{handicap_display_str}')", is_debug_message=True)

    Y_LINE_TOLERANCE_COEF = 15
    X_COEF_SEARCH_RANGE = 300

    # Define a character mapping for common OCR errors (simplified)
    # This map allows for typical misinterpretations by Tesseract
    OCR_CHAR_MAP = {
        '0': '[0o]', '1': '[1l]', '2': '[2z]', '3': '[3]', '4': '[4]',
        '5': '[5s]', '6': '[6]', '7': '[7]', '8': '[8]', '9': '[9]',
        '-': '[—\-a]', # Hyphen, em-dash, or 'a' (common for minus sign)
        '.': '[\.,]', # Dot or comma
        '(': '[\(]',
        ')': '[\)]',
        '+': '[\+]',
    }

    def create_flexible_pattern(input_string):
        """Creates a regex pattern from a string, allowing for common OCR errors and optional spaces."""
        pattern_chars = []
        for char in input_string.lower().replace(' ', ''): # Normalize input string (remove spaces, convert to lower)
            pattern_chars.append(OCR_CHAR_MAP.get(char, re.escape(char))) # Use map or escape if not in map
        return '(?:\s*' + ''.join(pattern_chars) + '\s*)' # Allow optional spaces around the full pattern

    # Convert the target display string and numeric value into flexible regex patterns for Tesseract's output.
    flexible_display_pattern = create_flexible_pattern(handicap_display_str) # e.g., "(-1.5)" -> "[\(\s]*[—\-a]?1[\.,]5[\)\s]*"
    flexible_numeric_value_pattern = create_flexible_pattern(handicap_value_float_str) # e.g., "-1.5" -> "[—\-a]?1[\.,]5"
    flexible_abs_numeric_pattern = create_flexible_pattern(handicap_value_abs_str) # e.g., "1.5" -> "1[\.,]5"


    potential_handicap_blocks = []
    for idx, block in enumerate(ocr_blocks):
        block_text = block[1] # Original text from Tesseract
        # Aggressively clean Tesseract's block text for matching
        # Remove all whitespace, and replace commas with dots, and em-dashes with hyphens
        block_text_cleaned_for_match = block_text.lower().replace(' ', '').replace(',', '.').replace('—', '-')

        is_handicap_value_matched = False

        # Try matching the full display string (e.g., "(-1.5)") with flexibility
        if re.search(flexible_display_pattern, block_text_cleaned_for_match):
            is_handicap_value_matched = True
            telegram_log(f"[DEBUG_CLICK_HELPER] Tesseract matched flexible display pattern: '{block_text}' (cleaned:'{block_text_cleaned_for_match}') against '{handicap_display_str}'", is_debug_message=True)
        # Try matching just the numeric value (e.g., "-1.5") with flexibility
        elif re.search(flexible_numeric_value_pattern, block_text_cleaned_for_match):
            is_handicap_value_matched = True
            telegram_log(f"[DEBUG_CLICK_HELPER] Tesseract matched flexible numeric value pattern: '{block_text}' (cleaned:'{block_text_cleaned_for_match}') against '{handicap_value_float_str}'", is_debug_message=True)
        # Try matching just the absolute numeric value (e.g., "1.5"), might miss sign or parentheses, with flexibility
        elif re.search(flexible_abs_numeric_pattern, block_text_cleaned_for_match):
            is_handicap_value_matched = True
            telegram_log(f"[DEBUG_CLICK_HELPER] Tesseract matched flexible absolute numeric pattern: '{block_text}' (cleaned:'{block_text_cleaned_for_match}') against '{handicap_value_abs_str}'", is_debug_message=True)
        else:
            telegram_log(f"[DEBUG_CLICK_HELPER] Tesseract did NOT match: '{block_text}' (cleaned:'{block_text_cleaned_for_match}') with any handicap patterns for '{handicap_display_str}'", is_debug_message=True)


        if is_handicap_value_matched:
            potential_handicap_blocks.append(block)
            telegram_log(f"[HANDICAP_CLICK_HELPER] Найден ПОТЕНЦИАЛЬНЫЙ целевой блок форы: '{block_text}'.", is_debug_message=True)


    if not potential_handicap_blocks:
        telegram_log(f"[HANDICAP_CLICK_HELPER] Потенциальные блоки форы для '{handicap_display_str}' не найдены Tesseract'ом.", is_debug_message=True)
        return False

    # Теперь из потенциальных блоков форы находим соответствующий коэффициент и кликаем
    for block in potential_handicap_blocks:
        x_block, y_block = block[0][0][0], block[0][0][1]
        block_width = block[0][1][0] - block[0][0][0]
        block_height = block[0][2][1] - block[0][0][1]

        telegram_log(f"[HANDICAP_CLICK_HELPER] Обработка потенциального блока форы: '{block[1]}'. Ищем коэффициент.", is_debug_message=True)

        # 1. Попытка найти коэффициент ВСТРОЕННЫЙ (если текст блока форы его содержит)
        text_in_block_normalized = block[1].replace(' ', '').replace(',', '.')
        handicap_text_in_target_outcome_normalized = handicap_display_str.replace(' ', '')
        
        coef_match_inline = re.search(r'\b(\d+(?:\.\d+)?)$', text_in_block_normalized)
        if coef_match_inline:
            handicap_idx_in_block = text_in_block_normalized.find(handicap_text_in_target_outcome_normalized)
            coef_idx_in_block = text_in_block_normalized.find(coef_match_inline.group(1))

            if handicap_idx_in_block != -1 and coef_idx_in_block != -1 and coef_idx_in_block > handicap_idx_in_block:
                coef_text = coef_match_inline.group(1)
                block_center_x = x_block + block_width / 2
                block_center_y = y_block + block_height / 2
                
                pyautogui.click(int(block_center_x) + region_offset_x, int(block_center_y) + region_offset_y)
                telegram_log(f"[HANDICAP_CLICK_HELPER][INLINE_COEF] Кликнута фора '{block[1]}' со встроенным коэф. '{coef_text}' по ({int(block_center_x) + region_offset_x}, {int(block_center_y) + region_offset_y}).", is_debug_message=True)
                time.sleep(0.5) # Добавлена задержка
                return True

        # 2. Попытка найти коэффициент в соседних блоках
        for j in range(ocr_blocks.index(block) + 1, len(ocr_blocks)):
            next_block = ocr_blocks[j]
            x_next_block, y_next_block = next_block[0][0][0], next_block[0][0][1]

            if abs(y_next_block - y_block) < Y_LINE_TOLERANCE_COEF and \
               x_next_block > x_block and \
               (x_next_block - (x_block + block_width)) < X_COEF_SEARCH_RANGE:
                
                if re.match(r'^\d+(?:\.\d+)?$', next_block[1].replace(',', '.')):
                    coef_block_width = next_block[0][1][0] - next_block[0][0][0]
                    coef_block_height = next_block[0][2][1] - next_block[0][0][1]

                    coef_block_center_x = x_next_block + coef_block_width / 2
                    coef_block_center_y = y_next_block + coef_block_height / 2
                    
                    pyautogui.click(int(coef_block_center_x) + region_offset_x, int(coef_block_center_y) + region_offset_y)
                    telegram_log(f"[HANDICAP_CLICK_HELPER][ADJ_COEF] Кликнута фора '{block[1]}' с соседним коэф. '{next_block[1]}' по ({int(coef_block_center_x) + region_offset_x}, {int(coef_block_center_y) + region_offset_y}).", is_debug_message=True)
                    time.sleep(0.5) # Добавлена задержка
                    return True
                else:
                    telegram_log(f"[HANDICAP_CLICK_HELPER][DEBUG] Блок '{next_block[1]}' горизонтально близко, но не является коэффициентом. (dx={x_next_block - (x_block + block_width)}).", is_debug_message=True)
            elif y_next_block - y_block > Y_LINE_TOLERANCE_COEF * 2:
                break
            elif x_next_block - (x_block + block_width) > X_COEF_SEARCH_RANGE:
                break

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

        debug_outcome_path = f"debug_outcome_screenshot_{iteration+1}.png"
        current_screenshot_pil.save(debug_outcome_path)
        for chat_id in subscribers:
            send_photo(chat_id, debug_outcome_path, caption=f"Скриншот поиска исхода, итерация {iteration+1}")

        full_text, results = get_ocr_results(current_screenshot_pil)
        ocr_debug = f"[OCR] Итерация {iteration+1}\nFull text:\n{full_text}\nBlocks:\n" + "\n".join([str(r) for r in results])
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
    if last_screenshot_path:
        for chat_id in subscribers:
            send_photo(chat_id, last_screenshot_path, caption="Скриншот последней попытки поиска исхода")
    if last_full_text is not None and last_results is not None:
        ocr_debug = f"[OCR] Последняя попытка\nFull text:\n{last_full_text}\nBlocks:\n" + "\n".join([str(r) for r in last_results])
        for chat_id in subscribers:
            send_message(chat_id, ocr_debug[:4000])
    return None, None

# ======================= Координаты и константы для ставок =======================

BET_INPUT_CANDIDATES_SET1 = [(1202, 445), (1200, 491), (1200, 660)]
BET_INPUT_CANDIDATES_SET2 = [(1201, 657)]

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

def check_yellow_pixel(x, y, r_tolerance=10, g_tolerance=10, b_tolerance=10):
    """
    Проверяет, является ли пиксель жёлтым с ожидаемыми значениями RGB ~ (96, 100, 75).
    Допуски по умолчанию установлены в 10.
    """
    try:
        screenshot = pyautogui.screenshot(region=(x, y, 1, 1))
        pixel = screenshot.getpixel((0, 0))
        r, g, b = pixel[:3]
        is_yellow = (
            abs(r - 96) <= r_tolerance and
            abs(g - 100) <= g_tolerance and
            abs(b - 75) <= b_tolerance
        )
        return is_yellow
    except Exception as e:
        telegram_log(f"[ERROR] Ошибка при проверке желтого пикселя в ({x},{y}): {e}")
        return False

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
            except:
                valid = False
        elif token.startswith("<"):
            try:
                threshold = float(token[1:])
                if not (found_coef <= threshold):
                    valid = False
            except:
                valid = False
        else:
            try:
                exact_value = float(token)
                if not (found_coef == exact_value):
                    valid = False
            except:
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

    debug_coef_path = "debug_coef_screenshot.png"
    image_to_process.save(debug_coef_path)
    for chat_id in subscribers:
        send_photo(chat_id, debug_coef_path, caption=f"Скрин коэффициента (область {region})")

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
                except Exception as e:
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
        except:
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
    # ОБНОВЛЕННЫЕ КООРДИНАТЫ И ЦВЕТА ИНДИКАТОРА ГОТОВНОСТИ (ПОСЛЕ АНАЛИЗА СКРИНШОТОВ)
    # Если на скриншотах видно, что (1250, 249) все еще не зеленый, то нужно
    # подобрать новую точку или цвет, либо увеличить таймаут.
    READY_CHECK_COORDS = (1250, 249)
    TARGET_READY_COLOR = (6, 136, 69) 
    CHECK_REGION_SIZE = 5

    BET_INPUT_PRIMARY = (1199, 319)
    BET_INPUT_SECONDARY = (1199, 336)
    TARGET_WHITE_COLOR = (255, 255, 255)

    start_time = time.time()
    telegram_log(f"[DEBUG] Ожидание индикатора готовности поля ввода (цвет {TARGET_READY_COLOR} в {READY_CHECK_COORDS})...", is_debug_message=True)

    ready_indicator_found = False
    debug_ss_counter = 0 # Добавляем счетчик для отладочных скриншотов
    while time.time() - start_time < timeout:
        region_ready = (READY_CHECK_COORDS[0], READY_CHECK_COORDS[1], CHECK_REGION_SIZE, CHECK_REGION_SIZE)
        try:
            screenshot_ready = pyautogui.screenshot(region=region_ready)
            stat_ready = ImageStat.Stat(screenshot_ready)
            avg_color_ready = tuple(int(c) for c in stat_ready.mean) # Используем средний цвет для устойчивости
            
            telegram_log(f"[DEBUG] Проверка индикатора готовности: {READY_CHECK_COORDS}, текущий цвет {avg_color_ready}", is_debug_message=True)

            # Отправка отладочного скриншота маленькой области для анализа
            if DEBUG_SCREENSHOT: # Отправляем только если DEBUG_SCREENSHOT включен
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
                 text_between_cleaned = text_between_cleaned.lower().replace("победа с учетом фотом", "").strip()
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

    Y_TOLERANCE_LINE = 20

    OUTCOME_SEARCH_REGION = (206, 151, 958, 641)
    x1_region, y1_region, x2_region, y2_region = OUTCOME_SEARCH_REGION
    region_width = x2_region - x1_region
    region_height = y2_region - y1_region

    for scroll_iter in range(max_scrolls):
        screenshot = pyautogui.screenshot(region=(x1_region, y1_region, region_width, region_height))
        time.sleep(0.5)

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

            ocr_debug_msg = (f"[OCR][HT_HANDICAP][SCROLL {scroll_iter+1}] Ищем: {outcome_str}\\n"
                           f"Parsed: Half='{half_identifier_search}', Team='{team_name_search}', Handicap='{handicap_display_search}'\\n"
                           f"Blocks (first 30):\\n" + "\\n".join([f'{b[1]} @ {b[0]}' for b in ocr_blocks[:30]]))
            if len(ocr_blocks) > 30:
                ocr_debug_msg += "\\n..."
            for chat_id in subscribers:
                send_message(chat_id, ocr_debug_msg[:4000])
        except Exception as e_ocr_section:
            telegram_log(f"[ERROR][HT_HANDICAP] Исключение во время OCR или отправки OCR лога для скролла {scroll_iter+1}: {e_ocr_section}", is_debug_message=True)
            import traceback
            tb_str = traceback.format_exc()
            telegram_log(f"[TRACEBACK_OCR_HT_HANDICAP]:\\n{tb_str[:3500]}", is_debug_message=True)
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
                block_text_no_space = block[1].replace(" ", "")
                handicap_search_no_space = handicap_display_search.replace(" ", "")
                if handicap_search_no_space in block_text_no_space:
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
                for i_team_check in range(idx_handicap_block - 1, max(-1, idx_handicap_block - 5), -1):
                    if i_team_check < 0: 
                        break
                    check_block_team = ocr_blocks[i_team_check]
                    y_check_block_team = check_block_team[0][0][1]
                    if abs(y_check_block_team - y_handicap_block) < Y_TOLERANCE_LINE + 10:
                        if team_name_search.lower() in check_block_team[1].lower():
                            team_found_for_this_handicap = True
                            telegram_log(f"[HT_HANDICAP_NEW] Команда '{team_name_search}' найдена ('{check_block_team[1]}') рядом с форой '{block_handicap_display[1]}'", is_debug_message=True)
                            break
                if not team_found_for_this_handicap:
                    telegram_log(f"[HT_HANDICAP_NEW] Команда '{team_name_search}' НЕ найдена для форы '{block_handicap_display[1]}'. Пропускаем эту фору.", is_debug_message=True)
                    continue

            text_in_handicap_block = block_handicap_display[1].replace(" ", "")
            handicap_text_no_space = handicap_display_search.replace(" ", "")

            handicap_end_pos = -1
            match_display_str_pos = text_in_handicap_block.find(handicap_text_no_space)
            if match_display_str_pos != -1:
                handicap_end_pos = match_display_str_pos + len(handicap_text_no_space)
            elif re.match(r'^\(?[-+]?' + re.escape(handicap_text_no_space.replace('(', '').replace(')', '')) + r'\)?$', text_in_handicap_block):
                match_value_str_pos = text_in_handicap_block.find(handicap_text_no_space.replace('(', '').replace(')', ''))
                if match_value_str_pos != -1:
                    handicap_end_pos = match_value_str_pos + len(handicap_text_no_space.replace('(', '').replace(')', '')) # ИСПРАВЛЕНИЕ: опечатка handacap
                    if handicap_end_pos < len(text_in_handicap_block) and handicap_end_pos < len(text_in_handicap_block) and block_text_normalized[handicap_end_pos] == ')':
                        handicap_end_pos += 1

            if handicap_end_pos != -1:
                remaining_text_in_block = text_in_handicap_block[handicap_end_pos:].strip()
                coef_match_inline = re.match(r'^\s*([0-9]+(?:\.[0-9]+)?)$', remaining_text_in_block.replace(",", "."))
                if coef_match_inline:
                    coef_value = coef_match_inline.group(1)
                    telegram_log(f"[HT_HANDICAP_NEW] Коэффициент '{coef_value}' найден ВНУТРИ блока с форой: '{block_handicap_display[1]}'", is_debug_message=True)
                    abs_click_x = x_handicap_block + x1_region
                    abs_click_y = y_handicap_block + y1_region
                    pyautogui.click(abs_click_x, abs_click_y)
                    telegram_log(f"[КЛИК][HT_HANDICAP_NEW] Клик по '{block_handicap_display[1]}' на ({abs_click_x}, {abs_click_y}) (коэф. inline)")
                    return True

            if block_handicap_display[1].strip().startswith(handicap_display_search):
                telegram_log(f"[HT_HANDICAP_NEW][SCROLL][DEBUG] Найден блок с форой, начинающийся с {handicap_display_search}: '{block_handicap_display[1].strip()}'")
                found_coef = False
                checked_blocks = []
                for offset in range(1, 8):
                    idx_coef = idx_handicap_block + offset
                    if idx_coef >= len(ocr_blocks): 
                        break
                    next_block = ocr_blocks[idx_coef]
                    checked_blocks.append(next_block[1])
                    coef_match = re.search(r'\d+(?:\.\d+)?', next_block[1])
                    if coef_match:
                        abs_click_x = next_block[0][0][0] + x1_region
                        abs_click_y = next_block[0][0][1] + y1_region
                        if (abs_click_x, abs_click_y) == (0, 0):
                            telegram_log(f"[HT_HANDICAP_NEW][SCROLL][ERROR] Координаты для клика равны (0, 0) — клик не совершен!")
                        else:
                            pyautogui.click(abs_click_x, abs_click_y)
                            telegram_log(f"[КЛИК][HT_HANDICAP_NEW] Клик по исходу: {team_name_search} {handicap_display_search} по абсолютным координатам {abs_click_x}, {abs_click_y} (коэффициент в следующем блоке #{idx_coef})")
                            return True
                        found_coef = True
                if not found_coef:
                    telegram_log(f"[HT_HANDICAP_NEW][SCROLL] Блок с форой {handicap_display_search} найден (начинается с {handim_display_search}), но кэф не найден ни в этом, ни в следующих 7 блоках. Просмотренные блоки: {checked_blocks}")
        telegram_log(f"[HT_HANDICAP_NEW][SCROLL] Исход не найден на текущем экране (скролл {scroll_iter+1}).", is_debug_message=True)
        pyautogui.scroll(-4)
        time.sleep(1)

    telegram_log(f"[HT_HANDICAP_NEW][FAIL] Не удалось найти исход: {outcome_str} после {max_scrolls} скроллов.", is_debug_message=True)
    return False

def find_outcome(match_name: str, outcome: str, coef_condition: str, bet_amount: float):
    global chosen_candidate
    PREDEFINED_OUTCOME_COORDS = {
        "1": (483, 570),
        "X": (543, 570),
        "2": (606, 570)
    }
    OUTCOME_SEARCH_REGION = (206, 151, 958, 641)
    FINISH_COORDS = (1160, 597)
    RETRY_COORDS = (1254, 363)

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
    elif "тайм" in outcome.lower() and "тотал" in outcome.lower():
        telegram_log(f"Обнаружен исход halftime total: '{outcome}'. Использую find_halftime_total_and_click_new.", is_debug_message=True)
        if find_halftime_total_and_click_new(outcome):
            time.sleep(2)
            outcome_successfully_clicked = True
        else:
            telegram_log(f"[ERROR] Halftime total исход '{outcome}' не найден!")
    elif "тотал голов (" in outcome.lower():
        telegram_log(f"Обнаружен исход team total: '{outcome}'. Использую find_total_and_click_coef_team_new.", is_debug_message=True)
        if find_total_and_click_coef_team_new(outcome):
            time.sleep(2)
            outcome_successfully_clicked = True
        else:
            telegram_log("[ERROR] Тотал по команде не найден!")
    elif "тотал" in outcome.lower():
        telegram_log(f"Обнаружен общий исход total: '{outcome}'. Использую find_total_and_click_coef_new.", is_debug_message=True)
        if find_total_and_click_coef_new(outcome):
            time.sleep(2)
            outcome_successfully_clicked = True
        else:
            telegram_log("[ERROR] Тотал не найден!")
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
            if found_coords == (0, 0):
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

def finalize_bet_and_navigate():
    """Выполняет клики для завершения ставки и навигации/очистки купона."""
    telegram_log("[NAVIGATE] Запуск finalize_bet_and_navigate...")

    time.sleep(1)
    home_coords = (70, 142)
    telegram_log(f"[NAVIGATE] Попытка клика по координатам возврата домой: {home_coords}")
    pyautogui.click(home_coords[0], home_coords[1])
    telegram_log("[NAVIGATE] Возврат на главную страницу (клик выполнен)...")
    telegram_log("[NAVIGATE] Завершающие клики выполнены.")

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

⚠️ *Важно:*
- Разделяйте параметры запятыми
- Указывайте точное название матча как в Live
- Проверяйте правильность формата перед отправкой
"""
    for chat_id in subscribers:
        send_message(chat_id, instructions)

def parse_total_outcome_new(outcome, ocr_blocks):
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

def find_total_outcome_table_new(outcome, ocr_blocks):
    """
    Поиск исхода с тоталом в табличной структуре:
    1. Находит нужный тотал (например, 2.5)
    2. Определяет тип (больше/меньше)
    3. Ищет соответствующий коэффициент
    4. Кликает по найденному коэффициенту
    Координаты ocr_blocks уже масштабированы обратно к исходному размеру скриншота.
    """
    Y_TOLERANCE = 15
    Y_DOWN_TOLERANCE = 25

    base_type, total_value = parse_total_outcome_new(outcome, ocr_blocks)
    if not base_type or not total_value:
        return False

    total_str = f"({total_value})"

    for block_total in ocr_blocks:
        if total_str in block_total[1]:
            x_total = block_total[0][0][0]
            y_total = block_total[0][0][1]

            for block_coef in ocr_blocks:
                x_coef = block_coef[0][0][0]
                y_coef = block_coef[0][0][1]

                if abs(y_coef - y_total) < Y_TOLERANCE:
                    if base_type == "меньше" and x_coef < x_total:
                        if re.match(r'^\d+(?:\.\d+)?', block_coef[1].replace(',', '.')):
                            pyautogui.click(x_coef + globals()["OUTCOME_SEARCH_REGION"][0], y_coef + globals()["OUTCOME_SEARCH_REGION"][1])
                            telegram_log(f"[TOTAL_TABLE_NEW] Клик по исходу: тотал {total_str} {base_type} по координатам {x_coef}, {y_coef}")
                            return True
                    elif base_type == "больше" and x_coef > x_total:
                        if re.match(r'^\d+(?:\.\d+)?', block_coef[1].replace(',', '.')):
                            pyautogui.click(x_coef + globals()["OUTCOME_SEARCH_REGION"][0], y_coef + globals()["OUTCOME_SEARCH_REGION"][1])
                            telegram_log(f"[TOTAL_TABLE_NEW] Клик по исходу: тотал {total_str} {base_type} по координатам {x_coef}, {y_coef}")
                            return True

            for block_coef in ocr_blocks:
                x_coef = block_coef[0][0][0]
                y_coef = block_coef[0][0][1]

                if y_coef > y_total and y_coef - y_total < Y_DOWN_TOLERANCE:
                    if base_type == "меньше" and x_coef < x_total:
                        if re.match(r'\d+(?:\.\d+)?', block_coef[1].replace(',', '.')):
                            pyautogui.click(x_coef + globals()["OUTCOME_SEARCH_REGION"][0], y_coef + globals()["OUTCOME_SEARCH_REGION"][1])
                            telegram_log(f"[TOTAL_TABLE_NEW] Клик по исходу: тотал {total_str} {base_type} по координатам {x_coef}, {y_coef} (ниже)")
                            return True
                    elif base_type == "больше" and x_coef > x_total:
                        if re.match(r'\d+(?:\.\d+)?', block_coef[1].replace(',', '.')):
                            pyautogui.click(x_coef + globals()["OUTCOME_SEARCH_REGION"][0], y_coef + globals()["OUTCOME_SEARCH_REGION"][1])
                            telegram_log(f"[TOTAL_TABLE_NEW] Клик по исходу: тотал {total_str} {base_type} по координатам {x_coef}, {y_coef} (ниже)")
                            return True

    telegram_log(f"[TOTAL_TABLE_NEW] Не найден коэффициент для тотала {total_str} {base_type}")
    return False

def find_total_coef_candidates_new(total_block, ocr_blocks, base_type, x1_region, y1_region):
    """
    Ищет все числовые блоки-коэффициенты рядом с блоком тотала.
    Координаты ocr_blocks уже масштабированы обратно к исходному размеру скриншота.
    """
    total_x = total_block[0][0][0]
    total_y = total_block[0][0][1]
    Y_RADIUS = 30
    candidates = []
    for coef_block in ocr_blocks:
        coef_x = coef_block[0][0][0]
        coef_y = coef_block[0][0][1]
        if abs(coef_y - total_y) < Y_RADIUS and re.match(r'^\d+(?:\.\d+)?$', coef_block[1].replace(',', '.')):
            dx = abs(coef_x - total_x)
            candidates.append((coef_block, coef_x, coef_y, dx))
    candidates.sort(key=lambda c: c[3])
    if candidates:
        msg = '[TOTAL_NEW][FLEX] Найдены кандидаты коэффициентов рядом с тоталом: '
        msg += ', '.join([f"{c[0][1]}@({c[1]},{c[2]})" for c in candidates])
        telegram_log(msg)
    return candidates

def find_total_and_click_coef_new(outcome, max_scrolls=7):
    base_type, total_value = parse_total_outcome_new(outcome, [])
    if not base_type or not total_value:
        telegram_log(f"[TOTAL_NEW] Не удалось распарсить исход: {outcome}")
        return False

    total_str = f"({total_value})"
    telegram_log(f"[TOTAL_NEW] Ищем тотал: {total_str}, тип: {base_type}")

    OUTCOME_SEARCH_REGION = (206, 151, 958, 641)
    x1, y1, x2, y2 = OUTCOME_SEARCH_REGION
    region_width = x2 - x1
    region_height = y2 - y1

    for scroll_iter in range(max_scrolls):
        screenshot = pyautogui.screenshot(region=(x1, y1, region_width, region_height))
        time.sleep(1)
        debug_total_path = f"debug_total_scroll_{scroll_iter+1}.png"
        screenshot.save(debug_total_path)
        total_blocks = []
        full_text, ocr_blocks = get_ocr_results(screenshot)
        for idx, block in enumerate(ocr_blocks):
            if block[1].strip() == total_value.replace('.', '') and idx > 0 and ocr_blocks[idx - 1][1] == '(' and idx + 1 < len(ocr_blocks) and ocr_blocks[idx + 1][1] == ')':
                total_blocks.append(block)
            elif total_str in block[1]:
                total_blocks.append(block)

        candidates_list_for_diag = []
        found_any = False
        for total_block in total_blocks:
            coef_inline_match = re.search(r'\b\d+(?:\.\d+)?\b', total_block[1].replace(total_str, ''))
            if coef_inline_match:
                found_any = True
                candidates_list_for_diag.append(f"{coef_inline_match.group(0)} (inline)")

            coef_candidates = find_total_coef_candidates_new(total_block, ocr_blocks, base_type, x1, y1)
            if coef_candidates:
                candidates_list_for_diag.extend([f"{c[0][1]}@({c[1]},{c[2]})" for c in coef_candidates])
                found_any = True

        send_ocr_diagnostics_telegram(
            scroll_iter,
            "total",
            f"Ищу тотал: {total_str} {base_type}",
            debug_total_path,
            full_text,
            ocr_blocks,
            candidates=candidates_list_for_diag,
            found=found_any
        )
        if find_total_outcome_table_new(outcome, ocr_blocks):
            return True

        pyautogui.scroll(-4)
        time.sleep(1)

    telegram_log(f"[TOTAL_NEW] Не удалось найти исход: {outcome} после {max_scrolls} скроллов.")
    return False

def parse_total_team_from_outcome_new(outcome: str) -> Optional[str]:
    """
    Извлекает имя команды из исхода тотала, если указано в скобках: "Тотал голов (Лечче) Меньше (1.0)"
    Возвращает имя команды или None.
    """
    m = re.search(r'Тотал голов \(([^)]+)\)', outcome, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None

def find_total_and_click_coef_team_new(outcome, max_scrolls=7):
    """
    Поиск и клик по исходу тотала для конкретной команды.
    Координаты ocr_blocks уже масштабированы обратно к исходному размеру скриншота.
    """
    team = parse_total_team_from_outcome_new(outcome)
    base_type, total_value = parse_total_outcome_new(outcome, [])
    if not base_type or not total_value or not team:
        telegram_log(f"[TOTAL_TEAM_NEW] Не удалось распарсить исход или команду: {outcome}")
        return False

    total_str = f"({total_value})"
    telegram_log(f"[TOTAL_TEAM_NEW] Ищем тотал: {total_str}, тип: {base_type}, команда: {team}", is_debug_message=True)

    Y_TOLERANCE_HEADER_SEQUENCE = 20
    Y_GAP_FOR_NEXT_SECTION = 40
    X_COL_ALIGN_TOLERANCE = 70
    Y_TOLERANCE_TOTAL_COEF = 20

    OUTCOME_SEARCH_REGION = (206, 151, 958, 641)
    x1_region, y1_region, _, _ = OUTCOME_SEARCH_REGION
    region_width = OUTCOME_SEARCH_REGION[2] - OUTCOME_SEARCH_REGION[0]
    region_height = OUTCOME_SEARCH_REGION[3] - OUTCOME_SEARCH_REGION[1]

    for scroll_iter in range(max_scrolls):
        screenshot = pyautogui.screenshot(region=(x1_region, y1_region, region_width, region_height))
        time.sleep(1)
        debug_total_path = f"debug_total_team_scroll_{scroll_iter+1}.png"
        screenshot.save(debug_total_path)
        for chat_id in subscribers:
            send_photo(chat_id, debug_total_path, caption=f"Скриншот поиска тотала по команде {team}, скролл {scroll_iter+1}")

        full_text, ocr_blocks = get_ocr_results(screenshot)
        ocr_debug_msg = f"[OCR][TOTAL_TEAM][SCROLL {scroll_iter+1}]\nКоманда: {team}, Тотал: {total_str} {base_type}\nFull text:\n{full_text}\nBlocks:\n" + "\n".join([str(r) for r in ocr_blocks])
        for chat_id in subscribers:
            send_message(chat_id, ocr_debug_msg[:4000])

        section_content_blocks = []
        header_found = False
        header_avg_y = -1

        for i in range(len(ocr_blocks) - 1):
            b_total = ocr_blocks[i]
            # Fix Ruff E713: Test for membership should be `not in`
            if "тотал" not in b_total[1].lower():
                continue

            idx_after_total = i
            b_lparen = None
            idx_after_lparen = -1
            header_prefix_blocks = [b_total]

            if idx_after_total + 1 < len(ocr_blocks):
                b_next_after_total = ocr_blocks[idx_after_total + 1]
                if "голов" in b_next_after_total[1].lower():
                    header_prefix_blocks.append(b_next_after_total)
                    if idx_after_total + 2 < len(ocr_blocks) and ocr_blocks[idx_after_total + 2][1] == '(':
                        b_lparen = ocr_blocks[idx_after_total + 2]
                        idx_after_lparen = idx_after_total + 2
                        header_prefix_blocks.append(b_lparen)
                    else: 
                        continue
                elif b_next_after_total[1] == '(':
                    b_lparen = b_next_after_total
                    idx_after_lparen = idx_after_total + 1
                    header_prefix_blocks.append(b_lparen)
                else: 
                    continue
            else: 
                continue

            if not b_lparen: 
                continue

            y_coords_prefix_check = [b_check[0][0][1] for b_check in header_prefix_blocks]
            if max(y_coords_prefix_check) - min(y_coords_prefix_check) > Y_TOLERANCE_HEADER_SEQUENCE:
                continue

            current_header_avg_y_calc = sum(y_coords_prefix_check) / len(y_coords_prefix_check)

            for num_team_name_blocks_to_try in range(1, 6):
                end_idx_for_team_name = idx_after_lparen + num_team_name_blocks_to_try
                if end_idx_for_team_name >= len(ocr_blocks): 
                    break

                team_name_candidate_parts = [ocr_blocks[k][1] for k in range(idx_after_lparen + 1, end_idx_for_team_name + 1)]
                formed_ocr_team_name = " ".join(team_name_candidate_parts)

                all_team_parts_on_line = True
                for k_team_part in range(idx_after_lparen + 1, end_idx_for_team_name + 1):
                    y_team_part_block = ocr_blocks[k_team_part][0][0][1]
                    if abs(y_team_part_block - current_header_avg_y_calc) > Y_TOLERANCE_HEADER_SEQUENCE:
                        all_team_parts_on_line = False 
                        break
                if not all_team_parts_on_line: 
                    continue

                team_check_normalized = team.lower().replace('-', '').replace(' ', '')
                formed_ocr_check_normalized = formed_ocr_team_name.lower().replace('-', '').replace(' ', '')

                if team_check_normalized in formed_ocr_check_normalized:
                    idx_potential_rparen_check = end_idx_for_team_name + 1
                    if idx_potential_rparen_check < len(ocr_blocks):
                        b_rparen_cand_check = ocr_blocks[idx_potential_rparen_check]
                        y_rparen_cand_check = b_rparen_cand_check[0][0][1]
                        if b_rparen_cand_check[1] == ')' and abs(y_rparen_cand_check - current_header_avg_y_calc) < Y_TOLERANCE_HEADER_SEQUENCE:
                            header_found = True
                            header_avg_y = current_header_avg_y_calc
                            content_start_actual_idx = idx_potential_rparen_check + 1

                            section_end_determine_idx = len(ocr_blocks)
                            for k_sec_end in range(content_start_actual_idx, len(ocr_blocks)):
                                k_block_text_lower_sec_end = ocr_blocks[k_sec_end][1].lower()
                                k_block_y_min_sec_end = ocr_blocks[k_sec_end][0][0][1]
                                is_another_team_total = "тотал голов (" in k_block_text_lower_sec_end and team.lower() not in k_block_text_lower_sec_end
                                is_general_totals_header = ("тотал" == k_block_text_lower_sec_end.strip() or "тоталы" == k_block_text_lower_sec_end.strip())
                                if (is_another_team_total or (is_general_totals_header and k_block_y_min_sec_end > header_avg_y + Y_GAP_FOR_NEXT_SECTION)):
                                    section_end_determine_idx = k_sec_end 
                                    break

                            section_content_blocks = ocr_blocks[content_start_actual_idx : section_end_determine_idx]
                            telegram_log(f"[TOTAL_TEAM_NEW] Найдена секция для '{team}' (Y ~{header_avg_y:.0f}). Bлоков контента: {len(section_content_blocks)}")
                            break
            if header_found: 
                break

        if not header_found or not section_content_blocks:
            if not header_found: 
                telegram_log(f"[TOTAL_TEAM_NEW][SCROLL] Заголовок секции для '{team}' не найден (итерация {scroll_iter+1}).")
            elif not section_content_blocks: 
                telegram_log(f"[TOTAL_TEAM_NEW][SCROLL] Секция '{team}' найдена, но пуста (итерация {scroll_iter+1}).")
            pyautogui.scroll(-4) 
            time.sleep(1) 
            continue

        ocr_section_debug = f"[OCR][TOTAL_TEAM_SECTION] Bloки для секции '{team}' (после заголовка):\n" + "\n".join([str(r) for r in section_content_blocks])
        for chat_id in subscribers: 
            send_message(chat_id, ocr_section_debug[:4000])

        column_header_block = None
        for block_in_section_content in section_content_blocks:
            if base_type.lower() == block_in_section_content[1].strip().lower():
                column_header_block = block_in_section_content
                telegram_log(f"[TOTAL_TEAM_NEW] B секции {team} найден заголовок колонки: '{column_header_block[1]}'")
                break

        col_x_start_val = x1_region
        min_y_for_outcome_search = header_avg_y
        if column_header_block:
            col_x_start_val = column_header_block[0][0][0]
            col_x_end_val = column_header_block[0][1][0]
            min_y_for_outcome_search = column_header_block[0][2][1]
        else:
            telegram_log(f"[TOTAL_TEAM_NEW] Заголовок колонки '{base_type}' не найден в секции {team}. Поиск будет шире.")


        found_outcome_and_clicked = False
        for idx_in_section_content, current_block in enumerate(section_content_blocks):
            if current_block[0][0][1] <= min_y_for_outcome_search:
                continue

            block_x_center = (current_block[0][0][0] + current_block[0][1][0]) / 2
            if not (col_x_start_val - X_COL_ALIGN_TOLERANCE < block_x_center < col_x_end_val + X_COL_ALIGN_TOLERANCE):
                continue

            current_block_text_stripped = current_block[1].replace(',', '.')

            regex_target_total_escaped = re.escape(total_str)
            match_combined = re.search(rf"^{regex_target_total_escaped}\s*(\d+\.\d+)$", current_block_text_stripped)
            if match_combined:
                coef_text = match_combined.group(1)
                telegram_log(f"[DEBUG] Логика А: Найден комб. блок '{current_block[1]}' (коэф: {coef_text}) под колонкой '{base_type}'", is_debug_message=True)
                abs_click_x = current_block[0][0][0] + x1_region
                abs_click_y = current_block[0][0][1] + y1_region
                pyautogui.click(abs_click_x, abs_click_y)
                telegram_log(f"[КЛИК][А] Клик по halftime total: '{current_block[1]}' по ({abs_click_x}, {abs_click_y})", is_debug_message=True)
                found_outcome_and_clicked = True
                break

            if current_block_text_stripped == total_str.replace(' ', ''):
                for j in range(idx_in_section_content + 1, min(idx_in_section_content + 5, len(section_content_blocks))):
                    next_block = section_content_blocks[j]
                    y_next_block = next_block[0][0][1]
                    if abs(y_next_block - current_block[0][0][1]) < Y_TOLERANCE_TOTAL_COEF:
                        coef_match = re.match(r"^(\d+(?:\.\d+)?)$", next_block[1].replace(',', '.'))
                        if coef_match:
                            coef_text = coef_match.group(1)
                            telegram_log(f"[DEBUG] Логика Б: Найден блок '{current_block[1]}' и след. коэф. '{coef_text}' под '{base_type}'", is_debug_message=True)
                            abs_click_x = next_block[0][0][0] + x1_region
                            abs_click_y = next_block[0][0][1] + y1_region
                            pyautogui.click(abs_click_x, abs_click_y)
                            telegram_log(f"[КЛИК][Б] Клик по halftime total: '{current_block[1]}' (коэф: {coef_text}) по ({abs_click_x}, {abs_click_y})", is_debug_message=True)
                            found_outcome_and_clicked = True
                            break
                if found_outcome_and_clicked: 
                    break

        if found_outcome_and_clicked: 
            return True

        if not found_outcome_and_clicked:
            telegram_log(f"Исход '{outcome}' с коэффициентом не найден в секции команды '{team}' (итерация {scroll_iter+1}).", is_debug_message=True)
            pyautogui.scroll(-4) 
            time.sleep(1) 
            continue

    telegram_log(f"[SCROLL_END] Не удалось найти halftime total исход: {outcome} после {max_scrolls} скроллов.", is_debug_message=True)
    return False

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
    for chat_id in subscribers:
        send_photo(chat_id, screenshot_path, caption=header)
    msg = header + f"\nFull OCR text:\n{full_text}\n\nOCR blocks:\n"
    msg += "\n".join([str(b) for b in ocr_blocks[:30]])
    if len(ocr_blocks) > 30:
        msg += f"\n... (ещё {len(ocr_blocks)-30} блоков)"
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
    # Get the absolute numeric value (e.g., "1.0" from "-1.0")
    abs_handicap_value_str = str(abs(float(handicap_value_from_input)))
    abs_handicap_value_escaped = re.escape(abs_handicap_value_str.replace('.', r'\.'))

    # Pattern for (X.X) or X.X or -X.X or +X.X allowing optional spaces inside/around
    # This covers `(-1.0)`, `(1.0)`, `1.0`, `-1.0` as well as variations like `( -1.0 )`
    loose_handicap_value_pattern = re.compile(
        r'[\(\s]*[-+]?\s*' + abs_handicap_value_escaped + r'\s*[\)\s]*'
    )
    # Also consider the exact display string provided, but normalized for matching against cleaned Mistral output
    handicap_display_normalized_for_search = handicap_display_from_input.lower().replace(" ", "").replace(",", ".")

    for scroll_iter in range(max_scrolls):
        telegram_log(f"[HYBRID_FOR][SCROLL {scroll_iter+1}] Делаем скриншот и выполняем OCR.", is_debug_message=True)
        current_screenshot = pyautogui.screenshot(region=(x1_region, y1_region, region_width, region_height))

        debug_path = f"debug_hybrid_handicap_scroll_{scroll_iter+1}.png"
        current_screenshot.save(debug_path)
        for chat_id in subscribers:
            send_photo(chat_id, debug_path, caption=f"Поиск форы гибридным методом: Скролл {scroll_iter+1} для {search_text}")

        full_text_mistral, _ = extract_text_mistral_ocr(current_screenshot)
        full_text_mistral_lower = full_text_mistral.lower()

        # Clean Mistral's output string for matching
        full_text_mistral_cleaned_for_match = full_text_mistral_lower.replace(" ", "").replace(",", ".").replace("—", "-")


        handicap_type_present_mistral = "форы" in full_text_mistral_cleaned_for_match or \
                                        "победасучетомфоры" in full_text_mistral_cleaned_for_match

        # Check for handicap value presence using both the exact display string (normalized)
        # and the more flexible regex pattern for the numerical value.
        handicap_value_present_mistral = \
            handicap_display_normalized_for_search in full_text_mistral_cleaned_for_match or \
            loose_handicap_value_pattern.search(full_text_mistral_cleaned_for_match) is not None

        team_present_mistral = False
        team_name_for_mistral_check = "Н/Д"
        if team_name_target:
            if is_fuzzy_match(team_name_target, full_text_mistral_lower): # Using original lower text for fuzzy match
                team_present_mistral = True
                team_name_for_mistral_check = team_name_target

        telegram_log(f"[HYBRID_FOR][SCROLL {scroll_iter+1}] Результат Mistral OCR: Тип форы: {handicap_type_present_mistral}, Команда: {team_present_mistral} ('{team_name_for_mistral_check}'), Значение форы: {handicap_value_present_mistral} ('{handicap_display_from_input}').", is_debug_message=True)

        # Условие для запуска Tesseract: Mistral должен подтвердить наличие значения форы,
        # тип форы, и команды (если указана).
        if handicap_value_present_mistral and (not team_name_target or team_present_mistral) and handicap_type_present_mistral:
            telegram_log(f"[HYBRID_FOR][SCROLL {scroll_iter+1}] Конкретное значение форы '{handicap_display_from_input}' найдено Mistral OCR (и проверка команды/типа пройдена). Переходим к Tesseract для точных координат.", is_debug_message=True)

            full_text_tesseract, tesseract_blocks = extract_text_tesseract(current_screenshot, **BEST_TESSERACT_PARAMS)

            send_ocr_diagnostics_telegram(
                scroll_iter,
                "HYBRID_TESSERACT_BBOX",
                f"Tesseract BBox for: '{search_text}' (scroll {scroll_iter+1})",
                debug_path,
                full_text_tesseract,
                tesseract_blocks,
                found=False,
                extra=f"OCR_PROVIDER: {OCR_PROVIDER}. Match Name: {match_name}. Parsed Team: {team_name_target}. Parsed Handicap: {handicap_display_from_input}"
            )
            if _click_handicap_from_blocks(search_text, tesseract_blocks, team_name_target, x1_region, y1_region):
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