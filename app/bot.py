import time
import threading
import requests
import pyautogui
import os
import re
from PIL import ImageStat, Image
import pyperclip
import queue
import cv2
import numpy as np
import io
import logging
import base64
from mistralai import Mistral

try:
    from mistralai.models.chat import TextChunk
except ImportError:
    TextChunk = None
from typing import Optional, Tuple
import sys
import pytesseract

# --- Глобальные переменные и настройки ---
task_queue = queue.Queue()
is_logged_in = threading.Event()
DEBUG_LOGGING_ENABLED = True
OCR_PROVIDER = "mistral"
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
SUBSCRIBERS_FILE = "subscribers.txt"
subscribers = set()
mistral_client = None

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# --- Инициализация ---
if not TELEGRAM_BOT_TOKEN:
    print("[ERROR] TELEGRAM_BOT_TOKEN не найден!")
    sys.exit(1)
if OCR_PROVIDER == "mistral":
    if not MISTRAL_API_KEY:
        print("[CRITICAL_ERROR] MISTRAL_API_KEY не установлен.")
        sys.exit(1)
    try:
        mistral_client = Mistral(api_key=MISTRAL_API_KEY)
        print("[INFO] Mistral AI client initialized.")
    except Exception as e:
        print(f"[CRITICAL_ERROR] Не удалось инициализировать Mistral AI client: {e}.")
        sys.exit(1)

# --- Функции-утилиты ---


def telegram_log(message, is_debug_message: bool = False):
    if is_debug_message and not DEBUG_LOGGING_ENABLED:
        return
    message_to_send = f"[DEBUG] {message}" if is_debug_message else message
    print(message_to_send)
    for chat_id in set(subscribers):
        send_message(chat_id, message_to_send)


def send_message(chat_id, message):
    try:
        requests.post(
            f"{BASE_URL}/sendMessage",
            data={"chat_id": chat_id, "text": message},
            timeout=10,
        ).raise_for_status()
    except Exception as e:
        print(f"Ошибка отправки сообщения для {chat_id}: {e}")


def verify_screen_color(pixel_coord, target_color, tolerance=10, timeout=30):
    telegram_log(
        f"Проверка цвета {target_color} в точке {pixel_coord}...", is_debug_message=True
    )
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            pixel_color = pyautogui.pixel(pixel_coord[0], pixel_coord[1])
            if all(
                abs(c1 - c2) <= tolerance for c1, c2 in zip(pixel_color, target_color)
            ):
                telegram_log("Цвет совпал. Проверка пройдена.", is_debug_message=True)
                return True
        except Exception as e:
            telegram_log(f"Ошибка при проверке цвета: {e}", is_debug_message=True)
        time.sleep(2)
    telegram_log(
        f"Цвет {target_color} НЕ обнаружен за {timeout} сек.", is_debug_message=False
    )
    return False


def load_subscribers():
    if os.path.exists(SUBSCRIBERS_FILE):
        with open(SUBSCRIBERS_FILE, "r") as f:
            globals()["subscribers"] = {line.strip() for line in f if line.strip()}


def save_subscribers():
    with open(SUBSCRIBERS_FILE, "w") as f:
        for sub in globals().get("subscribers", set()):
            f.write(f"{sub}\n")


# --- Основные функции бота (заглушки) ---
def find_complex_outcome_and_click(
    search_text: str, max_scrolls: int, region: tuple
) -> bool:
    return False


def find_bet_input_coords(timeout=15):
    return None, None


def do_bet_placement(outcome, coef_condition, bet_amount):
    return False


def find_match(match_name):
    pass


def process_bet(task):
    match_name, outcome, coef_condition, bet_amount = task.values()
    telegram_log(f"Начинаю обработку ставки: {match_name} / {outcome}")
    find_match(match_name)
    PREDEFINED = {"1": (483, 570), "x": (543, 570), "2": (606, 570)}
    outcome_lower = outcome.strip().lower()
    if outcome_lower not in PREDEFINED:
        pyautogui.click(277, 526)
        time.sleep(5)
    clicked = False
    if outcome_lower in PREDEFINED:
        pyautogui.click(PREDEFINED[outcome_lower])
        clicked = True
    else:
        if find_complex_outcome_and_click(outcome.strip(), 7, (206, 151, 958, 641)):
            clicked = True
    if clicked and do_bet_placement(outcome, coef_condition, bet_amount):
        telegram_log("Ставка успешно размещена.")
    else:
        telegram_log("Не удалось разместить ставку.")
    pyautogui.click(70, 142)
    time.sleep(3)


# --- Потоки (Producer-Consumer) ---
def worker():
    try:
        open_browser_and_navigate()
        close_initial_popups()
        do_login()
        is_logged_in.set()
        telegram_log("Бот готов к работе.")
    except Exception as e:
        telegram_log(
            f"[FATAL_SETUP_ERROR] Ошибка при настройке: {e}. Worker не будет работать."
        )
        return
    while True:
        try:
            task = task_queue.get()
            process_bet(task)
            task_queue.task_done()
        except Exception as e:
            telegram_log(f"[WORKER_ERROR] Критическая ошибка: {e}")


def poll_updates():
    offset = None
    while True:
        try:
            r = requests.get(
                f"{BASE_URL}/getUpdates",
                params={"timeout": 10, "offset": offset},
                timeout=15,
            ).json()
            for update in r.get("result", []):
                offset = update["update_id"] + 1
                if not (message := update.get("message")) or not (
                    text := message.get("text", "")
                ):
                    continue
                chat_id = str(message["chat"]["id"])
                if text.lower() == "/start":
                    if chat_id not in subscribers:
                        subscribers.add(chat_id)
                        save_subscribers()
                        send_message(chat_id, "Вы подписались на логи бота!")
                else:
                    parts = [part.strip() for part in text.split(",")]
                    if len(parts) != 4:
                        continue
                    try:
                        task = {
                            "match_name": parts[0],
                            "outcome": parts[1],
                            "coef_condition": parts[2],
                            "bet_amount": float(parts[3].replace(",", ".")),
                        }
                        task_queue.put(task)
                        telegram_log(f"Ставка '{parts[1]}' принята в очередь.")
                    except ValueError:
                        pass
        except Exception as e:
            print(f"Ошибка обновлений: {e}")
            time.sleep(5)


# --- Функции инициализации ---
def close_initial_popups():
    telegram_log("Закрытие первоначальных поп-апов...")
    time.sleep(5)
    # Клик по "Cancel" окна перевода
    pyautogui.click(660, 520)
    time.sleep(1)
    # Клик по "Принять" для cookie
    pyautogui.click(930, 690)
    time.sleep(2)
    telegram_log("Поп-апы закрыты.")


def open_browser_and_navigate():
    telegram_log("Запуск браузера и навигация...")
    pyautogui.click(413, 363)
    time.sleep(0.3)
    pyautogui.write(
        "otwyn7rnye-mobile-country-RU-state-524894-city-524901-hold-session-session-68234428f20a1",
        interval=0.05,
    )
    pyautogui.click(410, 396)
    time.sleep(0.3)
    pyautogui.write("kVgpz87hTSt7wsF6")
    pyautogui.press("enter")
    time.sleep(2)
    pyautogui.hotkey("ctrl", "l")
    time.sleep(1)
    pyautogui.write("https://www.marathonbet.ru/", interval=0.05)
    pyautogui.press("enter")

    # ФИНАЛЬНОЕ ИСПРАВЛЕНИЕ: Проверка по стабильному цвету фона левой панели
    TARGET_PIXEL_COORD = (150, 400)  # Точка в левой панели навигации
    TARGET_PIXEL_COLOR = (247, 247, 247)  # Светло-серый/белый цвет фона
    if not verify_screen_color(
        TARGET_PIXEL_COORD, TARGET_PIXEL_COLOR, tolerance=5, timeout=45
    ):
        raise Exception(
            "Не удалось подтвердить загрузку главной страницы по цвету левой панели"
        )
    telegram_log("Главная страница загружена.")


def do_login():
    telegram_log("Выполнение входа...")
    login_button_main = (900, 240)
    pyautogui.click(login_button_main)
    time.sleep(5)

    login_button_popup = (600, 516)
    login_button_popup_color = (0, 170, 140)
    if not verify_screen_color(
        login_button_popup, login_button_popup_color, tolerance=20, timeout=10
    ):
        raise Exception("Не появилось окно для входа.")

    pyautogui.click(600, 377, clicks=2)
    time.sleep(0.3)
    pyautogui.write("9214111699", interval=0.1)
    pyautogui.click(600, 429, clicks=2)
    time.sleep(0.3)
    pyautogui.write("Gamma1488", interval=0.1)
    pyautogui.click(login_button_popup)

    chat_button_coord = (820, 290)
    chat_button_color = (0, 170, 140)
    if not verify_screen_color(
        chat_button_coord, chat_button_color, tolerance=20, timeout=25
    ):
        raise Exception("Не удалось подтвердить успешный вход.")
    telegram_log("Вход выполнен успешно.")
    time.sleep(2)
    pyautogui.click(10, 10)
    time.sleep(1)


def main():
    load_subscribers()
    threading.Thread(target=poll_updates, daemon=True).start()
    threading.Thread(target=worker, daemon=True).start()
    telegram_log("🤖 Бот запущен! Очередь задач активна.")
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
