import pyautogui
from pynput import keyboard

def record_position():
    pos = pyautogui.position()
    print(f"[INFO] Зафиксированы координаты: X: {pos.x}, Y: {pos.y}")

def exit_program():
    print("[INFO] Завершаю работу...")
    listener.stop()

# Настроим глобальные горячие клавиши:
# Ctrl+Shift+R — записать координаты,
# Ctrl+Shift+Q — выйти из программы.
hotkeys = keyboard.GlobalHotKeys({
    '<ctrl>+<shift>+r': record_position,
    '<ctrl>+<shift>+q': exit_program
})

print("[INFO] Глобальные горячие клавиши активны.")
print("Нажмите Ctrl+Shift+R для записи координат курсора.")
print("Нажмите Ctrl+Shift+Q для завершения работы.")

# Запускаем слушатель горячих клавиш
with hotkeys as listener:
    listener.join()