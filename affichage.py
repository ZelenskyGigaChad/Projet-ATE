import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from langdetect import detect
from googletrans import Translator
import threading
import time
from screeninfo import get_monitors



face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
num_frames_to_average = 20
face_positions = []




text_file_path = "texte.txt"
def update_displayed_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading the file: {e}")
        return ""
font = ImageFont.truetype("LEMONMILK-Bold.otf", 44)
last_known_x, last_known_y = None, None
decay_factor = 0.05
def wrap_text(text, max_letters_per_line=15):
    lines = []
    current_line = []
    for word in text.split():
        if len(" ".join(current_line + [word])) <= max_letters_per_line:
            current_line.append(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))
    return "\n".join(lines)
translator = Translator()
translated_text = ""
translated_text_color = (0, 255, 0)
def translate(text):
    try:
        translation = translator.translate(text, src='auto', dest='fr')
        return translation.text
    except Exception as e:
        print(f"Error translating text: {e}")
        return text




def translate_thread():
    global translated_text
    while True:
        displayed_text = update_displayed_text(text_file_path)
        detected_language = detect(displayed_text)
        if detected_language != 'fr':
            try:
                translation = translator.translate(displayed_text, src='auto', dest='fr')
                translated_text = translation.text
            except Exception as e:
                print(f"Error translating text: {e}")
                translated_text = displayed_text
        else:
            translated_text = ""
        time.sleep(5)
translation_thread = threading.Thread(target=translate_thread)
translation_thread.daemon = True
translation_thread.start()
monitors = get_monitors()
default_screen_index = 0
cv2.namedWindow('Face Tracking - Original', cv2.WINDOW_NORMAL)
cv2.namedWindow('Face Tracking - Flipped', cv2.WINDOW_NORMAL)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    black_background = Image.new("RGBA", (frame.shape[1], frame.shape[0]), (0, 0, 0, 255))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_positions.append((x, y))
        if len(face_positions) > num_frames_to_average:
            face_positions.pop(0)
    if len(face_positions) > 0:
        avg_x = int(sum([pos[0] for pos in face_positions]) / len(face_positions))
        avg_y = int(sum([pos[1] for pos in face_positions]) / len(face_positions))
    else:
        avg_x, avg_y = None, None
    if last_known_x is not None and last_known_y is not None and avg_x is not None and avg_y is not None:
        last_known_x = int(last_known_x + (avg_x - last_known_x) * decay_factor)
        last_known_y = int(last_known_y + (avg_y - last_known_y) * decay_factor)
    elif avg_x is not None and avg_y is not None:
        last_known_x, last_known_y = avg_x, avg_y
    displayed_text = update_displayed_text(text_file_path)
    wrapped_text = wrap_text(displayed_text)
    img_pil = Image.fromarray(frame)
    img_pil.paste(black_background, (0, 0), black_background)
    draw = ImageDraw.Draw(img_pil)
    original_text_color = (255, 255, 255)
    if last_known_x is not None and last_known_y is not None:
        draw.text((last_known_x, last_known_y - 10), wrapped_text, font=font, fill=original_text_color)
        if translated_text:
            translated_text_wrapped = wrap_text(translated_text, max_letters_per_line=15)
            draw.text((last_known_x, last_known_y + 90), translated_text_wrapped, font=font, fill=translated_text_color)
    frame = np.array(img_pil)
    frame_flipped = cv2.flip(frame, 1)
    cv2.imshow('Face Tracking - Original', frame)
    cv2.imshow('Face Tracking - Flipped', frame_flipped)
    if len(monitors) > 1:
        cv2.moveWindow('Face Tracking - Flipped', monitors[1].x, monitors[1].y)
        cv2.setWindowProperty('Face Tracking - Flipped', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.moveWindow('Face Tracking - Original', 0, 0)
        cv2.setWindowProperty('Face Tracking - Original', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow('Face Tracking - Flipped', 0, 0)
        cv2.setWindowProperty('Face Tracking - Flipped', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break











cap.release()
cv2.destroyAllWindows()
