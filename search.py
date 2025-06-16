import os
import pickle
import pyautogui
import pyttsx3
import speech_recognition as sr
from PIL import Image, ImageOps
from pynput import mouse
import torch
import clip
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QThread

# --- Voice Assistant ---
class VoiceAssistant:
    def __init__(self, preferred_gender="male", rate=200):
        self.engine = pyttsx3.init()
        self.set_voice(preferred_gender)
        self.set_rate(rate)
        self.engine.setProperty('volume', 1.0)

    def set_voice(self, preferred_gender):
        voices = self.engine.getProperty('voices')
        selected = None
        if preferred_gender.lower() == "female":
            for v in voices:
                if "zira" in v.name.lower() or "zira" in v.id.lower():
                    selected = v
                    break
        elif preferred_gender.lower() == "male":
            for v in voices:
                if "david" in v.name.lower() or "david" in v.id.lower():
                    selected = v
                    break
        if selected:
            self.engine.setProperty('voice', selected.id)
        else:
            self.engine.setProperty('voice', voices[0].id)

    def set_rate(self, rate):
        self.engine.setProperty('rate', rate)

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

# --- Command Mapping ---
COMMANDS = [
    {
        "name": "search_top_right",
        "keywords": ["top right search", "look at top right for"],
        "function": "search_keyword",
        "region": "top_right"
    },
    {
        "name": "search_top_left",
        "keywords": ["top left search","look at top left for"],
        "function": "search_keyword",
        "region": "top_left"
    },
    {
        "name": "search_bottom_right",
        "keywords": ["bottom right search","look at bottom right for"],
        "function": "search_keyword",
        "region": "bottom_right"
    },
    {
        "name": "search_bottom_left",
        "keywords": ["bottom left search", "look at bottom left for"],
        "function": "search_keyword",
        "region": "bottom_left"
    },
    {
        "name": "search_top",
        "keywords": ["top search", "look at top for"],
        "function": "search_keyword",
        "region": "top"
    },
    {
        "name": "search_bottom",
        "keywords": ["bottom search", "look at bottom for"],
        "function": "search_keyword",
        "region": "bottom"
    },
    {
        "name": "search",
        "keywords": [
            "search", "find", "look for", "please search", "could you search", "would you kindly search"
        ],
        "function": "search_keyword"
    },
    {
        "name": "incorrect",
        "keywords": [
            "incorrect", "not correct", "wrong", "that was wrong", "no, wrong", "this is wrong", "try again", "fix it"
        ],
        "function": "mark_incorrect"
    },
]

def map_command(text):
    text = text.lower()
    for cmd in COMMANDS:
        for kw in cmd["keywords"]:
            if kw in text:
                region = cmd.get("region", None)
                return cmd["function"], text.replace(kw, "").strip(), region
    return None, text, None

# --- Region Utilities ---
def get_region_bounds(region, screen_w, screen_h):
    if region == "top":
        return (0, 0, screen_w, screen_h // 2)
    elif region == "bottom":
        return (0, screen_h // 2, screen_w, screen_h)
    elif region == "top_left":
        return (0, 0, screen_w // 2, screen_h // 2)
    elif region == "top_right":
        return (screen_w // 2, 0, screen_w, screen_h // 2)
    elif region == "bottom_left":
        return (0, screen_h // 2, screen_w // 2, screen_h)
    elif region == "bottom_right":
        return (screen_w // 2, screen_h // 2, screen_w, screen_h)
    else:
        return (0, 0, screen_w, screen_h)  # full screen

def take_screenshot(path="temp_screenshot.png", region=None):
    screen_w, screen_h = pyautogui.size()
    if region is not None:
        left, top, right, bottom = get_region_bounds(region, screen_w, screen_h)
        width = right - left
        height = bottom - top
        img = pyautogui.screenshot(region=(left, top, width, height))
        img.save(path)
    else:
        pyautogui.screenshot(path)
    return path

def take_rectangular_screenshot(center_x, center_y, width=512, height=48, out_size=224, path="temp_crop.png", base_img=None, region_offset=(0,0)):
    if base_img is not None:
        img = base_img
        screen_w, screen_h = img.size
        offset_x, offset_y = region_offset
    else:
        img = pyautogui.screenshot()
        screen_w, screen_h = pyautogui.size()
        offset_x, offset_y = 0, 0

    left = max(center_x - width // 2 - offset_x, 0)
    top = max(center_y - height // 2 - offset_y, 0)
    right = min(center_x + width // 2 - offset_x, screen_w)
    bottom = min(center_y + height // 2 - offset_y, screen_h)
    crop = img.crop((left, top, right, bottom))
    crop = ImageOps.fit(crop, (out_size, out_size), Image.LANCZOS)
    crop.save(path)
    return path

# --- CLIP Recognizer ---
class CLIPScreenRecognizer:
    def __init__(self, device="cpu"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.db_path = "clip_screen_db.pkl"
        self.data = []
        self.load()

    def get_embedding(self, image_path):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(image)
            embedding /= embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy()[0]

    def find_best_match(self, embedding, threshold=0.90):
        if not self.data:
            return None, 0.0, None
        similarities = [float(np.dot(embedding, d["embedding"])) for d in self.data]
        best_idx = int(np.argmax(similarities))
        best_sim = similarities[best_idx]
        if best_sim >= threshold:
            return self.data[best_idx], best_sim, best_idx
        return None, best_sim, None

    def add_screen(self, embedding, coords, name):
        self.data.append({"embedding": embedding, "coords": coords, "name": name})
        self.save()

    def get_all_screen_names(self):
        return sorted(set(d["name"] for d in self.data))

    def save(self):
        with open(self.db_path, "wb") as f:
            pickle.dump(self.data, f)

    def load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                self.data = pickle.load(f)

# --- Mouse/Keyboard Automation ---
def move_and_double_click(percent_x, percent_y):
    screen_w, screen_h = pyautogui.size()
    x = int(screen_w * percent_x)
    y = int(screen_h * percent_y)
    pyautogui.moveTo(x, y, duration=0.4)
    pyautogui.doubleClick()
    return x, y

def type_and_search(keyword):
    import pyautogui
    try:
        pyautogui.hotkey('ctrl', 'a')
        pyautogui.press('backspace')
        pyautogui.write(keyword, interval=0.1)
        pyautogui.press('enter')
    except pyautogui.FailSafeException:
        return "PyAutoGUI fail-safe triggered! Move your mouse away from the screen corner."
    return None

def get_double_click_coordinates(va, return_pixel=False, region=None):
    screen_w, screen_h = pyautogui.size()
    if region is not None:
        left, top, right, bottom = get_region_bounds(region, screen_w, screen_h)
    else:
        left, top, right, bottom = 0, 0, screen_w, screen_h
    clicks = []

    def on_click(x, y, button, pressed):
        if pressed and button == mouse.Button.left:
            if left <= x < right and top <= y < bottom:
                clicks.append((x, y))
            if len(clicks) >= 2:
                return False

    with mouse.Listener(on_click=on_click) as listener:
        listener.join()

    if clicks:
        x, y = clicks[-1]
        percent_x = x / screen_w
        percent_y = y / screen_h
        if return_pixel:
            return percent_x, percent_y, x, y
        else:
            return percent_x, percent_y
    else:
        if return_pixel:
            return None, None, None, None
        else:
            return None, None

# --- Worker Thread for Heavy Search ---
class SearchWorker(QThread):
    result_ready = pyqtSignal(str)
    request_double_click = pyqtSignal(str)

    def __init__(self, assistant, keyword, region):
        super().__init__()
        self.assistant = assistant
        self.keyword = keyword
        self.region = region

    def run(self):
        result = self.assistant._search_keyword_heavy(self.keyword, self.region)
        if result is not None:
            self.result_ready.emit(result)
        # If result is None, the assistant is waiting for user input (double-click)

# --- PyQt5 GUI Integration for Assistant ---
class VoiceSearchAssistant(QObject):
    request_screen_name = pyqtSignal(list)
    screen_name_selected = pyqtSignal(str)
    request_double_click = pyqtSignal(str)
    double_click_received = pyqtSignal(float, float, int, int)
    result_ready = pyqtSignal(str)

    def __init__(self, preferred_gender="male", rate=200, parent=None):
        super().__init__(parent)
        self.va = VoiceAssistant(preferred_gender, rate)
        self.recognizer = CLIPScreenRecognizer(device="cpu")
        self.last_screenshot = None
        self.last_embedding = None
        self.last_keyword = None
        self.last_prediction = None
        self.last_screen_idx = None
        self.crop_width = 512
        self.crop_height = 48
        self.clip_input_size = 224
        self._pending_screen_name = None
        self._pending_double_click = None
        self._pending_region = None
        self._pending_keyword = None
        self._pending_region_img = None
        self._pending_left = None
        self._pending_top = None
        self._pending_embedding = None
        self._pending_coords = None
        self.learning_in_progress = False

        self.screen_name_selected.connect(self._on_screen_name_selected)
        self.double_click_received.connect(self._on_double_click_received)

    def set_voice(self, gender):
        self.va.set_voice(gender)

    def set_rate(self, rate):
        self.va.set_rate(rate)

    def listen_and_handle(self, command=None):
        if self.learning_in_progress:
            return "Please finish teaching the current screen first."
        if not command:
            return "No command recognized."
        func, arg, region = map_command(command)
        if func == "search_keyword":
            self.worker = SearchWorker(self, arg, region)
            self.worker.result_ready.connect(self.result_ready.emit)
            self.worker.start()
            return "Processing search..."
        elif func == "mark_incorrect":
            return self.mark_incorrect()
        elif "exit" in command or "quit" in command:
            return "Exiting."
        else:
            return "Unknown command."

    def _search_keyword_heavy(self, keyword, region=None):
        keyword = keyword.strip('" ')
        self.last_keyword = keyword

        screen_w, screen_h = pyautogui.size()
        self.last_screenshot = take_screenshot(region=region)
        self.last_embedding = None

        left, top, right, bottom = get_region_bounds(region, screen_w, screen_h)
        region_offset = (left, top)
        region_img = Image.open(self.last_screenshot)

        best_match = None
        best_sim = 0.0
        best_idx = None

        for idx, d in enumerate(self.recognizer.data):
            percent_x, percent_y = d["coords"]
            x = int(screen_w * percent_x)
            y = int(screen_h * percent_y)
            if not (left <= x < right and top <= y < bottom):
                continue
            local_x = x - left
            local_y = y - top
            crop_path = take_rectangular_screenshot(
                local_x, local_y,
                width=self.crop_width,
                height=self.crop_height,
                out_size=self.clip_input_size,
                path="temp_crop.png",
                base_img=region_img,
                region_offset=(0, 0)
            )
            embedding = self.recognizer.get_embedding(crop_path)
            match, sim, _ = self.recognizer.find_best_match(embedding)
            if sim > best_sim:
                best_sim = sim
                best_match = d
                best_idx = idx

        if best_match and best_sim >= 0.90:
            percent_x, percent_y = best_match["coords"]
            self.last_prediction = (percent_x, percent_y)
            self.last_screen_idx = best_idx
            move_and_double_click(percent_x, percent_y)
            fail_safe_msg = type_and_search(keyword)
            if fail_safe_msg:
                return fail_safe_msg
            self.va.speak(f"Searching for {keyword}")
            if os.path.exists("temp_crop.png"):
                os.remove("temp_crop.png")
            if os.path.exists(self.last_screenshot):
                os.remove(self.last_screenshot)
            return f"Searched for: {keyword}"
        else:
            self.learning_in_progress = True
            self.va.speak("This is an unknown screen. Please show me the correct location.")
            self._pending_region = region
            self._pending_keyword = keyword
            self._pending_region_img = region_img
            self._pending_left = left
            self._pending_top = top
            self._pending_embedding = None
            self._pending_coords = None
            self.request_double_click.emit(region if region else "")
            return None  # Do not emit a result yet

    def _on_double_click_received(self, percent_x, percent_y, x, y):
        if not self.learning_in_progress:
            return
        left = self._pending_left
        top = self._pending_top
        region_img = self._pending_region_img
        region = self._pending_region
        keyword = self._pending_keyword

        if percent_x is not None and percent_y is not None:
            local_x = x - left
            local_y = y - top
            crop_path = take_rectangular_screenshot(
                local_x, local_y,
                width=self.crop_width,
                height=self.crop_height,
                out_size=self.clip_input_size,
                path="temp_crop.png",
                base_img=region_img,
                region_offset=(0, 0)
            )
            embedding = self.recognizer.get_embedding(crop_path)
            existing_names = self.recognizer.get_all_screen_names()
            self._pending_embedding = embedding
            self._pending_coords = (percent_x, percent_y)
            self.request_screen_name.emit(existing_names)
        else:
            self.va.speak("No click detected. Please try again.")
            if os.path.exists("temp_crop.png"):
                os.remove("temp_crop.png")
            if os.path.exists(self.last_screenshot):
                os.remove(self.last_screenshot)
            self.learning_in_progress = False
            self.result_ready.emit("No click detected. Please try again.")

    def _on_screen_name_selected(self, screen_name):
        if not self.learning_in_progress:
            return
        embedding = self._pending_embedding
        coords = self._pending_coords
        self.recognizer.add_screen(embedding, coords, screen_name)
        self.va.speak("Thank you. I have learned the correct location for this screen.")
        if os.path.exists("temp_crop.png"):
            os.remove("temp_crop.png")
        if os.path.exists(self.last_screenshot):
            os.remove(self.last_screenshot)
        self.last_screenshot = None
        self.last_embedding = None
        self.last_prediction = None
        self.last_screen_idx = None
        self.learning_in_progress = False

        # --- NEW: Re-run the search for the pending keyword and region ---
        if self._pending_keyword is not None:
            # Run the search in a worker thread to avoid blocking
            self.worker = SearchWorker(self, self._pending_keyword, self._pending_region)
            self.worker.result_ready.connect(self.result_ready.emit)
            self.worker.start()
            # Clear pending values
            self._pending_keyword = None
            self._pending_region = None
        else:
            self.result_ready.emit("Learned new screen location.")

    def mark_incorrect(self):
        if self.learning_in_progress:
            return "Please finish teaching the current screen first."
        if self.last_screenshot is None or self.last_prediction is None:
            self.va.speak("No previous prediction to correct.")
            return "No previous prediction to correct."
        self._pending_region = None
        self._pending_keyword = None
        self._pending_region_img = None
        self._pending_left = 0
        self._pending_top = 0
        self.learning_in_progress = True
        self.request_double_click.emit("")
        return "Please double-click on the correct search bar location."

# --- Speech Recognizer ---
class InterruptibleRecognizer:
    def __init__(self, callback):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.callback = callback
        self.stopper = None
        self.listening = False

    def start(self):
        if not self.listening:
            self.stopper = self.recognizer.listen_in_background(
                self.microphone, self.callback, phrase_time_limit=8
            )
            self.listening = True

    def stop(self):
        if self.listening and self.stopper:
            self.stopper(wait_for_stop=False)
            self.listening = False