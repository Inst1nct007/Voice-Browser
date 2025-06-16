import sys
import math
import numpy as np
import sounddevice as sd
import threading
from search import VoiceSearchAssistant, InterruptibleRecognizer, get_double_click_coordinates
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QLabel, QComboBox,
    QSlider, QTextEdit, QSizePolicy, QPushButton, QHBoxLayout, QDialog, QDialogButtonBox,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QSystemTrayIcon, QMenu, QAction, QStyle
)
from PyQt5.QtCore import Qt, QTimer, QPointF, QThread, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QPen, QPainterPath, QLinearGradient, QBrush, QIcon

# --- Dialog for Screen Name Selection ---
class ScreenNameDialog(QDialog):
    def __init__(self, existing_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Screen Name")
        self.setModal(True)
        self.selected_name = None

        layout = QVBoxLayout(self)
        label = QLabel("Select a screen name or add a new one:")
        layout.addWidget(label)

        self.list_widget = QListWidget()
        for name in existing_names:
            item = QListWidgetItem(name)
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)

        self.new_name_edit = QLineEdit()
        self.new_name_edit.setPlaceholderText("Or enter a new screen name...")
        layout.addWidget(self.new_name_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accepted)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accepted(self):
        if self.new_name_edit.text().strip():
            self.selected_name = self.new_name_edit.text().strip()
            self.accept()
        elif self.list_widget.currentItem():
            self.selected_name = self.list_widget.currentItem().text()
            self.accept()
        else:
            QMessageBox.warning(self, "No Selection", "Please select or enter a screen name.")

    @staticmethod
    def get_screen_name(existing_names, parent=None):
        dialog = ScreenNameDialog(existing_names, parent)
        result = dialog.exec_()
        return dialog.selected_name if result == QDialog.Accepted else None

# --- Audio Thread ---
class AudioStreamThread(QThread):
    audio_data = pyqtSignal(np.ndarray)

    def __init__(self, chunk=1024, rate=16000):
        super().__init__()
        self.chunk = chunk
        self.rate = rate
        self.running = True
        self.paused = False
        self._lock = threading.Lock()

    def run(self):
        def callback(indata, frames, time, status):
            with self._lock:
                if not self.running or self.paused:
                    raise sd.CallbackStop()
            data = indata[:, 0]
            self.audio_data.emit(data.copy())

        while self.running:
            with self._lock:
                if self.paused:
                    self._lock.release()
                    self.msleep(100)
                    self._lock.acquire()
                    continue
            try:
                with sd.InputStream(channels=1, samplerate=self.rate, blocksize=self.chunk, callback=callback):
                    while self.running and not self.paused:
                        sd.sleep(100)
            except Exception:
                pass

    def stop(self):
        with self._lock:
            self.running = False
            self.paused = False

    def pause(self):
        with self._lock:
            self.paused = True

    def resume(self):
        with self._lock:
            self.paused = False

# --- Audio Recognizer Widget ---
class AudioRecognizerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.n_points = 64
        self.phase = 0
        self.idle = True
        self.wave = [0] * self.n_points
        self.volume = 0

        self.target_freq = 1.0
        self.target_amp = 0.15
        self.target_stretch = 1.0
        self.smooth_freq = 1.0
        self.smooth_amp = 0.15
        self.smooth_stretch = 1.0

        self.glow_offsets = [0.0, 0.7, 1.4, 2.1]

        self.audio_thread = AudioStreamThread()
        self.audio_thread.audio_data.connect(self.process_audio)
        self.audio_thread.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(40)  # Reduced frequency for less CPU

    def process_audio(self, data):
        self.volume = np.sqrt(np.mean(data ** 2))
        self.idle = self.volume < 0.01

    def pause(self):
        self.audio_thread.pause()

    def resume(self):
        self.audio_thread.resume()

    def animate(self):
        if self.idle:
            self.target_freq = 1.5
            self.target_amp = 0.1
            self.target_stretch = 0.1
        else:
            self.target_freq = 1.0 + min(self.volume * 12, 10.0)
            self.target_amp = 0.15 + min(self.volume * 2, 0.5)
            self.target_stretch = 0.5 + min(self.volume, 2.0)

        lerp_speed = 0.05
        self.smooth_freq += (self.target_freq - self.smooth_freq) * lerp_speed
        self.smooth_amp += (self.target_amp - self.smooth_amp) * lerp_speed
        self.smooth_stretch += (self.target_stretch - self.smooth_stretch) * lerp_speed

        self.phase += 0.12
        self.wave = [
            math.sin(self.phase + i * self.smooth_freq * 0.4 * self.smooth_stretch) * self.smooth_amp
            for i in range(self.n_points)
        ]
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        grad_colors = [QColor(0, 234, 255), QColor(0, 191, 255), QColor(255, 79, 216)]
        n = self.n_points

        glow_layers = [
            (60, 40, 0.0, 1.0, 1.0),
            (40, 80, 0.7, 0.92, 1.12),
            (20, 140, 1.4, 1.08, 0.92),
            (10, 220, 2.1, 1.15, 1.18),
        ]
        for alpha, thickness, phase_offset, freq_mult, amp_mult in glow_layers:
            glow_wave = [
                math.sin(self.phase + phase_offset + i * self.smooth_freq * 0.4 * self.smooth_stretch * freq_mult) *
                self.smooth_amp * amp_mult
                for i in range(self.n_points)
            ]
            for i in range(n - 1):
                t = i / (n - 2)
                if t < 0.5:
                    c = QColor(0, 234, 255)
                elif t < 0.75:
                    c = QColor(0, 191, 255)
                else:
                    c = QColor(255, 79, 216)
                c.setAlpha(alpha)
                pen = QPen(c, thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                painter.setPen(pen)
                painter.drawLine(
                    QPointF(w * i / (n - 1), h / 2 - glow_wave[i] * h / 2),
                    QPointF(w * (i + 1) / (n - 1), h / 2 - glow_wave[i + 1] * h / 2)
                )

        for layer in range(4, 0, -1):
            alpha = int(40 * (layer / 4))
            amp = 1.0 * (0.7 + 0.3 * (layer / 4))
            path = QPainterPath()
            for i, v in enumerate(self.wave):
                x = w * i / (n - 1)
                y = h / 2 - v * h / 2 * amp
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            for i in reversed(range(n)):
                x = w * i / (n - 1)
                y = h / 2 + self.wave[i] * h / 2 * amp
                path.lineTo(x, y)
            path.closeSubpath()
            grad = QLinearGradient(0, h / 2, w, h / 2)
            grad.setColorAt(0, QColor(0, 234, 255, alpha))
            grad.setColorAt(0.5, QColor(0, 191, 255, int(alpha * 0.7)))
            grad.setColorAt(1, QColor(255, 79, 216, alpha))
            painter.fillPath(path, QBrush(grad))

        for layer in range(2, 0, -1):
            alpha = int(180 * (layer / 2))
            thickness = 8 * (layer / 2)
            amp = 1.0
            for i in range(n - 1):
                t = i / (n - 2)
                if t < 0.5:
                    c = QColor(0, 234, 255)
                elif t < 0.75:
                    c = QColor(0, 191, 255)
                else:
                    c = QColor(255, 79, 216)
                c.setAlpha(alpha)
                pen = QPen(c, thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                painter.setPen(pen)
                painter.drawLine(
                    QPointF(w * i / (n - 1), h / 2 - self.wave[i] * h / 2 * amp),
                    QPointF(w * (i + 1) / (n - 1), h / 2 - self.wave[i + 1] * h / 2 * amp)
                )

    def closeEvent(self, event):
        self.audio_thread.stop()
        self.audio_thread.wait()
        super().closeEvent(event)

    def show_main_window(self):
        self.show()
        self.raise_()
        self.activateWindow()

    def on_tray_icon_activated(self, reason):
        if reason == QSystemTrayIcon.DoubleClick or reason == QSystemTrayIcon.Trigger:
            self.show_main_window()

    def quit_app(self):
        self._allow_close = True
        self.close()

# --- Voice Assistant Thread ---
class VoiceAssistantThread(QThread):
    result_signal = pyqtSignal(str)
    request_screen_name = pyqtSignal(list)
    request_double_click = pyqtSignal(str)

    def __init__(self, assistant):
        super().__init__()
        self.assistant = assistant
        self._should_exit = False
        self._muted = False
        self._last_command = None
        self._lock = threading.Lock()
        self.recognizer = InterruptibleRecognizer(self.callback)
        self.assistant.result_ready.connect(self.result_signal.emit)
        self.assistant.request_screen_name.connect(self.request_screen_name.emit)
        self.assistant.request_double_click.connect(self.request_double_click.emit)

    def callback(self, recognizer, audio):
        try:
            command = recognizer.recognize_google(audio).lower()
            with self._lock:
                self._last_command = command
        except Exception:
            pass

    def run(self):
        self.recognizer.start()
        while not self._should_exit:
            if self._muted:
                self.recognizer.stop()
                self.msleep(200)
                continue
            if not self.recognizer.listening:
                self.recognizer.start()
            with self._lock:
                if self._last_command:
                    command = self._last_command
                    self._last_command = None
                else:
                    command = None
            if command:
                result = self.assistant.listen_and_handle(command)
                if result and result != "Processing search...":
                    self.result_signal.emit(result)
                if result == "Exiting.":
                    self._should_exit = True
            self.msleep(100)

    def pause(self):
        self._muted = True

    def resume(self):
        self._muted = False

    def stop(self):
        self._should_exit = True
        self.recognizer.stop()

# --- Main Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        screen = QApplication.primaryScreen()
        size = screen.size()
        width = int(size.width() * 0.3)
        height = int(size.height() * 0.7)
        self.setFixedSize(width, height)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowMaximizeButtonHint)
        self.setWindowTitle("Voice Browser - Browse Hands Free")
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #020e24;
                color: #f5f1ed;
                font-family: 'Segoe UI', 'Arial', sans-serif;
            }
            QTabBar::tab {
                background: #021a3a;
                color: #f5f1ed;
                padding: 10px 24px;
                font-size: 20px;
                border-radius: 14px;
                margin: 0 10px;
                min-width: 120px;
                min-height: 36px;
                max-height: 48px;
            }
            QTabBar::tab:selected {
                background: #00bfff;
                color: #fff;
            }
            QLabel, QComboBox, QSlider, QTextEdit {
                font-size: 24px;
            }
            QComboBox {
                background: #021a3a;
                border: 2px solid #00eaff;
                border-radius: 10px;
                padding: 10px 20px;
                color: #fff;
                min-width: 180px;
            }
            QComboBox QAbstractItemView {
                background: #021a3a;
                color: #fff;
                selection-background-color: #00bfff;
            }
            QSlider::groove:horizontal {
                border: 1px solid #00eaff;
                height: 16px;
                background: #021a3a;
                border-radius: 8px;
            }
            QSlider::handle:horizontal {
                background: #00eaff;
                border: 2px solid #fff;
                width: 32px;
                margin: -10px 0;
                border-radius: 16px;
            }
            QTextEdit {
                background: #021a3a;
                border: none;
                color: #fff;
                border-radius: 10px;
                padding: 16px;
            }
        """)
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setMovable(False)
        self.setCentralWidget(self.tabs)
        self.init_tabs()
        self._allow_close = False

        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        self.tray_icon.setToolTip("Voice Browser")

        tray_menu = QMenu()
        restore_action = QAction("Show", self)
        quit_action = QAction("Exit", self)
        tray_menu.addAction(restore_action)
        tray_menu.addAction(quit_action)
        self.tray_icon.setContextMenu(tray_menu)

        restore_action.triggered.connect(self.show_main_window)
        quit_action.triggered.connect(self.quit_app)
        self.tray_icon.activated.connect(self.on_tray_icon_activated)
        self.tray_icon.show()

        # --- Voice Assistant Integration ---
        self.assistant = VoiceSearchAssistant(preferred_gender="female")
        self.assistant_thread = VoiceAssistantThread(self.assistant)
        self.assistant_thread.result_signal.connect(self.display_result)
        self.assistant_thread.request_screen_name.connect(self.handle_screen_name_request)
        self.assistant_thread.request_double_click.connect(self.handle_double_click_request)
        self.assistant.result_ready.connect(self.display_result)
        self.assistant_thread.start()
        self.mic_muted = False

    def show_main_window(self):
        self.show()
        self.raise_()
        self.activateWindow()

    def quit_app(self):
        self._allow_close = True
        self.close()

    def on_tray_icon_activated(self, reason):
        if reason == QSystemTrayIcon.DoubleClick or reason == QSystemTrayIcon.Trigger:
            self.show_main_window()

    def on_voice_changed(self, value):
        self.assistant.set_voice(value)

    def on_speed_changed(self, value):
        self.assistant.set_rate(value)

    def init_tabs(self):
        home_tab = QWidget()
        home_layout = QVBoxLayout()
        home_layout.setAlignment(Qt.AlignCenter)
        self.audio_widget = AudioRecognizerWidget()
        self.audio_widget.setMinimumHeight(220)
        self.audio_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        home_layout.addWidget(self.audio_widget)
        home_layout.addStretch()
        mic_layout = QHBoxLayout()
        mic_layout.addStretch()
        self.mic_button = QPushButton("🎤")
        self.mic_button.setFixedSize(48, 48)
        self.mic_button.setStyleSheet("""
            QPushButton {
                background: #021a3a;
                color: #00eaff;
                border-radius: 24px;
                font-size: 28px;
                border: 2px solid #00eaff;
            }
            QPushButton:hover {
                background: #00eaff;
                color: #021a3a;
            }
        """)
        self.mic_button.setToolTip("Mute/Unmute Microphone")
        self.mic_button.clicked.connect(self.toggle_mic)
        mic_layout.addWidget(self.mic_button)
        home_layout.addLayout(mic_layout)
        self.result_label = QLabel("")
        self.result_label.setStyleSheet("color: #00eaff; font-size: 20px;")
        home_layout.addWidget(self.result_label)
        home_tab.setLayout(home_layout)
        self.tabs.addTab(home_tab, "Home")

        settings_tab = QWidget()
        settings_layout = QVBoxLayout()
        settings_layout.setAlignment(Qt.AlignTop)
        voice_label = QLabel("Voice:")
        settings_layout.addWidget(voice_label)
        self.voice_combo = QComboBox()
        self.voice_combo.addItems(["Female", "Male"])
        self.voice_combo.setCurrentIndex(0)
        self.voice_combo.currentTextChanged.connect(self.on_voice_changed)
        settings_layout.addWidget(self.voice_combo)
        speed_label = QLabel("Voice Speed:")
        settings_layout.addWidget(speed_label)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(100)
        self.speed_slider.setMaximum(250)
        self.speed_slider.setValue(175)
        self.speed_slider.valueChanged.connect(self.on_speed_changed)
        settings_layout.addWidget(self.speed_slider)
        settings_tab.setLayout(settings_layout)
        self.tabs.addTab(settings_tab, "Settings")

        help_tab = QWidget()
        help_layout = QVBoxLayout()
        help_layout.setAlignment(Qt.AlignTop)
        help_label = QLabel("Available Commands:")
        help_layout.addWidget(help_label)
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setText(
            "search <keyword>\n"
            "incorrect\n"
            "exit\n"
            "You can also say: 'find', 'look for', 'fix it', etc.\n\n"
            "Suggestions: kingshuk.saha.uni@gmail.com or "
        )
        help_layout.addWidget(help_text)
        help_tab.setLayout(help_layout)
        self.tabs.addTab(help_tab, "Help")

    def toggle_mic(self):
        if not self.mic_muted:
            self.assistant_thread.pause()
            self.audio_widget.pause()
            self.mic_button.setText("🔇")
            self.mic_muted = True
        else:
            self.assistant_thread.resume()
            self.audio_widget.resume()
            self.mic_button.setText("🎤")
            self.mic_muted = False

    def display_result(self, result):
        self.result_label.setText(result)
        if result == "Exiting.":
            self.mic_button.setText("🔇")

    def handle_screen_name_request(self, existing_names):
        name = ScreenNameDialog.get_screen_name(existing_names, self)
        if name:
            self.assistant.screen_name_selected.emit(name)
        else:
            self.display_result("Screen naming cancelled.")

    def handle_double_click_request(self, region):
        QMessageBox.information(self, "Show Search Bar",
            f"Please double-click on empty search bar {(' in the ' + region.replace('_', ' ') + ' region') if region else ''}.\n"
            "You have 10 seconds to double-click on the screen."
        )
        def get_click():
            percent_x, percent_y, x, y = get_double_click_coordinates(None, return_pixel=True, region=region if region else None)
            self.assistant.double_click_received.emit(percent_x, percent_y, x, y)
        threading.Thread(target=get_click, daemon=True).start()

    def closeEvent(self, event):
        if self._allow_close:
            # Your cleanup code here
            super().closeEvent(event)
        else:
            event.ignore()
            self.hide()
            self.tray_icon.showMessage(
                "Voice Browser",
                "App is still running in the background. Double-click the tray icon to restore.",
                QSystemTrayIcon.Information,
                2000
            )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())