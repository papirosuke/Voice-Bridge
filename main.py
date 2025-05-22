import sys
import threading
import threading
import queue
import time
import traceback
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit,
                             QLabel, QComboBox, QSpinBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtCore import pyqtSignal
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import deepl
from googletrans import Translator as GoogleTranslator
import torch

DEEPL_API_KEY = ""
deepL_translator = deepl.Translator(DEEPL_API_KEY)
model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
audio_queue = queue.Queue()
recording_flag = threading.Event()

# Silero VAD
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
(get_speech_timestamps, _, _, _, _) = utils

lang_map = {
    "日本語": "ja",
    "中国語": "zh",
    "英語": "en",
    "ドイツ語": "de",
    "フランス語": "fr"
}

def pad_or_trim(audio, length=16000 * 20):
    """Whisperが要求するサイズに合わせて音声をパディングまたはトリミングする"""
    if len(audio) > length:
        return audio[:length]
    elif len(audio) < length:
        return np.pad(audio, (0, length - len(audio)))
    return audio

class TranslatorApp(QWidget):
    append_text_signal = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Language Real-Time Translator")
        self.append_text_signal.connect(self.safe_append_text)
        self.font_size = 12
        self.selected_langs = []
        self.lang_target_map = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.device_box = QComboBox()
        self.device_box.addItems([d["name"] for d in sd.query_devices() if d['max_input_channels'] > 0])
        layout.addWidget(QLabel("音声入力デバイス:"))
        layout.addWidget(self.device_box)

        self.lang_count_box = QSpinBox()
        self.lang_count_box.setRange(1, 3)
        self.lang_count_box.setValue(1)
        self.lang_count_box.valueChanged.connect(self.update_language_boxes)
        layout.addWidget(QLabel("検出する言語数 (最大3):"))
        layout.addWidget(self.lang_count_box)

        self.lang_boxes = []
        self.target_boxes = []
        self.language_box_layout = QVBoxLayout()
        layout.addLayout(self.language_box_layout)
        self.update_language_boxes()

        self.text_area = QTextEdit()
        self.text_area.setFont(QFont("Arial", self.font_size))
        layout.addWidget(self.text_area)

        self.translate_button = QPushButton("翻訳開始")
        self.translate_button.clicked.connect(self.toggle_translation)
        layout.addWidget(self.translate_button)

        self.setLayout(layout)
        self.installEventFilter(self)
        
    def safe_append_text(self, text):
        self.text_area.append(text)

    def update_language_boxes(self):
        while self.language_box_layout.count():
            item = self.language_box_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.lang_boxes.clear()
        self.target_boxes.clear()

        for i in range(self.lang_count_box.value()):
            hbox = QHBoxLayout()
            lang_box = QComboBox()
            lang_box.addItems(list(lang_map.keys()))
            self.lang_boxes.append(lang_box)
            hbox.addWidget(QLabel(f"言語{i + 1}:"))
            hbox.addWidget(lang_box)

            target_box = QComboBox()
            target_box.addItems(list(lang_map.keys())) 
            self.target_boxes.append(target_box)
            hbox.addWidget(QLabel("→"))
            hbox.addWidget(target_box)

            self.language_box_layout.addLayout(hbox)

    def eventFilter(self, source, event):
        if event.type() == event.Wheel and QApplication.keyboardModifiers() == Qt.ControlModifier:
            delta = event.angleDelta().y()
            self.font_size = max(6, self.font_size + (1 if delta > 0 else -1))
            self.text_area.setFont(QFont("Arial", self.font_size))
            return True
        return super().eventFilter(source, event)

    def toggle_translation(self):
        if recording_flag.is_set():
            recording_flag.clear()
            self.translate_button.setText("翻訳開始")
            print("[DEBUG] 停止しました")
        else:
            self.selected_langs = [lang_map[box.currentText()] for box in self.lang_boxes]
            self.lang_target_map = {lang_map[self.lang_boxes[i].currentText()]: lang_map[self.target_boxes[i].currentText()]
                                    for i in range(len(self.lang_boxes))}
            recording_flag.set()
            self.translate_button.setText("停止")
            print("[DEBUG] 開始 - 言語: ", self.selected_langs)
            print("[DEBUG] 翻訳マップ: ", self.lang_target_map)
            threading.Thread(target=self.stream_audio, daemon=True).start()
            threading.Thread(target=self.recognize_and_translate, daemon=True).start()

    def stream_audio(self):
        def callback(indata, frames, time_info, status):
            if status:
                print("[DEBUG] Status:", status)
            mono_audio = indata[:, 0] if indata.ndim > 1 else indata
            audio_queue.put(mono_audio.copy())

        try:
            device_index = self.device_box.currentIndex()
            device_name = self.device_box.currentText()
            print(f"[DEBUG] 使用デバイス: {device_name} (index={device_index})")

            with sd.InputStream(device=device_index, channels=1, samplerate=16000,
                                callback=callback, dtype='float32'):
                while recording_flag.is_set():
                    try:
                        time.sleep(0.1)
                    except Exception as e:
                        print("[ERROR] stream_audio 内部ループでエラー:", e)
                        traceback.print_exc()
        except Exception as e:
            print("[ERROR] ストリーム開始に失敗:", e)
            traceback.print_exc()



    def recognize_and_translate(self):
        while recording_flag.is_set():
            try:
                frames = []
                buffer = []
                start = time.time()
                last_audio_time = start
                speech_detected = False
                while time.time() - start < 10:
                    try:
                        audio_frame = audio_queue.get(timeout=0.1)
                        frames.append(audio_frame)
                        buffer.append(audio_frame)
                    except queue.Empty:
                        continue
                        
                    audio = np.concatenate(buffer)
                    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
                    
                    speech_timestamps = get_speech_timestamps(audio_tensor, vad_model, sampling_rate=16000)
                    
                    if speech_timestamps:
                        speech_detected = True
                        print("音声検知")
                        last_audio_time = time.time()
                        buffer = []
                    else:
                        # 無音の状態が続いている（ただし喋り始めていた場合は無視）
                        if speech_detected and time.time() - last_audio_time > 1.5:
                            print("[DEBUG] 1秒間無音が続いたので録音を終了します。")
                            break
                        elif not speech_detected and time.time() - last_audio_time > 1:
                            print(time.time() - last_audio_time)
                            last_audio_time = time.time()
                            continue

                if not frames:
                    continue

                audio = np.concatenate(frames)
                audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
                
                speech_timestamps = get_speech_timestamps(audio_tensor, vad_model, sampling_rate=16000)
                    
                if not speech_timestamps:
                    print("[DEBUG] 無音または非音声：スキップ")
                    continue

                audio_processed = pad_or_trim(audio)
                
                results = {}
                if len(self.selected_langs) == 1:
                    # selected_langsが1つの場合は直接langを指定
                    lang = self.selected_langs[0]
                    try:
                        segments, info = model.transcribe(audio_processed, language=lang, task="transcribe")
                        # print(f"info: {info}")
                        language_probability = info.language_probability if info.language_probability is not None else -float("inf")
                        text = "".join([seg.text for seg in segments])
                        results[lang] = (language_probability, text.strip())
                        print(f"[DEBUG] 言語 {lang}: 信頼度 {language_probability:.4f}")
                    except Exception as e:
                        print(f"[ERROR] {lang} の認識に失敗: {e}")

                else:
                    # selected_langsが複数の場合は一度のみ音声をtranscribeし、検出された言語を確認
                    try:
                        segments, info = model.transcribe(audio_processed, language=None, task="transcribe")

                        all_language_probs = info.all_language_probs
                        text = "".join([seg.text for seg in segments])

                        # selected_langsに含まれる言語のみ結果に追加
                        for lang, prob in all_language_probs:
                            if lang in self.selected_langs:
                                results[lang] = (prob, text.strip())
                                print(f"[DEBUG] 言語 {lang}: 信頼度 {prob:.4f}")

                    except Exception as e:
                        print(f"[ERROR] 複数言語認識に失敗: {e}")

                if not results:
                    print("[WARN] 認識失敗：翻訳をスキップ")
                    continue

                # 最も信頼度の高い言語を選択
                best_lang = max(results, key=lambda k: results[k][0])
                text = results[best_lang][1]
                print(f"[DEBUG] 最終言語: {best_lang}")
                print(f"[DEBUG] 認識結果({best_lang}): {text}")

                if best_lang not in self.lang_target_map:
                    print(f"[WARN] {best_lang} は翻訳対象外")
                    continue
                
                if self.lang_target_map[best_lang] == "en":
                    target_lang = "EN-US"
                else:
                    target_lang = self.lang_target_map[best_lang]
                        
                try:
                    translated = deepL_translator.translate_text(text, source_lang=best_lang,
                                                                target_lang=target_lang)
                    print(f"[DEBUG] 翻訳結果: {translated.text}")

                    self.append_text_signal.emit(f"{best_lang.upper()}: {text}")
                    self.append_text_signal.emit(f"{target_lang.upper()}: {translated.text}\n")
                    
                except Exception as e:
                    print(f"[ERROR] 翻訳失敗: {e}")
                    traceback.print_exc()

            except Exception as e:
                print("[ERROR] 認識・翻訳エラー:", e)
                traceback.print_exc()
        



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TranslatorApp()
    window.show()
    sys.exit(app.exec_())