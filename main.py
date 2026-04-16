

#
# import numpy as np
# import librosa
# import pyaudio
# import threading
# import cv2
# import os
# import time
# import uvicorn
# import pygame
# import serial
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
#
# # --- Global State ---
# latest_detection = {
#     "label": "Initializing...",
#     "confidence": 0.0,
#     "is_crying": False,
#     "status": "Listening",
#     "time_until_music": 0.0,
#     "latest_sensor_msg": "No data yet",
#     "is_moving": True,  # <-- NEW
#     "time_since_move": 0.0  # <-- NEW
# }
#
# # --- Import Logic for TFLite ---
# try:
#     import tflite_runtime.interpreter as tflite
# except ImportError:
#     try:
#         import tensorflow.lite as tflite
#     except ImportError:
#         tflite = None
#         print("❌ Error: TFLite not found. Run: pip install tensorflow")
#
# # --- Model Loading ---
# MODEL_PATH = "baby_cry_v2_pro.tflite"
# interpreter = None
# input_details = None
# output_details = None
#
# if tflite and os.path.exists(MODEL_PATH):
#     try:
#         interpreter = tflite.Interpreter(model_path=MODEL_PATH)
#         interpreter.allocate_tensors()
#         input_details = interpreter.get_input_details()
#         output_details = interpreter.get_output_details()
#     except Exception as e:
#         print(f"❌ Failed to load model: {e}")
#
# # --- Audio Config ---
# RATE = 22050
# CHUNK = 1024
# DURATION = 5
#
# # --- Music Config ---
# pygame.mixer.init()
# SONG_PATH = "song.mp3"
# CRY_TIME_THRESHOLD = 5.0
#
# if os.path.exists(SONG_PATH):
#     pygame.mixer.music.load(SONG_PATH)
# else:
#     print(f"⚠️ Warning: {SONG_PATH} not found. Music will not play.")
#
# # --- Serial / Motion Config ---
# SERIAL_PORT = 'COM15'
# BAUD_RATE = 9600
# MOTION_THRESHOLD = 2500  # Sum of X,Y,Z delta to count as "Movement" (tune this if it's too sensitive!)
# SLEEP_TIMEOUT = 10.0  # Seconds of stillness required to be considered "Sleeping"
#
#
# def process_audio(audio_data):
#     try:
#         if interpreter is None:
#             return "Model Error", 0
#
#         audio_data = librosa.util.fix_length(audio_data, size=RATE * DURATION)
#         if np.max(np.abs(audio_data)) > 0:
#             audio_data = librosa.util.normalize(audio_data)
#
#         if np.sqrt(np.mean(audio_data ** 2)) < 0.01:
#             return "Silence", 0.0
#
#         spec = librosa.feature.melspectrogram(y=audio_data, sr=RATE, n_mels=128)
#         log_spec = librosa.power_to_db(spec, ref=np.max)
#         resized = cv2.resize(log_spec, (128, 128))
#
#         inp = np.expand_dims(resized, axis=(0, -1)).astype(np.float32)
#
#         interpreter.set_tensor(input_details[0]['index'], inp)
#         interpreter.invoke()
#         conf = float(interpreter.get_tensor(output_details[0]['index'])[0][0])
#
#         label = "Baby Crying" if conf > 0.65 else "Normal"
#         confidence_score = conf * 100 if label == "Baby Crying" else (1 - conf) * 100
#         return label, confidence_score
#
#     except Exception as e:
#         print(f"⚠️ Inference Error: {e}")
#         return "Error", 0
#
#
# def mic_loop():
#     global latest_detection
#     p = pyaudio.PyAudio()
#     crying_start_time = None
#
#     try:
#         stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
#         buffer = np.zeros(RATE * DURATION, dtype=np.float32)
#
#         while True:
#             data = stream.read(CHUNK, exception_on_overflow=False)
#             new_samples = np.frombuffer(data, dtype=np.float32)
#
#             buffer = np.roll(buffer, -len(new_samples))
#             buffer[-len(new_samples):] = new_samples
#
#             label, conf = process_audio(buffer)
#             is_crying = (label == "Baby Crying")
#             time_until_music = 0.0
#
#             if is_crying:
#                 if crying_start_time is None:
#                     crying_start_time = time.time()
#                     time_until_music = float(CRY_TIME_THRESHOLD)
#                 else:
#                     elapsed = time.time() - crying_start_time
#                     time_until_music = max(0.0, CRY_TIME_THRESHOLD - elapsed)
#                     if elapsed > CRY_TIME_THRESHOLD and not pygame.mixer.music.get_busy() and os.path.exists(SONG_PATH):
#                         pygame.mixer.music.play()
#             else:
#                 crying_start_time = None
#                 time_until_music = 0.0
#
#             latest_detection["label"] = label
#             latest_detection["confidence"] = round(conf, 2)
#             latest_detection["is_crying"] = is_crying
#             latest_detection["status"] = "Monitoring"
#             latest_detection["time_until_music"] = round(time_until_music, 1)
#
#     except Exception as e:
#         pass
#     finally:
#         p.terminate()
#
#
# def serial_loop():
#     global latest_detection
#
#     prev_ax, prev_ay, prev_az = 0, 0, 0
#     last_movement_time = time.time()
#
#     try:
#         ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
#         print("\n" + "=" * 40)
#         print(f"📡 SERIAL ACTIVE: Connected to ESP32 on {SERIAL_PORT}")
#         print("=" * 40 + "\n")
#
#         while True:
#             line = ser.readline().decode('utf-8', errors='ignore').strip()
#
#             if line:
#                 # 1. Parse the Accel line (e.g. "Accel X: 1234 | Y: 5678 | Z: 9101")
#                 if line.startswith("Accel X:"):
#                     try:
#                         parts = line.split("|")
#                         ax = int(parts[0].split(":")[1].strip())
#                         ay = int(parts[1].split(":")[1].strip())
#                         az = int(parts[2].split(":")[1].strip())
#
#                         # 2. Calculate fluctuation Delta
#                         delta = abs(ax - prev_ax) + abs(ay - prev_ay) + abs(az - prev_az)
#
#                         # Ignore the very first boot-up reading jump
#                         if prev_ax != 0 and delta > MOTION_THRESHOLD:
#                             last_movement_time = time.time()
#
#                         prev_ax, prev_ay, prev_az = ax, ay, az
#                     except Exception as e:
#                         pass  # Ignore parse errors from mangled serial lines
#
#                 latest_detection["latest_sensor_msg"] = line
#
#             # 3. Update global state for Frontend
#             time_since_move = time.time() - last_movement_time
#             is_moving = time_since_move < SLEEP_TIMEOUT
#
#             latest_detection["is_moving"] = is_moving
#             latest_detection["time_since_move"] = round(time_since_move, 1)
#
#     except serial.SerialException as e:
#         print(f"\n❌ Serial Port Error: {e}")
#     except Exception as e:
#         print(f"\n❌ Unknown Serial Error: {e}")
#
#
# app = FastAPI(title="Baby Monitor AI - Real Time")
# app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
#
#
# @app.get("/status")
# async def get_status():
#     response_data = latest_detection.copy()
#     response_data["music_playing"] = pygame.mixer.music.get_busy()
#     return response_data
#
#
# if __name__ == "__main__":
#     threading.Thread(target=mic_loop, daemon=True).start()
#     threading.Thread(target=serial_loop, daemon=True).start()
#     uvicorn.run(app, host="0.0.0.0", port=8000)



import numpy as np
import librosa
import pyaudio
import threading
import cv2
import os
import time
import uvicorn
import pygame
import serial
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- Global State ---
latest_detection = {
    "label": "Initializing...",
    "confidence": 0.0,
    "is_crying": False,
    "status": "Listening",
    "time_until_music": 0.0,
    "latest_sensor_msg": "No data yet",
    "is_moving": True,
    "time_since_move": 0.0,
    "temperature": 0.0
}

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        tflite = None
        print("❌ Error: TFLite not found.")

MODEL_PATH = "baby_cry_v2_pro.tflite"
interpreter = None
input_details = None
output_details = None

if tflite and os.path.exists(MODEL_PATH):
    try:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    except Exception as e:
        pass

RATE = 22050
CHUNK = 1024
DURATION = 4

pygame.mixer.init()
SONG_PATH = "baby_track.mp3"
CRY_TIME_THRESHOLD = 4.0

if os.path.exists(SONG_PATH):
    pygame.mixer.music.load(SONG_PATH)

SERIAL_PORT = 'COM15'
BAUD_RATE = 9600
MOTION_THRESHOLD = 2500
SLEEP_TIMEOUT = 10.0


def process_audio(audio_data):
    try:
        if interpreter is None:
            return "Model Error", 0
        audio_data = librosa.util.fix_length(audio_data, size=RATE * DURATION)
        if np.max(np.abs(audio_data)) > 0:
            audio_data = librosa.util.normalize(audio_data)
        if np.sqrt(np.mean(audio_data ** 2)) < 0.01:
            return "Silence", 0.0
        spec = librosa.feature.melspectrogram(y=audio_data, sr=RATE, n_mels=128)
        log_spec = librosa.power_to_db(spec, ref=np.max)
        resized = cv2.resize(log_spec, (128, 128))
        inp = np.expand_dims(resized, axis=(0, -1)).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()
        conf = float(interpreter.get_tensor(output_details[0]['index'])[0][0])
        label = "Baby Crying" if conf > 0.65 else "Normal"
        confidence_score = conf * 100 if label == "Baby Crying" else (1 - conf) * 100
        return label, confidence_score
    except Exception as e:
        return "Error", 0

def mic_loop():
    global latest_detection
    p = pyaudio.PyAudio()
    crying_start_time = None
    try:
        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
        buffer = np.zeros(RATE * DURATION, dtype=np.float32)
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            new_samples = np.frombuffer(data, dtype=np.float32)
            buffer = np.roll(buffer, -len(new_samples))
            buffer[-len(new_samples):] = new_samples
            label, conf = process_audio(buffer)
            is_crying = (label == "Baby Crying")
            time_until_music = 0.0

            if is_crying:
                if crying_start_time is None:
                    crying_start_time = time.time()
                    time_until_music = float(CRY_TIME_THRESHOLD)
                else:
                    elapsed = time.time() - crying_start_time
                    time_until_music = max(0.0, CRY_TIME_THRESHOLD - elapsed)
                    if elapsed > CRY_TIME_THRESHOLD and not pygame.mixer.music.get_busy() and os.path.exists(SONG_PATH):
                        pygame.mixer.music.play()
            else:
                crying_start_time = None
                time_until_music = 0.0

            latest_detection["label"] = label
            latest_detection["confidence"] = round(conf, 2)
            latest_detection["is_crying"] = is_crying
            latest_detection["status"] = "Monitoring"
            latest_detection["time_until_music"] = round(time_until_music, 1)
    except Exception as e:
        pass
    finally:
        p.terminate()

def serial_loop():
    global latest_detection
    prev_ax, prev_ay, prev_az = 0, 0, 0
    last_movement_time = time.time()
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"\n📡 SERIAL ACTIVE: {SERIAL_PORT}")
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                if line.startswith("Accel X:"):
                    try:
                        parts = line.split("|")
                        ax = int(parts[0].split(":")[1].strip())
                        ay = int(parts[1].split(":")[1].strip())
                        az = int(parts[2].split(":")[1].strip())
                        delta = abs(ax - prev_ax) + abs(ay - prev_ay) + abs(az - prev_az)
                        if prev_ax != 0 and delta > MOTION_THRESHOLD:
                            last_movement_time = time.time()
                        prev_ax, prev_ay, prev_az = ax, ay, az
                    except Exception: pass
                elif line.startswith("Temp:"):
                    try:
                        temp_str = line.split(":")[1].replace("C", "").strip()
                        latest_detection["temperature"] = float(temp_str)
                    except Exception: pass
                latest_detection["latest_sensor_msg"] = line

            time_since_move = time.time() - last_movement_time
            latest_detection["is_moving"] = time_since_move < SLEEP_TIMEOUT
            latest_detection["time_since_move"] = round(time_since_move, 1)
    except Exception as e:
        print(f"\n❌ Serial Error: {e}")

app = FastAPI(title="Baby Monitor AI - Real Time")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/status")
async def get_status():
    response_data = latest_detection.copy()
    # THIS is what the frontend will watch!
    response_data["music_playing"] = pygame.mixer.music.get_busy()
    return response_data

if __name__ == "__main__":
    threading.Thread(target=mic_loop, daemon=True).start()
    threading.Thread(target=serial_loop, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)