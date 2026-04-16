import numpy as np
import librosa
import pyaudio
import threading
import cv2
import os
import time
import uvicorn
import pygame
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- Global State ---
# This keeps the latest results accessible to the FastAPI endpoint
latest_detection = {
    "label": "Initializing...",
    "confidence": 0.0,
    "is_crying": False,
    "status": "Listening",
    "time_until_music": 0.0  # Added field for the countdown
}

# --- Import Logic for TFLite ---
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        tflite = None
        print("❌ Error: TFLite not found. Run: pip install tensorflow")

# --- Model Loading ---
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
        print(f"❌ Failed to load model: {e}")

# --- Audio Config ---
RATE = 22050
CHUNK = 1024
DURATION = 4  # Analyze 5-second sliding windows

# --- Music Config ---
pygame.mixer.init()
SONG_PATH = "song.mp3"
CRY_TIME_THRESHOLD = 4.0  # Seconds of continuous crying required to play music

# Pre-load song if it exists
if os.path.exists(SONG_PATH):
    pygame.mixer.music.load(SONG_PATH)
else:
    print(f"⚠️ Warning: {SONG_PATH} not found. Music will not play.")


def process_audio(audio_data):
    """Processes audio buffer and returns AI prediction"""
    try:
        if interpreter is None:
            return "Model Error", 0

        # 1. Normalize and fix length
        audio_data = librosa.util.fix_length(audio_data, size=RATE * DURATION)
        if np.max(np.abs(audio_data)) > 0:
            audio_data = librosa.util.normalize(audio_data)

        # 2. Silence check (RMS)
        if np.sqrt(np.mean(audio_data ** 2)) < 0.01:
            return "Silence", 0.0

        # 3. Create Spectrogram (128x128)
        spec = librosa.feature.melspectrogram(y=audio_data, sr=RATE, n_mels=128)
        log_spec = librosa.power_to_db(spec, ref=np.max)
        resized = cv2.resize(log_spec, (128, 128))

        # 4. Prepare for Model (Batch, Height, Width, Channels)
        inp = np.expand_dims(resized, axis=(0, -1)).astype(np.float32)

        # 5. Inference
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()
        conf = float(interpreter.get_tensor(output_details[0]['index'])[0][0])

        # Thresholding
        label = "Baby Crying" if conf > 0.65 else "Normal"
        confidence_score = conf * 100 if label == "Baby Crying" else (1 - conf) * 100
        return label, confidence_score

    except Exception as e:
        print(f"⚠️ Inference Error: {e}")
        return "Error", 0


def mic_loop():
    """Background thread: Keeps the microphone ON and processes sound"""
    global latest_detection
    p = pyaudio.PyAudio()

    # Track when the crying started
    crying_start_time = None

    try:
        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE,
                        input=True, frames_per_buffer=CHUNK)

        # Rolling buffer to hold 5 seconds of audio
        buffer = np.zeros(RATE * DURATION, dtype=np.float32)

        print("\n" + "=" * 40)
        print("🎙️  MIC ACTIVE: Monitoring Real-Time...")
        print("=" * 40 + "\n")

        while True:
            # Read small chunks to keep it responsive
            data = stream.read(CHUNK, exception_on_overflow=False)
            new_samples = np.frombuffer(data, dtype=np.float32)

            # Slide the buffer
            buffer = np.roll(buffer, -len(new_samples))
            buffer[-len(new_samples):] = new_samples

            # Process AI prediction
            label, conf = process_audio(buffer)
            is_crying = (label == "Baby Crying")

            time_until_music = 0.0

            # --- TIMER AND MUSIC LOGIC ---
            if is_crying:
                if crying_start_time is None:
                    # Start the timer
                    crying_start_time = time.time()
                    time_until_music = float(CRY_TIME_THRESHOLD)
                else:
                    # Calculate elapsed and remaining time
                    elapsed_crying_time = time.time() - crying_start_time
                    time_until_music = max(0.0, CRY_TIME_THRESHOLD - elapsed_crying_time)

                    if elapsed_crying_time > CRY_TIME_THRESHOLD:
                        # If crying > 10s and music isn't already playing, play music
                        if not pygame.mixer.music.get_busy() and os.path.exists(SONG_PATH):
                            print(f"\n🎶 Baby has been crying for {CRY_TIME_THRESHOLD}s. Playing lullaby...")
                            pygame.mixer.music.play()
            else:
                # If baby stops crying, reset the timer
                crying_start_time = None
                time_until_music = 0.0

            # Update global state (Frontend can access time_until_music here)
            latest_detection = {
                "label": label,
                "confidence": round(conf, 2),
                "is_crying": is_crying,
                "status": "Monitoring",
                "time_until_music": round(time_until_music, 1)
            }

            # Print Alert & Countdown to Terminal
            if is_crying:
                if pygame.mixer.music.get_busy():
                    print(f"🚨 ALERT: Baby Crying! ({conf:.1f}%) - 🎶 Lullaby is currently playing.")
                else:
                    print(f"🚨 ALERT: Baby Crying! ({conf:.1f}%) - ⏳ Lullaby in {time_until_music:.1f}s")

    except Exception as e:
        print(f"❌ Microphone Error: {e}")
    finally:
        p.terminate()


# --- FastAPI Backend ---
app = FastAPI(title="Baby Monitor AI - Real Time")

# Enable CORS for React/Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],

    allow_headers=["*"],
)


@app.get("/status")
async def get_status():
    """Endpoint for frontend to poll real-time status"""
    response_data = latest_detection.copy()
    response_data["music_playing"] = pygame.mixer.music.get_busy()
    return response_data


@app.get("/")
async def health():
    return {"status": "Backend is running", "mic_active": True}


if __name__ == "__main__":
    # Start Microphone thread as a Daemon (closes when main program stops)
    threading.Thread(target=mic_loop, daemon=True).start()

    # Start FastAPI server
    print(f"🚀 Server starting at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)