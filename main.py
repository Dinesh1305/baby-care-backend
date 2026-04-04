import os
import tempfile
import traceback
import librosa
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import pydub for the audio conversion workaround
from pydub import AudioSegment

app = FastAPI(title="Baby Monitor AI Backend")

# Configure CORS for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. LOAD THE AI MODEL ON STARTUP
# ==========================================
print("Loading TFLite model...")
try:
    interpreter = tf.lite.Interpreter(model_path="baby_sound_classifier_v2.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model. Error: {e}")


# ==========================================
# 2. DEFINE DATA STRUCTURES
# ==========================================
class SoundData(BaseModel):
    intensity: int
    status: str
    duration: int = 0


@app.get("/")
def health_check():
    return {"status": "API is live and ready", "model_loaded": "interpreter" in globals()}


# ==========================================
# 3. ENDPOINT: IoT SENSORS (ESP32/Arduino)
# ==========================================
@app.post("/sound-detected")
async def process_sound_data(data: SoundData):
    try:
        is_crying = data.intensity > 70
        print(f"Received IoT Data - Intensity: {data.intensity}, Status: {data.status}")
        return {
            "success": True,
            "received_intensity": data.intensity,
            "action": "Triggering React alerts" if is_crying else "Normal"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 4. ENDPOINT: RAW AUDIO FOR AI PREDICTION
# ==========================================
@app.post("/predict-media")
async def predict_media(file: UploadFile = File(...)):
    temp_webm_path = ""
    temp_wav_path = ""
    try:
        # 1. Grab the actual extension from the file (e.g., '.webm' or '.wav')
        ext = os.path.splitext(file.filename)[1]
        if not ext:
            ext = ".webm"  # Default to webm if missing

        # 2. Save the incoming file (.webm from React)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_webm:
            temp_webm.write(await file.read())
            temp_webm_path = temp_webm.name

        # 3. Convert .webm to .wav using pydub
        audio = AudioSegment.from_file(temp_webm_path)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            temp_wav_path = temp_wav.name

        audio.export(temp_wav_path, format="wav")

        # 4. Load the pure .wav file into librosa
        y, sr = librosa.load(temp_wav_path, sr=22050)

        # 5. Generate the Mel Spectrogram
        melspec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=40, n_fft=2048, hop_length=512
        )

        # 6. Convert to DB and Normalize
        log_melspec = librosa.power_to_db(melspec, ref=np.max)
        log_melspec = (log_melspec - np.min(log_melspec)) / (np.max(log_melspec) - np.min(log_melspec) + 1e-10)

        # 7. Pad or truncate to exactly 216 frames
        if log_melspec.shape[1] < 216:
            pad_width = 216 - log_melspec.shape[1]
            log_melspec = np.pad(log_melspec, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            log_melspec = log_melspec[:, :216]

        # 8. Reshape and predict
        real_input = log_melspec.reshape(1, 40, 216, 1).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], real_input)
        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details[0]['index'])
        cry_probability = float(prediction[0][0])
        print(f"Analyzed {file.filename} -> Cry Probability: {cry_probability:.2f}")

        return {
            "success": True,
            "filename": file.filename,
            "ai_analysis": {
                "cry_probability": cry_probability,
                "is_crying": cry_probability > 0.85
            }
        }
    except Exception as e:
        print("\n=== ERROR DETAILS ===")
        traceback.print_exc()
        print("=====================\n")
        raise HTTPException(status_code=500, detail="Internal Server Error. Check server console.")
    finally:
        # Clean up both temp files to prevent filling up the hard drive
        if temp_webm_path and os.path.exists(temp_webm_path):
            try:
                os.remove(temp_webm_path)
            except:
                pass
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except:
                pass


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)