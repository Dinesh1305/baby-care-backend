import os
import tempfile
import librosa
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Baby Monitor AI Backend")

# Configure CORS for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (React frontend, IoT devices)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. LOAD THE AI MODEL ON STARTUP
# ==========================================
print("Loading TFLite model...")
try:
    # Make sure 'cry_detection_model.tflite' is in the same folder as main.py
    interpreter = tf.lite.Interpreter(model_path="cry_detection_model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ Model loaded successfully!")

except Exception as e:
    print(f"❌ Failed to load model. Is 'cry_detection_model.tflite' in the right folder? Error: {e}")


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
    temp_audio_path = ""
    try:
        # 1. Save temporarily to disk to prevent librosa file-reading errors
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(await file.read())
            temp_audio_path = temp_audio.name

        # 2. Load audio at 22,050 Hz (Standard ML training sample rate)
        y, sr = librosa.load(temp_audio_path, sr=22050)

        # 3. Generate the Mel Spectrogram
        melspec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=40,         # 40 frequency bins
            n_fft=2048,
            hop_length=512     # Creates the 216 time frames over 5 seconds
        )

        # 4. Convert power to decibels (log scale)
        log_melspec = librosa.power_to_db(melspec, ref=np.max)

        # 5. Normalize between 0 and 1 (Standard scaling for Neural Networks)
        log_melspec = (log_melspec - np.min(log_melspec)) / (np.max(log_melspec) - np.min(log_melspec) + 1e-10)

        # 6. Ensure exactly 216 time frames
        if log_melspec.shape[1] < 216:
            pad_width = 216 - log_melspec.shape[1]
            log_melspec = np.pad(log_melspec, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            log_melspec = log_melspec[:, :216]

        # 7. Reshape to the exact tensor shape: [1, 40, 216, 1]
        real_input = log_melspec.reshape(1, 40, 216, 1).astype(np.float32)

        # 8. Run the TFLite Model
        interpreter.set_tensor(input_details[0]['index'], real_input)
        interpreter.invoke()

        # 9. Get the result
        prediction = interpreter.get_tensor(output_details[0]['index'])
        cry_probability = float(prediction[0][0])
        print(cry_probability)
        return {
            "success": True,
            "filename": file.filename,
            "ai_analysis": {
                "cry_probability": cry_probability,
                "is_crying": cry_probability > 0.85
            }
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 10. Clean up the temporary file so your hard drive doesn't fill up!
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

if __name__ == "__main__":
    # Run the server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)