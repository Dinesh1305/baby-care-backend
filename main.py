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
    interpreter = tf.lite.Interpreter(model_path="baby_sound_classifier.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model. Error: {e}")

# Define classification categories mapping to the model's output neurons
CATEGORIES = ['baby_cry', 'baby_laugh', 'noise', 'silence']


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

        # 4. Load the pure .wav file into librosa (Match training: 22050Hz, 5 seconds)
        y, sr = librosa.load(temp_wav_path, sr=22050, duration=5)

        # 5. Pad or truncate raw audio to exactly 5 seconds
        target_length = 22050 * 5
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]

        # 6. Extract MFCC (instead of Mel Spectrogram)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        # 7. Shape fixing (Ensure width matches 216)
        expected_width = 216
        if mfcc.shape[1] < expected_width:
            mfcc = np.pad(mfcc, ((0, 0), (0, expected_width - mfcc.shape[1])))
        else:
            mfcc = mfcc[:, :expected_width]

        # 8. Reshape for TFLite model [1, 40, 216, 1]
        mfcc = mfcc.astype(np.float32)
        real_input = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=-1)

        # 9. Run Inference
        interpreter.set_tensor(input_details[0]['index'], real_input)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # 10. Process Multi-class Output
        predicted_index = int(np.argmax(output_data))
        top_result = CATEGORIES[predicted_index]
        top_confidence = float(output_data[0][predicted_index] * 100)

        # Keep backwards compatibility with React frontend tracking "cry probability" directly
        cry_probability = float(output_data[0][0])  # Index 0 is 'baby_cry'

        print(f"Analyzed {file.filename} -> {top_result} ({top_confidence:.2f}%)")

        return {
            "success": True,
            "filename": file.filename,
            "ai_analysis": {
                # Frontend relies on these two fields to plot graphs and trigger alerts
                "cry_probability": cry_probability,
                "is_crying": top_result == 'baby_cry',

                # Extra detailed information for potential future use
                "top_prediction": top_result,
                "confidence": top_confidence,
                "all_scores": {cat: float(output_data[0][i]) for i, cat in enumerate(CATEGORIES)}
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