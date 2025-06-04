# app.py

# --- Necessary Libraries (These will be installed via requirements.txt) ---
# fastapi, uvicorn, chatterbox, accelerate, transformers, torch, torchaudio,
# soundfile, pydub, psutil, pynvml, numpy

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import time
import os
import shutil # Used for copying reference audio if needed
from pydub import AudioSegment
from pydub.effects import normalize
import io # Used to send binary audio data directly from memory
import numpy as np # Used by create_dummy_audio

# --- Memory Monitoring (Optional, will show "N/A (CPU mode)" on local machine) ---
import psutil
try:
    from pynvml import *
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Chatterbox TTS API",
    description="Text-to-Speech service using ChatterboxTTS with custom voice via GPU (or CPU for local test).",
    version="1.0.0"
)

# --- Global Variables for Model and Configuration (Loaded once on app startup) ---
# IMPORTANT: Model loading is done once globally, NOT per request, for performance.
model = None
SR = None
# Assumes kplor_voice.wav is in the same directory as app.py
REFERENCE_AUDIO_PATH = "kplor_voice.wav"
DUMMY_AUDIO_PATH = "dummy_voice_prompt.wav" # Fallback if kplor_voice.wav is missing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Will be 'cpu' on your local machine

# --- Helper Functions ---
def get_memory_usage():
    """Returns current CPU and GPU memory usage."""
    pid = os.getpid()
    process = psutil.Process(pid)
    cpu_memory_mb = process.memory_info().rss / (1024 * 1024)

    gpu_memory_mb = "N/A (CPU mode)"
    if NVML_AVAILABLE and DEVICE == 'cuda': # This block won't activate on your local CPU machine
        try:
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(0)
            info = nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_mb = info.used / (1024 * 1024)
            nvmlShutdown()
        except NVMLError as error:
            gpu_memory_mb = f"NVML Error: {error}"
        except Exception as e:
            gpu_memory_mb = f"GPU Mem Error: {e}"
    return {"cpu_mb": cpu_memory_mb, "gpu_mb": gpu_memory_mb}

def create_dummy_audio(path):
    """Creates a simple dummy WAV file for fallback."""
    sample_rate_dummy = 24000
    duration_dummy = 5
    frequency_dummy = 500
    t_dummy = np.linspace(0, duration_dummy, int(sample_rate_dummy * duration_dummy), endpoint=False)
    amplitude_dummy = 0.6
    dummy_audio_tensor = torch.tensor(amplitude_dummy * np.sin(2 * np.pi * frequency_dummy * t_dummy),
                                      dtype=torch.float32).unsqueeze(0)
    ta.save(path, dummy_audio_tensor, sample_rate_dummy)


# --- Model Initialization (Called once when FastAPI app starts up) ---
@app.on_event("startup")
async def startup_event():
    """Initializes TTS model and sets up custom voice on app startup."""
    global model, SR, REFERENCE_AUDIO_PATH, DEVICE

    print(f"[INIT] Using device: {DEVICE}") # Will print 'cpu' on your local machine
    print("[INIT] Loading Chatterbox TTS model...")
    try:
        # Modified line:
        # Modified line:
        model = ChatterboxTTS.from_pretrained(device=DEVICE)
        SR = model.sr
        print(f"[INIT] Chatterbox TTS model loaded successfully on {DEVICE}.")
    except Exception as e:
        print(f"[INIT] ERROR: Could not load Chatterbox model: {e}")
        # In FastAPI, this will cause the server to fail to start, which is good
        raise RuntimeError("Failed to load TTS model. Check PyTorch/CUDA setup (if using GPU).") from e

    print("[INIT] Setting up custom voice...")
    if not os.path.exists(REFERENCE_AUDIO_PATH):
        print(f"[INIT] Warning: Custom voice '{REFERENCE_AUDIO_PATH}' not found. Creating dummy.")
        create_dummy_audio(DUMMY_AUDIO_PATH)
        REFERENCE_AUDIO_PATH = DUMMY_AUDIO_PATH
    else:
        print(f"[INIT] Using custom voice from: {REFERENCE_AUDIO_PATH}")

    mem = get_memory_usage()
    print(f"[INIT] Initial Memory Usage (after model load): CPU: {mem['cpu_mb']:.2f} MB, GPU: {mem['gpu_mb']:.2f} MB")
    print("-" * 60)


# --- TTS Generation Logic (per request) ---
def generate_audio_for_text(
    text_to_speak: str,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    target_loudness_dbfs: float = -3.0
) -> io.BytesIO:
    """
    Generates audio for a single text segment, processes it, and returns
    the audio data as a BytesIO object (in-memory WAV).
    """
    if model is None or SR is None:
        raise RuntimeError("TTS model not initialized. Server setup error (model is None or SR is None).")
    if not os.path.exists(REFERENCE_AUDIO_PATH):
        raise FileNotFoundError(f"Reference audio not found at {REFERENCE_AUDIO_PATH}. Server setup error (voice file missing).")

    start_gen_time = time.time()
    try:
        # Generate speech using Chatterbox
        wav = model.generate(
            text=text_to_speak,
            audio_prompt_path=REFERENCE_AUDIO_PATH,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight
        )

        # Convert torch tensor to numpy array, then to pydub AudioSegment
        audio_np = wav.cpu().squeeze().numpy()
        audio_segment = AudioSegment(
            audio_np.tobytes(),
            frame_rate=SR,
            sample_width=audio_np.dtype.itemsize,
            channels=1
        )

        # Normalize loudness
        processed_audio = audio_segment.normalize(headroom=target_loudness_dbfs)

        # Export to BytesIO object (in-memory WAV)
        audio_buffer = io.BytesIO()
        processed_audio.export(audio_buffer, format="wav")
        audio_buffer.seek(0) # Rewind to the beginning of the buffer for reading

        generation_duration = time.time() - start_gen_time
        audio_duration = processed_audio.duration_seconds

        print(f"[PROCESS] Generated {audio_duration:.2f}s audio in {generation_duration:.2f}s.")
        return audio_buffer

    except Exception as e:
        print(f"[PROCESS] ERROR during audio generation: {e}")
        raise


# --- Pydantic Model for Request Body (for automatic validation and docs) ---
class TTSRequest(BaseModel):
    text: str
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    target_loudness_dbfs: float = -3.0

# --- FastAPI API Endpoint ---
@app.post("/generate-voice",
          summary="Generate speech from text",
          description="Generates an audio clip (WAV format) from provided text using ChatterboxTTS. Supports custom voice prompt and emotional parameters.")
async def generate_voice_endpoint(request: TTSRequest):
    """
    Receives a POST request with text and optional parameters,
    generates speech, and returns an audio/wav response.
    """
    request_start_time = time.time()
    print(f"\n[API] Received POST request.")

    text_to_speak = request.text
    exaggeration = request.exaggeration
    cfg_weight = request.cfg_weight
    target_loudness_dbfs = request.target_loudness_dbfs

    print(f"[API] Processing text: '{text_to_speak[:80]}...' (Exagg: {exaggeration}, CFG: {cfg_weight}, Loudness: {target_loudness_dbfs} dBFS)")

    try:
        audio_buffer = generate_audio_for_text(
            text_to_speak=text_to_speak,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            target_loudness_dbfs=target_loudness_dbfs
        )
        total_request_time = time.time() - request_start_time
        current_mem = get_memory_usage()
        print(f"[API] Request processed in {total_request_time:.2f}s. Memory: CPU: {current_mem['cpu_mb']:.2f} MB, GPU: {current_mem['gpu_mb']:.2f} MB")

        # FastAPI way to return binary data
        return Response(content=audio_buffer.getvalue(), media_type="audio/wav")

    except Exception as e:
        print(f"[API] Internal server error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate audio: {str(e)}")


# --- Health Check Endpoint (Optional but Recommended) ---
@app.get("/health", summary="Health Check", description="Checks if the API is running and the model is loaded.")
async def health_check():
    if model is not None and SR is not None and os.path.exists(REFERENCE_AUDIO_PATH):
        return {"status": "healthy", "message": "TTS model and voice are loaded and ready."}
    else:
        raise HTTPException(status_code=503, detail="Service Unavailable: TTS model or voice not initialized.")

# To run this FastAPI app, you would use Uvicorn (see instructions below).
# You won't use app.run() as you would with Flask.