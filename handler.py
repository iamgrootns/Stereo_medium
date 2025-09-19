import os
import torch
import numpy as np
import soundfile as sf
from io import BytesIO
import base64
import traceback
import runpod
from transformers import pipeline

# --- Path for Initialization Error Logging ---
INIT_ERROR_FILE = "/tmp/init_error.log"

# --- Global Model Loading ---
synth = None
try:
    # Clear any previous error logs
    if os.path.exists(INIT_ERROR_FILE):
        os.remove(INIT_ERROR_FILE)

    print("Loading facebook/musicgen-stereo-medium...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the stereo model using the transformers pipeline
    synth = pipeline(
        "text-to-audio",
        "facebook/musicgen-stereo-medium",
        device=device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    print("✅ Model loaded successfully.")

except Exception as e:
    # If loading fails, log the full error traceback
    tb_str = traceback.format_exc()
    with open(INIT_ERROR_FILE, "w") as f:
        f.write(f"Failed to initialize model: {tb_str}")
    synth = None # Ensure the model is None if loading failed


# --- Runpod Handler ---
def handler(event):
    """
    The handler function that will be called by Runpod for each job.
    """
    # --- Check for Initialization Error First ---
    if os.path.exists(INIT_ERROR_FILE):
        with open(INIT_ERROR_FILE, "r") as f:
            error_message = f.read()
        return {"error": f"Worker initialization failed: {error_message}"}

    if synth is None:
        return {"error": "Model is not loaded. Check initialization logs."}

    # --- Input Validation ---
    job_input = event.get("input", {})
    text = job_input.get("text")
    duration = job_input.get("duration", 30)  # Default to 30 seconds

    if not text:
        return {"error": "No 'text' prompt provided."}

    # --- Music Generation ---
    try:
        print(f"Generating {duration}s stereo audio for prompt: '{text}'")

        # Break generation into manageable chunks of 30 seconds
        chunk_size = 30
        num_chunks = (duration + chunk_size - 1) // chunk_size
        all_chunks = []
        final_sr = 32000  # Will be updated by the model

        for i in range(num_chunks):
            chunk_duration = min(chunk_size, duration - (i * chunk_size))
            max_new_tokens = int(chunk_duration * 50) # Approx. 50 tokens per second
            print(f"Generating chunk {i+1}/{num_chunks} ({chunk_duration}s)...")

            result = synth(
                text,
                forward_params={"max_new_tokens": max_new_tokens}
            )

            audio = result["audio"][0]
            final_sr = result["sampling_rate"]

            # Ensure the output is stereo
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=-1)
            elif audio.shape[0] < audio.shape[1]:
                audio = audio.T # Ensure shape is (samples, channels)

            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            all_chunks.append(audio)

        # Concatenate all generated audio chunks
        final_audio = np.concatenate(all_chunks, axis=0)

        # Save to an in-memory buffer
        buffer = BytesIO()
        sf.write(buffer, final_audio, final_sr, format="WAV", subtype="PCM_16")
        wav_bytes = buffer.getvalue()

        # Encode the audio data as a base64 string
        audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')

        print("✅ Generation complete.")
        return {
            "audio_base64": audio_base64,
            "sample_rate": final_sr,
            "format": "wav"
        }

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Generation failed: {error_trace}")
        return {"error": f"An error occurred during generation: {error_trace}"}


# --- Start Serverless Worker ---
if __name__ == "__main__":
    print("Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})
