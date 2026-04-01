import os
import logging
import queue
import threading
import time
import numpy as np
import torch
import io
from speech_recognition import Recognizer, AudioData
from openai import OpenAI
from pocket_tts import TTSModel
import scipy.io.wavfile

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class AudioBuffer:
    """
    Buffers incoming audio chunks and detects silence to determine
    when a user has finished a sentence.
    """
    def __init__(self, sample_rate=16000, silence_duration=1.0, threshold=300):
        self.sample_rate = sample_rate
        # Approximate number of chunks to wait before declaring silence
        self.silence_limit = int(silence_duration * (sample_rate / 1024)) 
        self.threshold = threshold # Energy threshold for silence
        self.buffer = []
        self.silence_counter = 0
        self.lock = threading.Lock()
        self.has_started_talking = False
        
    def add_chunk(self, data):
        with self.lock:
            self.buffer.append(data)
            
            # Simple VAD: Calculate energy
            # NOTE: This assumes raw PCM audio. 
            # If you are sending WebM/Opus from browser, this VAD might not work 
            # until you decode the audio to PCM.
            try:
                audio_np = np.frombuffer(data, dtype=np.int16)
                energy = np.linalg.norm(audio_np)
                
                if energy > self.threshold:
                    self.has_started_talking = True
                    self.silence_counter = 0
                elif self.has_started_talking:
                    self.silence_counter += 1
            except Exception:
                pass # Data might not be raw PCM yet

    def is_complete_utterance(self):
        with self.lock:
            return (self.has_started_talking and 
                    self.silence_counter > self.silence_limit and 
                    len(self.buffer) > 0)
            
    def get_and_clear(self):
        with self.lock:
            if not self.buffer:
                return None
            full_audio = b"".join(self.buffer)
            self.buffer = []
            self.silence_counter = 0
            self.has_started_talking = False
            return full_audio

class ConversationalAgent:
    def __init__(self, response_queue):
        self.response_queue = response_queue
        
        # 1. OpenAI Client (from main_gpt.py)
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "HIDDEN_TOKEN"))
        
        self.context = """You are a helpful AI assistant. Keep answers concise and natural for speech.
        Avoid using markdown, symbols, or formatting that cannot be read aloud easily."""
        self.messages = [{"role": "system", "content": self.context}]

        # 2. TTS Model (from main.py - PocketTTS)
        logger.info("Loading TTS model...")
        self.tts_model = TTSModel.load_model()
        self.voice_state = self.tts_model.get_state_for_audio_prompt("alba")
        logger.info("TTS model loaded.")

        # 3. STT (SpeechRecognition)
        self.recognizer = Recognizer()
        
        # Buffer for real-time processing
        self.audio_buffer = AudioBuffer()

    def process_audio_chunk(self, chunk_bytes):
        """Called by server when new audio data arrives."""
        self.audio_buffer.add_chunk(chunk_bytes)

        if self.audio_buffer.is_complete_utterance():
            raw_audio = self.audio_buffer.get_and_clear()
            if raw_audio:
                # Offload processing to a thread so it doesn't block the gRPC stream
                threading.Thread(
                    target=self.handle_conversation_turn, 
                    args=(raw_audio,), 
                    daemon=True
                ).start()

    def handle_conversation_turn(self, raw_audio):
        try:
            logger.info("Processing speech turn...")
            
            # 1. STT
            # NOTE: If 'raw_audio' is WebM/Opus from the browser, recognize_google might fail
            # as it expects WAV/FLAC. You may need to convert WebM -> WAV here if you haven't already.
            try:
                # Assuming 16000 sample rate and 2 bytes width for PCM
                audio_data = AudioData(raw_audio, 16000, 2) 
                text = self.recognizer.recognize_google(audio_data, language="en-US")
                logger.info(f"User said: {text}")
            except Exception as e:
                logger.warning(f"STT Failed: {e}")
                return

            if not text or text.lower() in ["quit", "exit", "stop"]:
                return

            # 2. LLM (OpenAI)
            self.messages.append({"role": "user", "content": text})
            response = self.client.chat.completions.create(
                model="gpt-4o-mini", 
                messages=self.messages
            )
            reply = response.choices[0].message.content
            logger.info(f"AI Reply: {reply}")
            self.messages.append({"role": "assistant", "content": reply})

            # 3. TTS (PocketTTS)
            logger.info("Generating audio...")
            audio_tensor = self.tts_model.generate_audio(self.voice_state, reply)
            
            # Convert to WAV bytes in memory
            buffer = io.BytesIO()
            audio_np = audio_tensor.cpu().numpy() if torch.is_tensor(audio_tensor) else audio_tensor
            scipy.io.wavfile.write(buffer, self.tts_model.sample_rate, audio_np)
            wav_bytes = buffer.getvalue()
            
            # 4. Queue chunks for sending back
            chunk_size = 4096 
            for i in range(0, len(wav_bytes), chunk_size):
                self.response_queue.put(("audio", wav_bytes[i:i+chunk_size]))
            
            # Signal end of turn
            self.response_queue.put(("end", None))

        except Exception as e:
            logger.error(f"Error in conversation turn: {e}", exc_info=True)
        
        