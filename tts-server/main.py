from pocket_tts import TTSModel
import scipy.io.wavfile

tts_model = TTSModel.load_model()
voice_state = tts_model.get_state_for_audio_prompt(
    "alba"  
)
audio = tts_model.generate_audio(voice_state, "API usage is limited by concurrency (i.e., the number of in-flight requests). Below are the current rate limits for each model.")
# Audio is a 1D torch tensor containing PCM data.
scipy.io.wavfile.write("output.wav", tts_model.sample_rate, audio.numpy())
