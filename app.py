import time
import librosa
import threading
import streamlit as st
from src.audio_record import StreamParams, Recorder
from src.transcribe import Transcription

whisper_ckpt = "openai/whisper-small"
transcription = Transcription(model_ckpt=whisper_ckpt, sr=16000)

def record_and_save_audio(duration, save_path):
    stream_params = StreamParams(channels=1)
    recorder = Recorder(stream_params)
    recorder.record(duration, save_path)


def load_audio_data(data_path, sr=16000):
    signal, _ = librosa.load(data_path, sr=sr)
    return signal


def transcribe_audio(data_path):
    signal = load_audio_data(data_path=data_path, sr=16000)
    audio_text = transcription.transcribe(signal=signal)
    return audio_text


st.header("Audio Transcription Application")


with st.form("form_key"):
    duration = st.slider("Duration", min_value=5, max_value=10, value=5)
    save_path = st.text_area("Save Path", value="data/audio.wav")
    submit_btn = st.form_submit_button("Submit")


# st.write(f"""
#     Duration: {duration} | save_path: {save_path}
# """)
print(f"Duration: {duration}, Save Path: {save_path}")
x = threading.Thread(target=record_and_save_audio, \
                     kwargs={'duration': duration, \
                             'save_path': save_path})
x.start()

time.sleep(duration)
st.audio(save_path)

audio_text = transcribe_audio(data_path=save_path)

st.text(audio_text)




