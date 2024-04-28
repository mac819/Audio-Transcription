import os
from dotenv import load_dotenv
# load_dotenv()


from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration
)

import librosa


# hf_token = os.environ['HF_TOKEN']


class Transcription:
    def __init__(self, model_ckpt, sr):
        self.processor = WhisperProcessor.from_pretrained(model_ckpt, language="en", task="transcribe")
        self.model = WhisperForConditionalGeneration.from_pretrained(model_ckpt)
        self.model.config.forced_decoder_ids = None

        self.sampling_rate = sr

    def transcribe(self, signal):    
        input_feature = self.processor(signal, sampling_rate=self.sampling_rate, return_tensors="pt").input_features
        # generate token ids
        predicted_ids = self.model.generate(input_feature)
        # decode token ids to text
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription

