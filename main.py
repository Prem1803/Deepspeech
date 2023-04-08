import deepspeech
import numpy as np
import wave

# create a DeepSpeech model with the downloaded pre-trained models
model = deepspeech.Model('./deepseech-model/deepspeech-0.9.3-models.pbmm')
model.enableExternalScorer('./deepseech-model/deepspeech-0.9.3-models.scorer')

# load an audio file to transcribe
with wave.open('audio/2830-3980-0043.wav', 'rb') as w:
    rate = w.getframerate()
    frames = w.getnframes()
    buffer = w.readframes(frames)

# transcribe the audio file
text = model.stt(np.frombuffer(buffer,dtype=np.int16))
print(f"\nTranscription : {text}")
