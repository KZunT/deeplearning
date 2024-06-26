# pip install --upgrade torch torchaudio librosa vocos encodec huggingface_hub

from inference import Mars5TTS, InferenceConfig as config_class
import librosa

mars5 = Mars5TTS.from_pretrained("CAMB-AI/MARS5-TTS")
# The `mars5` contains the AR and NAR model, as well as inference code.
# The `config_class` contains tunable inference config settings like temperature.

# load reference audio between 1-12 seconds.
wav, sr = librosa.load("<path to arbitrary 24kHz waveform>.wav", sr=mars5.sr, mono=True)
wav = torch.from_numpy(wav)
ref_transcript = "<transcript of the reference audio>"

# Pick whether you want a deep or shallow clone. Set to False if you don't know prompt transcript or want fast inference. Set to True if you know transcript and want highest quality.
deep_clone = True
# Below you can tune other inference settings, like top_k, temperature, top_p, etc...
cfg = config_class(
    deep_clone=deep_clone,
    rep_penalty_window=100,
    top_k=100,
    temperature=0.7,
    freq_penalty=3,
)

ar_codes, output_audio = mars5.tts("The quick brown rat.", wav, ref_transcript, cfg=cfg)
# output_audio is (T,) shape float tensor corresponding to the 24kHz output audio.
