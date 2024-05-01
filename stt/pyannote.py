# instantiate the model
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection, OverlappedSpeechDetection

model = Model.from_pretrained(
    "pyannote/segmentation-3.0", use_auth_token="HUGGINGFACE_ACCESS_TOKEN_GOES_HERE"
)
pipeline = VoiceActivityDetection(segmentation=model)
HYPER_PARAMETERS = {
    # remove speech regions shorter than that many seconds.
    "min_duration_on": 0.0,
    # fill non-speech regions shorter than that many seconds.
    "min_duration_off": 0.0,
}
pipeline.instantiate(HYPER_PARAMETERS)
vad = pipeline("audio.wav")
# `vad` is a pyannote.core.Annotation instance containing speech regions

pipeline = OverlappedSpeechDetection(segmentation=model)
HYPER_PARAMETERS = {
    # remove overlapped speech regions shorter than that many seconds.
    "min_duration_on": 0.0,
    # fill non-overlapped speech regions shorter than that many seconds.
    "min_duration_off": 0.0,
}
pipeline.instantiate(HYPER_PARAMETERS)
osd = pipeline("audio.wav")
# `osd` is a pyannote.core.Annotation instance containing overlapped speech regions
