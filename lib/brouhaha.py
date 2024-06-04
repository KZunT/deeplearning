# 1. visit hf.co/pyannote/brouhaha and accept user conditions
# 2. visit hf.co/settings/tokens to create an access token
# 3. instantiate pretrained model
from pyannote.audio import Model

model = Model.from_pretrained(
    "pyannote/brouhaha", use_auth_token="ACCESS_TOKEN_GOES_HERE"
)

# apply model
from pyannote.audio import Inference

inference = Inference(model)
output = inference("audio.wav")

# iterate over each frame
for frame, (vad, snr, c50) in output:
    t = frame.middle
    print(f"{t:8.3f} vad={100*vad:.0f}% snr={snr:.0f} c50={c50:.0f}")

#  ...
# 12.952 vad=100% snr=51 c50=17
# 12.968 vad=100% snr=52 c50=17
# 12.985 vad=100% snr=53 c50=17
# ...
