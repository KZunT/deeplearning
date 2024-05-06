# instantiate pretrained model
from pyannote.audio import Model

model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")

from pyannote.audio import Inference

inference = Inference(model, window="whole")
embedding1 = inference("speaker1.wav")
embedding2 = inference("speaker2.wav")
# `embeddingX` is (1 x D) numpy array extracted from the file as a whole.

from scipy.spatial.distance import cdist

distance = cdist(embedding1, embedding2, metric="cosine")[0, 0]
# `distance` is a `float` describing how dissimilar speakers 1 and 2 are.
