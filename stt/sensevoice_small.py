from funasr import AutoModel

model_dir = "iic/SenseVoiceSmall"
input_file = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"

model = AutoModel(
    model=model_dir,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    trust_remote_code=True,
    device="cuda:0",
)

res = model.generate(
    input=input_file,
    cache={},
    language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
    use_itn=False,
    batch_size_s=0,
)

print(res)
