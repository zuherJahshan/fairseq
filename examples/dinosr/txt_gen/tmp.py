import whisper

model_name = "medium.en"
model = whisper.load_model(model_name)
print(model.is_multilingual)
