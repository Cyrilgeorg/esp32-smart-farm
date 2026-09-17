import os
import uuid
import speech_recognition as sr
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from huggingface_hub import InferenceClient
from gtts import gTTS

app = FastAPI(title="Smart Plant API")

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
    provider="auto",
    api_key=HF_TOKEN,
)

# إضافة دعم طلبات GET و HEAD معاً لمعالجة Render Health Check
@app.api_route("/", methods=["GET", "HEAD"])
def read_root():
    return {"status": "running", "message": "Smart Plant API is ready!"}

@app.post("/chat")
async def plant_chat(file: UploadFile = File(...)):
    request_id = uuid.uuid4().hex
    input_path = f"temp_{request_id}.wav"
    
    with open(input_path, "wb") as f:
        f.write(await file.read())

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(input_path) as source:
            audio_data = recognizer.record(source)
            user_text = recognizer.recognize_google(audio_data, language="ar-EG")
    except Exception as e:
        print(f"STT Error: {e}")
        user_text = "أهلاً"
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

    messages = [
        {"role": "system", "content": "أنت نبتة نعناع ذكية في أصيص. أجب بأسلوب لطيف ومختصر جداً (جملة واحدة فقط) باللغة العربية."},
        {"role": "user", "content": user_text}
    ]

    try:
        response = client.chat.completions.create(messages=messages, max_tokens=40)
        ai_reply = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        ai_reply = "أهلاً بك! أنا نبتة سعيدة اليوم."

    output_audio_path = f"output_{request_id}.mp3"
    tts = gTTS(text=ai_reply, lang='ar')
    tts.save(output_audio_path)

    return FileResponse(path=output_audio_path, media_type="audio/mpeg", filename="response.mp3")
