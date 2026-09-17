import os
import uuid
import speech_recognition as sr
from fastapi import FastAPI, UploadFile, File, Form
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

@app.api_route("/", methods=["GET", "HEAD"])
def read_root():
    return {"status": "running", "message": "Smart Plant API is ready!"}

@app.post("/chat")
async def plant_chat(
    file: UploadFile = File(...),
    temp: str = Form(default="غير معروفة"),
    humidity: str = Form(default="غير معروفة")
):
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

    # تضمين درجة الحرارة والرطوبة المباشرة داخل تعليمات النظام
    system_instruction = (
        f"أنت نبتة نعناع ذكية تعيش في أصيص. "
        f"درجة الحرارة الحالية حولك هي {temp} درجة مئوية، ونسبة الرطوبة هي {humidity}%. "
        f"أجب على سؤال المستخدم بأسلوب لطيف ومختصر جداً (جملة واحدة فقط) باللغة العربية. "
        f"إذا سألك عن الحرارة أو كيف تشعر، استخدم البيانات الحقيقية المتاحة لديك."
    )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_text}
    ]

    try:
        response = client.chat.completions.create(messages=messages, max_tokens=40)
        ai_reply = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        ai_reply = f"حرارتي الحالية هي {temp} درجة مئوية."

    output_audio_path = f"output_{request_id}.mp3"
    tts = gTTS(text=ai_reply, lang='ar')
    tts.save(output_audio_path)

    return FileResponse(path=output_audio_path, media_type="audio/mpeg", filename="response.mp3")
