import os
import uuid
import speech_recognition as sr
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
from gtts import gTTS

app = FastAPI(title="Smart Plant API")

# --- 1. تفعيل الـ CORS لمنع حظر الطلبات القادمة من ESP32 المتصلة بالشبكة المحلية ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # يسمح بالطلبات من جميع المصادر
    allow_credentials=True,
    allow_methods=["*"],        # يسمح بجميع أنواع الطلبات (GET, POST, OPTIONS...)
    allow_headers=["*"],        # يسمح بكل أنواع الـ Headers
)

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
    provider="auto",
    api_key=HF_TOKEN,
)

# دالة لحذف الصوت بعد إرساله للمستخدم لتوفير المساحة
def remove_file(path: str):
    if os.path.exists(path):
        os.remove(path)

@app.api_route("/", methods=["GET", "HEAD"])
def read_root():
    return {"status": "running", "message": "Smart Plant API is ready!"}

@app.post("/chat")
async def plant_chat(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    temp: str = Form(default="25"),
    humidity: str = Form(default="50")
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
        ai_reply = f"درجة الحرارة الحالية لدي هي {temp} مئوية."

    output_audio_path = f"output_{request_id}.mp3"
    tts = gTTS(text=ai_reply, lang='ar')
    tts.save(output_audio_path)

    # حجز مهمة خلفية لحذف الملف الصوتي الناتج بعد الإرسال فوراً
    background_tasks.add_task(remove_file, output_audio_path)

    return FileResponse(path=output_audio_path, media_type="audio/mpeg", filename="response.mp3")
