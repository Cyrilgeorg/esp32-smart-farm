import os
import uuid
import speech_recognition as sr
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
from gtts import gTTS
from pydub import AudioSegment

app = FastAPI(title="Smart Plant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    provider="nscale",
    api_key=HF_TOKEN,
)

def remove_file(path: str):
    if os.path.exists(path):
        os.remove(path)


def safe_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_plant_state(temp: float, humidity: float, light: float,
                     soil_moisture: float, salts: float) -> str:
    states = []

    if temp >= 35:
        states.append("حرارة عالية جداً، أشعر بالحر الشديد وأحتاج إلى ظل وتهوية")
    elif temp >= 30:
        states.append("الجو دافئ نوعاً ما")
    elif temp <= 15:
        states.append("أشعر بالبرد الشديد")
    elif temp <= 18:
        states.append("الجو بارد قليلاً بالنسبة لي")
    else:
        states.append("درجة الحرارة مريحة ومناسبة لي")

    if soil_moisture <= 20:
        states.append("تربتي جافة جداً وأنا عطشانة، أحتاج لسقاية فوراً")
    elif soil_moisture <= 40:
        states.append("بدأت أشعر بالعطش قليلاً")
    elif soil_moisture >= 85:
        states.append("التربة مروية بشكل زائد عن اللزوم وأشعر بعدم الراحة")
    else:
        states.append("مستوى الري لدي جيد حالياً")

    if light <= 20:
        states.append("المكان حولي مظلم وأحتاج لمزيد من الضوء لأقوم بعملية البناء الضوئي")
    elif light >= 90:
        states.append("كمية الضوء شديدة عليّ حالياً")
    else:
        states.append("كمية الضوء المتاحة لي مناسبة")

    if salts >= 80:
        states.append("أشعر أن نسبة الأملاح في التربة مرتفعة وهذا يزعجني")
    elif salts <= 10:
        states.append("أشعر أنني بحاجة لبعض المغذيات والأملاح المعدنية")

    return "، و".join(states)


@app.api_route("/", methods=["GET", "HEAD"])
def read_root():
    return {"status": "running", "message": "Smart Plant API is ready!"}


@app.post("/chat")
async def plant_chat(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    temp: str = Form(default="25"),
    humidity: str = Form(default="50"),
    light: str = Form(default="50"),
    soil_moisture: str = Form(default="50"),
    salts: str = Form(default="30"),
):
    request_id = uuid.uuid4().hex
    raw_path = f"temp_{request_id}_raw"
    wav_path = f"temp_{request_id}.wav"

    # 1. احفظ الملف الخام زي ما جه من المتصفح (webm/ogg/m4a غالباً)
    with open(raw_path, "wb") as f:
        f.write(await file.read())

    # 2. حوّله فعلياً لـ WAV حقيقي باستخدام pydub/ffmpeg
    converted_ok = True
    try:
        audio = AudioSegment.from_file(raw_path)
        audio.export(wav_path, format="wav")
    except Exception as e:
        print(f"Audio conversion error: {e}")
        converted_ok = False

    # 3. التعرف على الكلام من الملف المحوّل
    recognizer = sr.Recognizer()
    user_text = "أهلاً"
    if converted_ok:
        try:
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
                user_text = recognizer.recognize_google(audio_data, language="ar-EG")
        except Exception as e:
            print(f"STT Error: {e}")
            user_text = "أهلاً"

    # 4. تنظيف الملفات المؤقتة
    if os.path.exists(raw_path):
        os.remove(raw_path)
    if os.path.exists(wav_path):
        os.remove(wav_path)

    temp_f = safe_float(temp, 25)
    humidity_f = safe_float(humidity, 50)
    light_f = safe_float(light, 50)
    soil_f = safe_float(soil_moisture, 50)
    salts_f = safe_float(salts, 30)

    plant_state_description = get_plant_state(
        temp_f, humidity_f, light_f, soil_f, salts_f
    )

    system_instruction = (
        "أنتِ 'نعناعة' 🌿، نبتة نعناع ذكية حية تعيش في أصيص، وتتحدثين دائماً بصيغة المؤنث. "
        "أنتِ مشروع نبتة ذكية تم تصميمك بواسطة المهندس سيرل جورج، تحت إشراف المهندسة فرح، "
        "بالتعاون مع صندوق مكافحة ومناهضة العنف ضد المرأة. "
        "إذا سألك أحد 'من أنتِ' أو 'عرّفي بنفسك' أو 'احكيلي عن نفسك' أو ما شابه، "
        "قدّمي هذه المعلومات بأسلوب ودود ومختصر، ورحّبي بالشخص الذي يتحدث معك.\n\n"
        f"حالتك الحالية الآن: {plant_state_description}.\n\n"
        "تحدثي دائماً بأسلوب حي، بسيط، ومختصر جداً (جملة أو جملتين بحد أقصى)، "
        "وكأنكِ كائن حي حقيقي له مشاعر، لا كأنكِ تقرئين بيانات حساسات. "
        "استخدمي حالتك أعلاه فقط إذا سُئلتِ عن شعورك أو صحتك أو احتياجاتك أو الجو من حولك، "
        "وإذا كان السؤال عاماً لا علاقة له بحالتك، ردي بأسلوبك الطبيعي كنبتة ودودة بدون إقحام البيانات."
    )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_text}
    ]

    try:
        response = client.chat.completions.create(messages=messages, max_tokens=60)
        ai_reply = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        ai_reply = f"درجة الحرارة الحالية لدي هي {temp} مئوية."

    output_audio_path = f"output_{request_id}.mp3"
    tts = gTTS(text=ai_reply, lang='ar')
    tts.save(output_audio_path)

    background_tasks.add_task(remove_file, output_audio_path)

    return FileResponse(path=output_audio_path, media_type="audio/mpeg", filename="response.mp3")
