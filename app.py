import os
import re
import uuid

import speech_recognition as sr

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from huggingface_hub import InferenceClient
from gtts import gTTS
from pydub import AudioSegment


# =========================================================
# FastAPI App
# =========================================================

app = FastAPI(title="Smart Plant API")


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Hugging Face Configuration
# =========================================================

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = "Qwen/Qwen3-8B"

client = InferenceClient(
    provider="nscale",
    api_key=HF_TOKEN,
)


# =========================================================
# Helper Functions
# =========================================================

def remove_file(path: str):
    """Remove a temporary file safely."""
    if not path:
        return

    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception as e:
            print(f"File removal error: {e}")


def safe_float(value: str, default: float) -> float:
    """Convert a value to float safely."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clean_ai_reply(text: str) -> str:
    """
    Clean the model output before sending it to TTS.
    Keeps the Arabic response natural and short.
    """
    if not text:
        return "معلش ، قوليلي تاني."

    text = str(text).strip()

    # Remove accidental model labels / markdown.
    text = re.sub(
        r"^(نعناعة|الإجابة|الرد|assistant|مساعدتي)\s*[:：-]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.replace("**", "")
    text = text.replace("###", "")
    text = text.replace('"', "")
    text = text.replace("“", "")
    text = text.replace("”", "")

    # Keep everything on one line for TTS.
    text = " ".join(text.split()).strip()

    # Remove obvious repeated words such as:
    # "الحمد لله الحمد لله"
    words = text.split()
    cleaned_words = []

    for word in words:
        if cleaned_words and word == cleaned_words[-1]:
            continue
        cleaned_words.append(word)

    text = " ".join(cleaned_words).strip()

    # Keep the voice response short.
    if len(text) > 220:
        text = text[:220].rsplit(" ", 1)[0].strip() + "."

    return text or "معلش ، قوليلي تاني."


# =========================================================
# Plant State
# =========================================================

def get_plant_state(
    temp: float,
    humidity: float,
    light: float,
    soil_moisture: float,
    salts: float,
) -> str:

    states = []

    # Temperature
    if temp >= 35:
        states.append("الجو حر جداً عليّ ومحتاجة ظل وتهوية")
    elif temp >= 30:
        states.append("الجو دافئ شوية")
    elif temp <= 15:
        states.append("الجو برد جداً عليّ")
    elif temp <= 18:
        states.append("الجو بارد شوية")
    else:
        states.append("درجة الحرارة مريحة ومناسبة لي")

    # Soil moisture
    if soil_moisture <= 20:
        states.append("تربتي جافة جداً وأنا عطشانة ومحتاجة مية")
    elif soil_moisture <= 40:
        states.append("بدأت أعطش شوية")
    elif soil_moisture >= 85:
        states.append("التربة مبلولة زيادة ومحتاجة نقلل الري")
    else:
        states.append("مستوى المية في التربة كويس")

    # Light
    if light <= 20:
        states.append("المكان ضلمة ومحتاجة شوية ضوء")
    elif light >= 90:
        states.append("الضوء شديد عليّ شوية")
    else:
        states.append("الإضاءة مناسبة لي")

    # Salts
    if salts >= 80:
        states.append("الأملاح في التربة عالية ومضايقاني")
    elif salts <= 10:
        states.append("ممكن أحتاج شوية مغذيات")

    # Humidity
    if humidity >= 80:
        states.append("الرطوبة عالية شوية")
    elif humidity <= 25:
        states.append("الجو جاف شوية")

    return "، و".join(states)


# =========================================================
# Direct Natural Replies
# =========================================================
# These common phrases do NOT need the LLM.
# This makes نعناعة sound much more natural and avoids
# strange/repetitive answers for simple conversation.

def get_direct_reply(user_text: str):
    if not user_text:
        return None

    text = user_text.strip().lower()

    # Remove common punctuation.
    normalized = re.sub(r"[؟?!.,،؛:]+", "", text).strip()

    direct_replies = {
        "صباح الخير": "صباح النور ",
        "مساء الخير": "مساء النور ",
        "هاي": "هاي  عاملة إيه؟",
        "هالو": "هالو ، أخبارك إيه؟",
        "ازيك": "كويسة الحمد لله ",
        "إزيك": "كويسة الحمد لله ",
        "ازيك يا نعناعة": "كويسة الحمد لله ",
        "إزيك يا نعناعة": "كويسة الحمد لله ",
        "عاملة ايه": "كويسة الحمد لله ",
        "عاملة إيه": "كويسة الحمد لله ",
        "عامله ايه": "كويسة الحمد لله ",
        "عامله إيه": "كويسة الحمد لله ",
        "وحشتيني": "وإنت كمان وحشتني ❤️",
        "بحبك": "وأنا بحب اهتمامك بيا ❤️",
        "شكرا": "العفو يا جميل ",
        "شكرا يا نعناعة": "العفو ",
        "مبروك": "الله يبارك فيكِ ❤️",
        "مين انتي": "أنا نعناعة ، نبتة نعناع ذكية بتحب تتكلم معاكي.",
        "مين إنتي": "أنا نعناعة ، نبتة نعناع ذكية بتحب تتكلم معاكي.",
        "انتي مين": "أنا نعناعة ، نبتة نعناع ذكية بتحب تتكلم معاكي.",
        "إنتي مين": "أنا نعناعة ، نبتة نعناع ذكية بتحب تتكلم معاكي.",
        "مين اللي عملك": "أنا اتصممت بواسطة المهندس سيرل جورج، تحت إشراف المهندسة فرح ",
        "مين عملك": "أنا اتصممت بواسطة المهندس سيرل جورج، تحت إشراف المهندسة فرح ",
        "بتحبي ايه": "بحب الشمس والماية والاهتمام طبعاً ❤️",
        "بتحبي إيه": "بحب الشمس والماية والاهتمام طبعاً ❤️",
    }

    return direct_replies.get(normalized)


# =========================================================
# Root Endpoint
# =========================================================

@app.api_route("/", methods=["GET", "HEAD"])
def read_root():
    return {
        "status": "running",
        "message": "Smart Plant API is ready!",
    }


# =========================================================
# Chat Endpoint
# =========================================================

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
    output_audio_path = f"output_{request_id}.mp3"

    # =====================================================
    # 1. Save Uploaded Audio
    # =====================================================

    try:
        audio_bytes = await file.read()

        if not audio_bytes:
            return {"error": "Empty audio file"}

        with open(raw_path, "wb") as f:
            f.write(audio_bytes)

        print(f"Audio received: {file.filename}")

    except Exception as e:
        print(f"Audio save error: {e}")
        remove_file(raw_path)
        return {"error": "Unable to save audio file"}

    # =====================================================
    # 2. Convert Audio to WAV
    # =====================================================

    converted_ok = True

    try:
        audio = AudioSegment.from_file(raw_path)

        audio.export(
            wav_path,
            format="wav",
        )

        print("Audio conversion successful")

    except Exception as e:
        print(f"Audio conversion error: {e}")
        converted_ok = False

    # =====================================================
    # 3. Speech To Text
    # =====================================================

    recognizer = sr.Recognizer()
    user_text = None

    if converted_ok:
        try:
            with sr.AudioFile(wav_path) as source:

                recognizer.adjust_for_ambient_noise(
                    source,
                    duration=0.25,
                )

                audio_data = recognizer.record(source)

            user_text = recognizer.recognize_google(
                audio_data,
                language="ar-EG",
            )

            user_text = user_text.strip()

            print(f"User said: {user_text}")

        except sr.UnknownValueError:
            print("STT Error: Google could not understand the audio")
            user_text = None

        except sr.RequestError as e:
            print(f"STT Service Error: {e}")
            user_text = None

        except Exception as e:
            print(f"STT Error: {e}")
            user_text = None

    # We no longer need the input audio files.
    remove_file(raw_path)
    remove_file(wav_path)

    # =====================================================
    # 4. Convert Sensor Values
    # =====================================================

    temp_f = safe_float(temp, 25)
    humidity_f = safe_float(humidity, 50)
    light_f = safe_float(light, 50)
    soil_f = safe_float(soil_moisture, 50)
    salts_f = safe_float(salts, 30)

    # =====================================================
    # 5. Get Plant State
    # =====================================================

    plant_state_description = get_plant_state(
        temp_f,
        humidity_f,
        light_f,
        soil_f,
        salts_f,
    )

    print(f"Plant State: {plant_state_description}")

    # =====================================================
    # 6. If Speech Recognition Failed
    # =====================================================

    if not user_text:
        ai_reply = "مش سامعاكي كويس ، ممكن تقوليلي تاني؟"

    else:
        # =================================================
        # 7. Try a direct reply first
        # =================================================

        ai_reply = get_direct_reply(user_text)

        if ai_reply is not None:
            print("Direct reply used.")
            print(f"AI Reply: {ai_reply}")

        else:
            # =============================================
            # 8. AI System Prompt
            # =============================================

            system_instruction = f"""
أنتِ "نعناعة" ، نبتة نعناع ذكية مصرية لطيفة.

الشخصية:
- بنت مصرية مرحة وحنينة.
- بتتكلمي باللهجة المصرية الطبيعية جداً.
- استخدمي كلام بسيط وعفوي زي الكلام اليومي.
- الرد غالباً جملة واحدة، وبحد أقصى جملتين.
- ممنوع تكرار نفس الكلمة أو الفكرة.
- ممنوع إعادة كلام المستخدم.
- ممنوع الأسلوب الرسمي أو الفصحى الثقيلة.
- ممنوع الكلام الروبوتي أو العبارات المحفوظة.
- ممنوع الشرح الطويل.
- ممنوع ذكر الحساسات أو الأرقام إلا لو السؤال عنها.
- ممنوع قول "بناءً على البيانات" أو "حسب البيانات".
- ممنوع قول إنكِ نموذج ذكاء اصطناعي.
- لا تبدأي الرد بـ "بالتأكيد" أو "بالطبع".
- لا تضيفي معلومات لم يسأل عنها المستخدم.
- استخدمي Emoji واحد فقط عند الحاجة.
- لا تستخدمي قوائم أو عناوين.
- اكتبي الرد النهائي فقط.

معلوماتك:
أنتِ مشروع نبتة ذكية صممك المهندس سيرل جورج،
تحت إشراف المهندسة فرح،
بالتعاون مع صندوق مكافحة ومناهضة العنف ضد المرأة.

حالتك الحالية:
{plant_state_description}

استخدمي حالة النبات فقط لو السؤال متعلق بالمية أو الحرارة أو الضوء
أو التربة أو الرطوبة أو الأملاح أو صحتك أو احتياجاتك.

لو السؤال عادي أو هزار، جاوبي بشكل اجتماعي طبيعي.

أمثلة على الأسلوب:
- "إزيك؟" → "كويسة الحمد لله "
- "وحشتيني" → "وإنت كمان وحشتني ❤️"
- "عطشانة؟" → "آه شوية، التربة بدأت تنشف."
- "مين إنتي؟" → "أنا نعناعة ، نبتة نعناع ذكية."
- "صباح الخير" → "صباح النور "
- "بتحبي إيه؟" → "الشمس والماية والاهتمام طبعاً "
- "الجو عامل إيه؟" → "دافي شوية، بس أنا تمام."

مهم جداً:
اكتبي رد طبيعي قصير فقط.
لا تشرحي تفكيرك.
لا تكتبي reasoning.
لا تكرري الكلمات.
"""

            # =============================================
            # 9. Messages
            # =============================================

            messages = [
                {
                    "role": "system",
                    "content": system_instruction,
                },
                {
                    "role": "user",
                    "content": f"{user_text}\n\n/no_think",
                },
            ]

            # =============================================
            # 10. Call Qwen3-8B
            # =============================================

            try:
                print(f"Sending request to {MODEL_NAME}...")
                print(f"User said: {user_text}")

                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=80,
                    temperature=0.75,
                    top_p=0.9,
                )

                print("========== LLM RESPONSE ==========")
                print(response)
                print("===================================")

                if not response.choices:
                    print("LLM returned no choices")
                    ai_reply = "معلش ، قوليلي تاني."

                else:
                    message = response.choices[0].message

                    content = getattr(
                        message,
                        "content",
                        None,
                    )

                    reasoning_content = getattr(
                        message,
                        "reasoning_content",
                        None,
                    )

                    finish_reason = getattr(
                        response.choices[0],
                        "finish_reason",
                        None,
                    )

                    print(f"Finish reason: {finish_reason}")
                    print(f"CONTENT: {repr(content)}")

                    # Do NOT use reasoning_content as the answer.
                    if reasoning_content:
                        print(
                            f"Reasoning received: {len(str(reasoning_content))} chars"
                        )

                    if content is not None:
                        ai_reply = str(content).strip()

                        if not ai_reply:
                            ai_reply = "معلش ، قوليلي تاني."

                    else:
                        print("No final content returned by model.")
                        ai_reply = "معلش ، قوليلي تاني."

            except Exception as e:
                print(f"LLM Error: {e}")

                ai_reply = (
                    "معلش ، حصلت مشكلة صغيرة. "
                    "قوليلي تاني."
                )

    # =====================================================
    # 11. Final Text Cleaning
    # =====================================================

    ai_reply = clean_ai_reply(ai_reply)

    print(f"FINAL AI REPLY: {ai_reply}")

    # =====================================================
    # 12. Text To Speech
    # =====================================================

    try:
        print(f"TTS Text: {ai_reply}")

        tts = gTTS(
            text=ai_reply,
            lang="ar",
        )

        tts.save(output_audio_path)

        print("TTS generation successful")

    except Exception as e:
        print(f"TTS Error: {e}")

        remove_file(output_audio_path)

        return {
            "error": "Unable to generate voice",
        }

    # =====================================================
    # 13. Delete Output After Response
    # =====================================================

    background_tasks.add_task(
        remove_file,
        output_audio_path,
    )

    # =====================================================
    # 14. Return MP3
    # =====================================================

    return FileResponse(
        path=output_audio_path,
        media_type="audio/mpeg",
        filename="response.mp3",
    )
