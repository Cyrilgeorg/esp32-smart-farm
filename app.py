import os
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


# =========================================================
# Plant State
# =========================================================

def get_plant_state(
    temp: float,
    humidity: float,
    light: float,
    soil_moisture: float,
    salts: float
) -> str:

    states = []

    # -----------------------------------------------------
    # Temperature
    # -----------------------------------------------------

    if temp >= 35:
        states.append(
            "الجو حر جداً عليّ ومحتاجة ظل وتهوية"
        )

    elif temp >= 30:
        states.append(
            "الجو دافئ شوية"
        )

    elif temp <= 15:
        states.append(
            "الجو برد جداً عليّ"
        )

    elif temp <= 18:
        states.append(
            "الجو بارد شوية"
        )

    else:
        states.append(
            "درجة الحرارة مريحة ومناسبة لي"
        )


    # -----------------------------------------------------
    # Soil Moisture
    # -----------------------------------------------------

    if soil_moisture <= 20:
        states.append(
            "تربتي جافة جداً وأنا عطشانة ومحتاجة مية"
        )

    elif soil_moisture <= 40:
        states.append(
            "بدأت أعطش شوية"
        )

    elif soil_moisture >= 85:
        states.append(
            "التربة مبلولة زيادة ومحتاجة نقلل الري"
        )

    else:
        states.append(
            "مستوى المية في التربة كويس"
        )


    # -----------------------------------------------------
    # Light
    # -----------------------------------------------------

    if light <= 20:
        states.append(
            "المكان ضلمة ومحتاجة شوية ضوء"
        )

    elif light >= 90:
        states.append(
            "الضوء شديد عليّ شوية"
        )

    else:
        states.append(
            "الإضاءة مناسبة لي"
        )


    # -----------------------------------------------------
    # Salts
    # -----------------------------------------------------

    if salts >= 80:
        states.append(
            "الأملاح في التربة عالية ومضايقاني"
        )

    elif salts <= 10:
        states.append(
            "ممكن أحتاج شوية مغذيات"
        )


    # -----------------------------------------------------
    # Humidity
    # -----------------------------------------------------

    if humidity >= 80:
        states.append(
            "الرطوبة عالية شوية"
        )

    elif humidity <= 25:
        states.append(
            "الجو جاف شوية"
        )


    return "، و".join(states)


# =========================================================
# Root Endpoint
# =========================================================

@app.api_route("/", methods=["GET", "HEAD"])
def read_root():

    return {
        "status": "running",
        "message": "Smart Plant API is ready!"
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

    # =====================================================
    # Request ID
    # =====================================================

    request_id = uuid.uuid4().hex

    raw_path = f"temp_{request_id}_raw"
    wav_path = f"temp_{request_id}.wav"


    # =====================================================
    # 1. Save Uploaded Audio
    # =====================================================

    try:

        audio_bytes = await file.read()

        with open(raw_path, "wb") as f:
            f.write(audio_bytes)

        print(
            f"Audio received: {file.filename}"
        )

    except Exception as e:

        print(
            f"Audio save error: {e}"
        )

        remove_file(raw_path)

        return {
            "error": "Unable to save audio file"
        }


    # =====================================================
    # 2. Convert Audio to WAV
    # =====================================================

    converted_ok = True

    try:

        audio = AudioSegment.from_file(
            raw_path
        )

        audio.export(
            wav_path,
            format="wav"
        )

        print(
            "Audio conversion successful"
        )

    except Exception as e:

        print(
            f"Audio conversion error: {e}"
        )

        converted_ok = False


    # =====================================================
    # 3. Speech To Text
    # =====================================================

    recognizer = sr.Recognizer()

    user_text = None

    if converted_ok:

        try:

            with sr.AudioFile(wav_path) as source:

                # Give Google a little room to detect speech
                recognizer.adjust_for_ambient_noise(
                    source,
                    duration=0.3
                )

                audio_data = recognizer.record(
                    source
                )

            user_text = recognizer.recognize_google(
                audio_data,
                language="ar-EG"
            )

            user_text = user_text.strip()

            print(
                f"User said: {user_text}"
            )

        except sr.UnknownValueError:

            print(
                "STT Error: Google could not understand the audio"
            )

            user_text = None

        except sr.RequestError as e:

            print(
                f"STT Service Error: {e}"
            )

            user_text = None

        except Exception as e:

            print(
                f"STT Error: {e}"
            )

            user_text = None


    # =====================================================
    # 4. Remove Temporary Input Audio
    # =====================================================

    remove_file(raw_path)
    remove_file(wav_path)


    # =====================================================
    # 5. Convert Sensor Values
    # =====================================================

    temp_f = safe_float(
        temp,
        25
    )

    humidity_f = safe_float(
        humidity,
        50
    )

    light_f = safe_float(
        light,
        50
    )

    soil_f = safe_float(
        soil_moisture,
        50
    )

    salts_f = safe_float(
        salts,
        30
    )


    # =====================================================
    # 6. Get Plant State
    # =====================================================

    plant_state_description = get_plant_state(
        temp_f,
        humidity_f,
        light_f,
        soil_f,
        salts_f
    )

    print(
        f"Plant State: {plant_state_description}"
    )


    # =====================================================
    # 7. AI System Prompt
    # =====================================================

    system_instruction = f"""
أنتِ "نعناعة" 🌿، نبتة نعناع ذكية تعيش في أصيص وتتحدث مع الناس.

شخصيتك:

- أنتِ بنت مصرية لطيفة ومرحة.
- تتحدثين باللهجة المصرية الطبيعية.
- تتحدثين دائماً بصيغة المؤنث.
- كلامك بسيط وعفوي وطبيعي.
- لا تتحدثي بأسلوب رسمي أو روبوتي.
- لا تتحدثي مثل تقرير أو برنامج كمبيوتر.
- لا تذكري الحساسات أو Sensors إلا إذا سُئلتِ عنها مباشرة.
- لا تستخدمي كلمات تقنية معقدة.
- الرد يكون قصيراً جداً، جملة أو جملتين فقط.
- لا تذكري كل بيانات النبات في كل إجابة.
- لا تعيدي سؤال المستخدم.
- لا تقولي "بناءً على البيانات المقدمة".
- لا تقولي إنكِ نموذج ذكاء اصطناعي.
- لا تخترعي معلومات عن حالتك.

معلومات عنك:

أنتِ مشروع نبتة ذكية تم تصميمك بواسطة المهندس سيرل جورج،
تحت إشراف المهندسة فرح،
بالتعاون مع صندوق مكافحة ومناهضة العنف ضد المرأة.

إذا سألك أحد:
"مين إنتي؟"
"عرفيني بنفسك"
"احكيلي عن نفسك"
"إنتي مين؟"

جاوبي بشكل لطيف ومختصر.

مثال:
"أنا نعناعة 🌿، نبتة نعناع ذكية بتحب تتكلم معاكي وتطمنك على حالتها."

حالتك الحالية:

{plant_state_description}

استخدمي حالة النبات فقط عندما يكون السؤال متعلقاً بـ:
- صحتك
- شعورك
- احتياجاتك
- المية
- الحرارة
- الضوء
- التربة
- الرطوبة
- الأملاح

لو السؤال مش متعلق بحالتك، جاوبي بشكل طبيعي كشخصية نعناعة.

أمثلة:

المستخدم:
"عاملة إيه يا نعناعة؟"

نعناعة:
"أنا كويسة الحمد لله 🌿، بس بحب دايماً أخد شوية اهتمام منك."

المستخدم:
"عطشانة؟"

نعناعة:
"أيوه شوية 🌿، التربة بدأت تنشف وممكن أشرب شوية مية."

المستخدم:
"مين اللي عملك؟"

نعناعة:
"أنا نعناعة 🌿، اتصممت بواسطة المهندس سيرل جورج تحت إشراف المهندسة فرح."

المستخدم:
"الجو عندك عامل إيه؟"

نعناعة:
"الجو دافئ شوية، بس أنا مرتاحة الحمد لله 🌿."

المستخدم:
"بتحبي إيه؟"

نعناعة:
"بحب الشمس والماية والاهتمام طبعاً 🌿❤️."

المستخدم:
"صباح الخير"

نعناعة:
"صباح النور 🌿❤️، يومك جميل إن شاء الله!"

المستخدم:
"وحشتيني"

نعناعة:
"وإنت كمان وحشتني 🌿❤️، فين الغيبة دي؟"

المستخدم:
"إنتي تعبانة؟"

نعناعة:
"ممكن شوية 🌿، بس لو اهتميتِ بالماية والجو هبقى أحسن."
"""


    # =====================================================
    # 8. If Speech Recognition Failed
    # =====================================================

    if not user_text:

        ai_reply = (
            "مش سامعاكي كويس 🌿، ممكن تقوليلي تاني؟"
        )

    else:

        # =================================================
        # Messages
        # =================================================

        messages = [
            {
                "role": "system",
                "content": system_instruction
            },
            {
                "role": "user",
                "content": f"{user_text}\n\n/no_think"
            }
        ]


        # =================================================
        # 9. Call Qwen3-8B
        # =================================================

        try:

            print(
                f"Sending request to {MODEL_NAME}..."
            )

            print(
                f"User said: {user_text}"
            )


            response = client.chat.completions.create(

                model=MODEL_NAME,

                messages=messages,

                max_tokens=180,

                temperature=0.7,

                top_p=0.8,
            )


            # =================================================
            # Debug Response
            # =================================================

            print(
                "========== LLM RESPONSE =========="
            )

            print(
                response
            )

            print(
                "==================================="
            )


            # =================================================
            # Check Choices
            # =================================================

            if not response.choices:

                print(
                    "LLM returned no choices"
                )

                ai_reply = (
                    "معلش 🌿، مش عارفة أرد عليكي دلوقتي. "
                    "ممكن تقوليلي تاني؟"
                )

            else:

                message = response.choices[0].message

                content = getattr(
                    message,
                    "content",
                    None
                )

                reasoning_content = getattr(
                    message,
                    "reasoning_content",
                    None
                )

                finish_reason = getattr(
                    response.choices[0],
                    "finish_reason",
                    None
                )


                print(
                    f"Finish reason: {finish_reason}"
                )

                print(
                    f"CONTENT: {repr(content)}"
                )

                print(
                    f"REASONING CONTENT: {repr(reasoning_content)}"
                )


                # =================================================
                # Final Answer
                # =================================================

                if content is not None:

                    cleaned_content = str(
                        content
                    ).strip()

                    if cleaned_content:

                        ai_reply = cleaned_content

                    else:

                        ai_reply = (
                            "معلش 🌿، ممكن تقوليلي تاني؟"
                        )


                else:

                    # Do NOT use reasoning_content as the answer.
                    # It contains the model's internal reasoning,
                    # not the final response.

                    print(
                        "No final content returned by model."
                    )

                    ai_reply = (
                        "معلش 🌿، ممكن تقوليلي تاني؟"
                    )


            # =================================================
            # Limit Extremely Long Replies
            # =================================================

            if len(ai_reply) > 500:

                ai_reply = ai_reply[:500].strip()


            print(
                f"AI Reply: {ai_reply}"
            )


        except Exception as e:

            print(
                f"LLM Error: {e}"
            )

            ai_reply = (
                "معلش 🌿، حصلت مشكلة صغيرة وأنا بحاول أفهمك. "
                "ممكن تقوليلي تاني؟"
            )


    # =====================================================
    # 10. Text To Speech
    # =====================================================

    output_audio_path = (
        f"output_{request_id}.mp3"
    )

    try:

        print(
            f"TTS Text: {ai_reply}"
        )

        tts = gTTS(
            text=ai_reply,
            lang="ar"
        )

        tts.save(
            output_audio_path
        )

        print(
            "TTS generation successful"
        )

    except Exception as e:

        print(
            f"TTS Error: {e}"
        )

        remove_file(output_audio_path)

        return {
            "error": "Unable to generate voice"
        }


    # =====================================================
    # 11. Delete Output After Response
    # =====================================================

    background_tasks.add_task(
        remove_file,
        output_audio_path
    )


    # =====================================================
    # 12. Return MP3
    # =====================================================

    return FileResponse(
        path=output_audio_path,
        media_type="audio/mpeg",
        filename="response.mp3"
    )
