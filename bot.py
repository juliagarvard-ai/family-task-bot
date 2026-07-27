import datetime
import json
import logging
import os

from dotenv import load_dotenv
from groq import Groq
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ALLOWED_USER_IDS = {
    int(uid.strip())
    for uid in os.getenv("ALLOWED_USER_IDS", "").split(",")
    if uid.strip()
}

groq_client = Groq(api_key=GROQ_API_KEY)


def transcribe_voice(audio_bytes: bytes) -> str:
    transcription = groq_client.audio.transcriptions.create(
        file=("voice.ogg", audio_bytes),
        model="whisper-large-v3",
        language="ru",
    )
    return transcription.text


def parse_task(text: str) -> dict:
    today = datetime.date.today().isoformat()
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты помощник семейного бота задач. Из текста пользователя "
                    "извлеки задачу и верни СТРОГО JSON с полями:\n"
                    '- "assignee": имя человека, который должен выполнить задачу '
                    "(если явно не названо конкретное имя — null);\n"
                    '- "task": краткое описание самой задачи, без имени и даты;\n'
                    '- "due_date": дата выполнения в формате YYYY-MM-DD, если в '
                    "тексте есть дата, день недели или относительное указание "
                    f"(вычисли относительно сегодняшней даты {today}); если даты "
                    "нет — null."
                ),
            },
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(completion.choices[0].message.content)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id

    if user_id not in ALLOWED_USER_IDS:
        logger.info("Игнорирую голосовое от постороннего ID: %s", user_id)
        return

    voice_file = await update.message.voice.get_file()
    audio_bytes = bytes(await voice_file.download_as_bytearray())

    text = transcribe_voice(audio_bytes)
    parsed = parse_task(text)

    await update.message.reply_text(
        f"Распознал: {text}\n\n"
        f"Кто: {parsed.get('assignee') or '—'}\n"
        f"Что: {parsed.get('task') or '—'}\n"
        f"Когда: {parsed.get('due_date') or '—'}"
    )


def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.run_polling()


if __name__ == "__main__":
    main()
