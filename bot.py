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


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id

    if user_id not in ALLOWED_USER_IDS:
        logger.info("Игнорирую голосовое от постороннего ID: %s", user_id)
        return

    voice_file = await update.message.voice.get_file()
    audio_bytes = bytes(await voice_file.download_as_bytearray())

    text = transcribe_voice(audio_bytes)
    await update.message.reply_text(f"Распознал: {text}")


def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.run_polling()


if __name__ == "__main__":
    main()
