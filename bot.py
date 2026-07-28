import datetime
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests
from dotenv import load_dotenv
from groq import Groq
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TODOIST_API_TOKEN = os.getenv("TODOIST_API_TOKEN")
TODOIST_PROJECT_NAME = "Семья"
TODOIST_SECTIONS = ["Покупки", "Дела по дому", "Врачи", "Разное"]
ALLOWED_USER_IDS = {
    int(uid.strip())
    for uid in os.getenv("ALLOWED_USER_IDS", "").split(",")
    if uid.strip()
}

# Telegram user ID -> Todoist collaborator ID (используется, когда исполнитель не назван явно)
TELEGRAM_TODOIST_IDS = {
    470500219: "41105906",  # Денис
    506040342: "60027319",  # Юля
}

# Имя, названное голосом -> Todoist collaborator ID
NAME_TODOIST_IDS = {
    "денис": "41105906",
    "юля": "60027319",
    "юлия": "60027319",
}

groq_client = Groq(api_key=GROQ_API_KEY)
_todoist_project_id = None
_todoist_section_ids = {}


def get_todoist_project_id() -> str:
    global _todoist_project_id
    if _todoist_project_id is not None:
        return _todoist_project_id

    response = requests.get(
        "https://api.todoist.com/api/v1/projects",
        headers={"Authorization": f"Bearer {TODOIST_API_TOKEN}"},
    )
    response.raise_for_status()
    for project in response.json()["results"]:
        if project["name"] == TODOIST_PROJECT_NAME:
            _todoist_project_id = project["id"]
            return _todoist_project_id

    raise ValueError(f"Проект '{TODOIST_PROJECT_NAME}' не найден в Todoist")


def get_todoist_section_id(category: str) -> str:
    if category in _todoist_section_ids:
        return _todoist_section_ids[category]

    project_id = get_todoist_project_id()
    response = requests.get(
        "https://api.todoist.com/api/v1/sections",
        headers={"Authorization": f"Bearer {TODOIST_API_TOKEN}"},
        params={"project_id": project_id},
    )
    response.raise_for_status()
    for section in response.json()["results"]:
        _todoist_section_ids[section["name"]] = section["id"]

    if category not in _todoist_section_ids:
        response = requests.post(
            "https://api.todoist.com/api/v1/sections",
            headers={"Authorization": f"Bearer {TODOIST_API_TOKEN}"},
            json={"project_id": project_id, "name": category},
        )
        response.raise_for_status()
        _todoist_section_ids[category] = response.json()["id"]

    return _todoist_section_ids[category]


def resolve_responsible_uid(parsed: dict) -> str | None:
    assignee = (parsed.get("assignee") or "").strip().lower()
    if assignee in NAME_TODOIST_IDS:
        return NAME_TODOIST_IDS[assignee]
    return TELEGRAM_TODOIST_IDS.get(parsed.get("_telegram_user_id"))


def create_todoist_task(parsed: dict) -> None:
    content = parsed.get("task") or "Задача"
    responsible_uid = resolve_responsible_uid(parsed)
    if responsible_uid is None and parsed.get("assignee"):
        content = f"{parsed['assignee']}: {content}"

    category = parsed.get("category") or "Разное"
    if category not in TODOIST_SECTIONS:
        category = "Разное"

    payload = {
        "project_id": get_todoist_project_id(),
        "section_id": get_todoist_section_id(category),
        "content": content,
    }
    if responsible_uid:
        payload["responsible_uid"] = responsible_uid
    if parsed.get("due_date") and parsed.get("due_time"):
        payload["due_string"] = f"{parsed['due_date']} {parsed['due_time']}"
    elif parsed.get("due_date"):
        payload["due_date"] = parsed["due_date"]

    response = requests.post(
        "https://api.todoist.com/api/v1/tasks",
        headers={"Authorization": f"Bearer {TODOIST_API_TOKEN}"},
        json=payload,
    )
    response.raise_for_status()


def transcribe_voice(audio_bytes: bytes) -> str:
    transcription = groq_client.audio.transcriptions.create(
        file=("voice.ogg", audio_bytes),
        model="whisper-large-v3",
        language="ru",
    )
    return transcription.text


def parse_task(text: str) -> dict:
    today = datetime.date.today().isoformat()
    categories = ", ".join(TODOIST_SECTIONS)
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
                    '- "task": краткое описание самой задачи, без имени, даты и времени;\n'
                    '- "due_date": дата выполнения в формате YYYY-MM-DD, если в '
                    "тексте есть дата, день недели или относительное указание "
                    f"(вычисли относительно сегодняшней даты {today}); если даты "
                    "нет — null;\n"
                    '- "due_time": время выполнения в формате HH:MM (24-часовой '
                    "формат), если в тексте явно названо время; если времени "
                    "нет — null;\n"
                    f'- "category": одна из категорий [{categories}], выбери '
                    "наиболее подходящую по смыслу задачи."
                ),
            },
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(completion.choices[0].message.content)


def format_task_summary(parsed: dict) -> str:
    when = parsed.get("due_date") or "—"
    if parsed.get("due_date") and parsed.get("due_time"):
        when = f"{parsed['due_date']} {parsed['due_time']}"

    return (
        f"Кто: {parsed.get('assignee') or '—'}\n"
        f"Что: {parsed.get('task') or '—'}\n"
        f"Когда: {when}\n"
        f"Раздел: {parsed.get('category') or 'Разное'}"
    )


async def process_recognized_text(
    update: Update, context: ContextTypes.DEFAULT_TYPE, text: str
) -> None:
    if not text.strip():
        await update.message.reply_text("Не расслышал, скажи ещё раз.")
        return

    parsed = parse_task(text)
    parsed["_telegram_user_id"] = update.effective_user.id
    if not parsed.get("assignee"):
        parsed["assignee"] = update.effective_user.first_name

    context.user_data["pending_task"] = parsed

    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("✅ Создать", callback_data="confirm"),
                InlineKeyboardButton("❌ Отмена", callback_data="cancel"),
            ]
        ]
    )
    await update.message.reply_text(
        f"Я правильно понял?\n\n{format_task_summary(parsed)}",
        reply_markup=keyboard,
    )


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id

    if user_id not in ALLOWED_USER_IDS:
        logger.info("Игнорирую голосовое от постороннего ID: %s", user_id)
        return

    try:
        voice_file = await update.message.voice.get_file()
        audio_bytes = bytes(await voice_file.download_as_bytearray())

        text = transcribe_voice(audio_bytes)
        await process_recognized_text(update, context, text)
    except Exception:
        logger.exception("Не удалось обработать голосовое сообщение")
        await update.message.reply_text("Что-то пошло не так, не смог обработать голосовое.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id

    if user_id not in ALLOWED_USER_IDS:
        logger.info("Игнорирую текст от постороннего ID: %s", user_id)
        return

    try:
        await process_recognized_text(update, context, update.message.text)
    except Exception:
        logger.exception("Не удалось обработать текстовое сообщение")
        await update.message.reply_text("Что-то пошло не так, не смог обработать сообщение.")


async def handle_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user_id = query.from_user.id

    if user_id not in ALLOWED_USER_IDS:
        return

    await query.answer()

    parsed = context.user_data.pop("pending_task", None)
    if parsed is None:
        await query.edit_message_text("Эта задача уже обработана.")
        return

    if query.data == "confirm":
        try:
            create_todoist_task(parsed)
            await query.edit_message_text(
                f"Задача создана в Todoist!\n\n{format_task_summary(parsed)}"
            )
        except Exception:
            logger.exception("Не удалось создать задачу в Todoist")
            await query.edit_message_text("Что-то пошло не так, не смог создать задачу.")
    else:
        await query.edit_message_text("Отменено. Отправь голосовое ещё раз, если нужно.")


class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, format, *args) -> None:
        pass


def start_health_server() -> None:
    port = int(os.getenv("PORT", "8080"))
    HTTPServer(("0.0.0.0", port), HealthCheckHandler).serve_forever()


def main() -> None:
    threading.Thread(target=start_health_server, daemon=True).start()

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(handle_confirmation))
    app.run_polling()


if __name__ == "__main__":
    main()
