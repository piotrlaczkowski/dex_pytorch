import os

from dotenv import load_dotenv
from notifiers import get_notifier

load_dotenv()


def msg_telegram(text: str):
    """
    Helper to send telegram msgs.
    """
    get_notifier("telegram").notify(
        message=text,
        token=os.environ.get("TELEGRAM_TOKEN"),
        chat_id=os.environ.get("TELEGRAM_CHAT_ID"),
        parse_mode="markdown",
    )
