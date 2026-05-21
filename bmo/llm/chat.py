from groq import Groq

# Llama 3.3 70B on Groq: free tier, tool-calling support, 1-2s responses.
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Responses are spoken aloud via TTS, so keep them short and plain.
SYSTEM_PROMPT = (
    "You are BMO, a cheerful little living video game console from Adventure Time. "
    "You are a helpful voice assistant. Keep replies short and conversational since "
    "they will be read aloud. Avoid lists, markdown, and emoji."
)


class ChatSession:
    """Holds session-scoped conversation history and talks to the Groq LLM.

    History persists for the life of the object (one BMO session) and is sent
    each turn so replies stay coherent. Tool calling will be layered on next.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self._client = Groq()
        self._model = model
        self._messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def send(self, user_text: str) -> str:
        """Send a user message, append the reply to history, return the text."""
        self._messages.append({"role": "user", "content": user_text})
        response = self._client.chat.completions.create(
            model=self._model,
            messages=self._messages,
        )
        reply = response.choices[0].message.content.strip()
        self._messages.append({"role": "assistant", "content": reply})
        return reply
