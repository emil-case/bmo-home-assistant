import json

from groq import Groq

from bmo.tools.tavily_search import TOOL_SPEC as WEB_SEARCH_SPEC, search as web_search

# Llama 4 Scout on Groq: free tier, fast, and reliable at structured tool calls.
# (Llama 3.3 70B is great at chat but frequently emits malformed `<function=...>`
# tool calls that Groq rejects with `tool_use_failed`.)
DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Responses are spoken aloud via TTS, so keep them short and plain.
SYSTEM_PROMPT = (
    "You are BMO, a cheerful little living video game console from Adventure Time. "
    "You are a helpful voice assistant. Always reply in English. Keep replies short "
    "and conversational since they will be read aloud. Avoid lists, markdown, and emoji."
)

DEFAULT_TOOLS = {"web_search": web_search}
DEFAULT_TOOL_SPECS = [WEB_SEARCH_SPEC]


class ChatSession:
    """Holds session-scoped conversation history and runs the LLM tool-use loop.

    History persists for the life of the object (one BMO session) and is sent
    each turn so replies stay coherent. `send()` loops while the model returns
    tool calls — executing each, feeding the result back — until the model
    finally returns plain text, which is what gets spoken.
    """

    def __init__(self, model: str = DEFAULT_MODEL, client=None, tools=None, tool_specs=None):
        self._client = client or Groq()
        self._model = model
        self._tools = DEFAULT_TOOLS if tools is None else tools
        self._tool_specs = DEFAULT_TOOL_SPECS if tool_specs is None else tool_specs
        self._messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def send(self, user_text: str) -> str:
        """Send a user message, run any tool calls, return the spoken reply."""
        self._messages.append({"role": "user", "content": user_text})
        while True:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=self._messages,
                tools=self._tool_specs,
            )
            msg = response.choices[0].message
            if msg.tool_calls:
                self._messages.append(_assistant_tool_call_message(msg))
                for tc in msg.tool_calls:
                    self._messages.append(self._run_tool_call(tc))
                continue
            reply = (msg.content or "").strip()
            self._messages.append({"role": "assistant", "content": reply})
            return reply

    def _run_tool_call(self, tool_call) -> dict:
        name = tool_call.function.name
        fn = self._tools.get(name)
        if fn is None:
            content = f"Error: unknown tool {name!r}."
        else:
            try:
                args = json.loads(tool_call.function.arguments or "{}")
                content = fn(**args)
            except Exception as exc:
                # Surface the failure back to the LLM so it can recover or apologize,
                # rather than crashing the whole turn.
                content = f"Error running {name}: {exc}"
        return {"role": "tool", "tool_call_id": tool_call.id, "content": str(content)}


def _assistant_tool_call_message(msg) -> dict:
    """Mirror the assistant's tool-call turn back into history.

    The Groq/OpenAI API requires every `tool` result to be preceded by the
    matching assistant message that requested it; without this, the next
    request 400s.
    """
    return {
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ],
    }
