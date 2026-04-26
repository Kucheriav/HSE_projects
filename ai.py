from openai import OpenAI
import os


class OpenRouterClient:
    BASE_URL = "https://openrouter.ai/api/v1"
    SITE_URL = "hse.ru"
    APP_NAME = "Test AI Bot"

    def __init__(self, api_key: str, model: str, system_prompt: str,
            max_message_history: int = 6, temperature: float = 0.7, max_tokens: int = 500,
                 tools: list[dict] | None = None):

        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_message_history = max_message_history
        self.tools = tools

        self._client = OpenAI(
            base_url=self.BASE_URL,
            api_key=api_key,
        )

        self._extra_headers = {
            "HTTP-Header": self.SITE_URL,
            "X-Title": self.APP_NAME
        }

    def chat(self, history: list[dict]) -> dict:
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=history,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tools=self.tools,
            extra_headers=self._extra_headers,
        )
        return completion.choices[0].message

    def choose_tool(self, history: list[dict]) -> list:
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=history,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tools=self.tools,
            extra_headers=self._extra_headers,
        )
        # Предположим, что tool_calls — это list, как у OpenAI
        if hasattr(completion.choices[0].message, "tool_calls"):
            return completion.choices[0].message.tool_calls
        return []