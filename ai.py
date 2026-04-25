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

        self._client = OpenAI(
            base_url=self.BASE_URL,
            api_key=api_key,
        )

        self._extra_headers = {
            "HTTP-Header": self.SITE_URL,
            "X-Title": self.APP_NAME
        }
        self.messages:list[dict] = [{"role": "system", "content": self.system_prompt}]
        self.tools = tools

    def chat(self, user_message: str) -> list[dict]:


        self.messages.append({"role": "user", "content": user_message})

        completion = self._client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tools=self.tools,
            extra_headers=self._extra_headers,
        )

        self.messages.append(completion.choices[0].message)

        if len(self.messages) > self.max_message_history:
            messages = self.messages[-self.max_message_history:]

        return messages

    def choose_tool(self, user_message: str):
        self.messages.append({"role": "user", "content": user_message})

        tools = self._client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tools=self.tools,
            extra_headers=self._extra_headers,
        ).message.tool_calls
        return tools