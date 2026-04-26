from ai import OpenRouterClient
from tools_functions import *
import json
import os

user_chats: dict[int, list[dict]] = {}
AI_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "qwen/qwen3.5-flash-02-23"
SYSTEM_PROMPT = "Bot for VK. Receives user requests, determines which tool to use, calls it and outputs the result."


TOOL_MAPPING = {
    "wiki_search": wiki_search,
    "generate_picture": generate_picture,
    "get_content_info": get_content_info
}

#TODO
tools_description = [{}]

ai_model = OpenRouterClient(
    api_key=AI_API_KEY,
    model=MODEL,
    system_prompt=SYSTEM_PROMPT,
    tools=tools_description,
)
def handle(user_message: str, user_id: int):

    # Получаем историю или инициализируем первой системной ролью
    history = user_chats.get(user_id)
    if not history:
        history = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Добавляем сообщение пользователя
    history.append({"role": "user", "content": user_message})

    # LLM выбирает инструмент
    tool_calls = ai_model.choose_tool(history)

    if not tool_calls:
        # Просто обычный ответ модели
        bot_reply = ai_model.chat(history)
        history.append(bot_reply)
        user_chats[user_id] = history[-8:]  # Обрезаем историю при необходимости
        return bot_reply['content'], None

    # Если инструмент выбран, вызываем его
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        func = TOOL_MAPPING.get(tool_name)
        if not func:
            answer = f"Неизвестная функция: {tool_name}"
            results.append(answer)
            continue
        try:
            tool_response = func(**tool_args)
        except Exception as e:
            tool_response = f"Ошибка выполнения: {str(e)}"
        # Добавляем ответ инструмента в историю
        history.append({
            "role": "tool",
            "tool_call_id": tool_call.id if hasattr(tool_call, "id") else None,
            "name": tool_name,
            "content": json.dumps(tool_response)
        })
        results.append(tool_response)

    # Получим итоговый ответ от LLM с дополненной историей
    bot_reply = ai_model.chat(history)
    history.append(bot_reply)
    user_chats[user_id] = history[-8:]  # max_message_history или сколько вам нужно

    # Возвращаем первый результат +, optionally, вложение для VK
    if len(results) > 1:
        reply_text = "\n".join(str(res) for res in results)
    else:
        reply_text = results[0]

    # Предполагаем, что для VK вложение — это url-картинка/идентификатор, если есть
    attachment = None
    if isinstance(results[0], dict) and "picture_url" in results[0]:
        attachment = results[0]["picture_url"]

    return reply_text, attachment
