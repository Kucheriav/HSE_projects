from ai import OpenRouterClient
from tools_functions import *
import json
import os

user_chats: dict[str, list[dict]] = {}
AI_API_KEY = os.getenv("OPENROUTER_API_KEY")


TOOL_MAPPING = {
    "wiki_search": wiki_search,
    "generate_picture": generate_picture,
    "get_content_info": get_content_info,
    'text_generation':
}

#TODO
tools_description = {}


def tool_choosing(text: str, user_id):
    system_prompt = 'You are helping to choose the proper function for the user text request'
    user_chats[user_id] = ai_model.chat(user_chats.get(user_id, None), query)
    ai_model = OpenRouterClient(AI_API_KEY,"qwen/qwen3.5-flash-02-23", system_prompt)
    tools = ai_model.choose_tool(user_message=text)
    for tool_call in tools:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        tool_response = TOOL_MAPPING[tool_name](**tool_args)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(tool_response),
        })


