from langchain.schema import AIMessage, HumanMessage, SystemMessage

import config


def get_messages(dialog_messages, chat_mode):
    prompt = config.chat_modes[chat_mode]["prompt_start"]

    messages = [SystemMessage(content=prompt)]
    for dialog_message in dialog_messages:
        messages.append(HumanMessage(content=dialog_message["user"]))
        messages.append(AIMessage(content=dialog_message["bot"]))

    return messages
