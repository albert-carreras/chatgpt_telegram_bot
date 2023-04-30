"""Main message sending"""
import os
from datetime import datetime

import openai
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import FAISS

import ingest
import generate_prompt
import config

openai.api_key = config.openai_api_key
os.environ["OPENAI_API_KEY"] = config.openai_api_key

OPENAI_COMPLETION_OPTIONS = {
    "temperature": 0.5,
    "max_tokens": 1000,
    "top_p": 0.5,
    "frequency_penalty": 0.5,
    "presence_penalty": 0,
}

embeddings = OpenAIEmbeddings(chunk_size=1000)

class ChatGPT:
    def __init__(self, model="gpt-3.5-turbo"):
        assert model in {"text-davinci-003", "gpt-3.5-turbo", "gpt-4"}, f"Unknown model: {model}"
        self.model = model
        self.vector_db = ""
        ingest.ingest_docs(embeddings)

    async def send_message(self, message, dialog_messages=[], chat_mode="assistant"):
        if chat_mode not in config.chat_modes.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")
        self.vector_db = FAISS.load_local("faiss_index", embeddings)

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                if self.model in {"gpt-3.5-turbo", "gpt-4"}:
                    messages = generate_prompt.get_messages(dialog_messages, chat_mode)
                    r = await self._create_chain(messages=messages, message=message, model=self.model)
                    answer = r
                else:
                    raise ValueError(f"Unknown model: {self.model}")

            except openai.error.InvalidRequestError as error:
                if len(dialog_messages) == 0:
                    raise ValueError(
                        "Dialog messages is reduced to zero, but still has too many tokens to make completion") from error

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)

        return answer.strip(), n_first_dialog_messages_removed

    async def _create_chain(self, messages, message, model="gpt-4"):
        template = f"""
            Current Date and Time: {datetime.now().strftime("%d-%B-%Y at %H:%M")} \n\n
            The following is Data about Albert, use it to provide better answers about him: {{albert_data}} \n\n
            The following is this Conversation's History: {{conversation_history}} \n\n
            Last User Prompt: {message} \n
            Assistant Response: 
            """

        prompt_template = PromptTemplate(
            input_variables=["conversation_history", "albert_data"],
            template=template
        )

        llm_chain = LLMChain(
            verbose=True,
            llm=ChatOpenAI(model=model, verbose=True, **OPENAI_COMPLETION_OPTIONS),
            prompt=prompt_template,
        )

        docs_data = self.vector_db.similarity_search(message)
        answer = llm_chain.run(conversation_history=messages, albert_data=docs_data)

        return answer


async def transcribe_audio(audio_file):
    result = await openai.Audio.atranscribe("whisper-1", audio_file)
    return result["text"]


async def generate_images(prompt, n_images=4):
    r = await openai.Image.acreate(prompt=prompt, n=n_images, size="512x512")
    image_urls = [item.url for item in r.data]
    return image_urls


async def is_content_acceptable(prompt):
    result = await openai.Moderation.acreate(input=prompt)
    return not all(result.results[0].categories.values())
