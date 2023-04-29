"""Main message sending"""
import os
from datetime import datetime

import openai
import tiktoken
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader

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

embeddings = OpenAIEmbeddings(openai_api_key=config.openai_api_key, chunk_size=1000)

class ChatGPT:
    def __init__(self, model="gpt-3.5-turbo"):
        assert model in {"text-davinci-003", "gpt-3.5-turbo", "gpt-4"}, f"Unknown model: {model}"
        self.model = model
        self.vector_db = ""
        self._ingest_docs()

    async def send_message(self, message, dialog_messages=[], chat_mode="assistant"):
        if chat_mode not in config.chat_modes.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")
        self.vector_db = FAISS.load_local("faiss_index", embeddings)

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                if self.model in {"gpt-3.5-turbo", "gpt-4"}:
                    messages = self._generate_prompt_messages(dialog_messages, chat_mode)
                    r = await self._create_chain(messages=messages, message=message, model=self.model)
                    answer = r
                else:
                    raise ValueError(f"Unknown model: {self.model}")

                answer = self._postprocess_answer(answer)
    
            except openai.error.InvalidRequestError as error:
                if len(dialog_messages) == 0:
                    raise ValueError("Dialog messages is reduced to zero, but still has too many tokens to make completion") from error

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)

        return answer, n_first_dialog_messages_removed

    def _generate_prompt_messages(self, dialog_messages, chat_mode):
        prompt = config.chat_modes[chat_mode]["prompt_start"]

        messages = [SystemMessage(content=prompt)]
        for dialog_message in dialog_messages:
            messages.append(HumanMessage(content=dialog_message["user"]))
            messages.append(AIMessage(content=dialog_message["bot"]))

        return messages

    def _postprocess_answer(self, answer):
        answer = answer.strip()
        return answer

    def _count_tokens_from_messages(self, messages, answer, model="gpt-3.5-turbo"):
        encoding = tiktoken.encoding_for_model(model)

        if model == "gpt-3.5-turbo":
            tokens_per_message = 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif model == "gpt-4":
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise ValueError(f"Unknown model: {model}")

        # input
        n_input_tokens = 0
        for message in messages:
            n_input_tokens += tokens_per_message
            for key, value in message.items():
                n_input_tokens += len(encoding.encode(value))
                if key == "name":
                    n_input_tokens += tokens_per_name

        n_input_tokens += 2

        # output
        n_output_tokens = 1 + len(encoding.encode(answer))

        return n_input_tokens, n_output_tokens

    def _count_tokens_from_prompt(self, prompt, answer, model="text-davinci-003"):
        encoding = tiktoken.encoding_for_model(model)

        n_input_tokens = len(encoding.encode(prompt)) + 1
        n_output_tokens = len(encoding.encode(answer))

        return n_input_tokens, n_output_tokens

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

        print("\n\n\n\n QUESTION -------------------------------------- \n")
        print(message)
        print("\n ---------------------------------------------- \n\n\n\n")
        print("\n\n\n HISTORY -------------------------- \n")
        print(messages)
        print("\n ---------------------------------------------- \n\n\n\n")

        answer = llm_chain.run(conversation_history=messages, albert_data=docs_data)
        return answer

    def _ingest_docs(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(script_dir, 'data', 'Basic.txt')

        documents = TextLoader(file).load()

        texts = CharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap  = 0,
        ).split_documents(documents)

        FAISS.from_documents(texts, embeddings).save_local("faiss_index")
        return

async def transcribe_audio(audio_file):
    r = await openai.Audio.atranscribe("whisper-1", audio_file)
    return r["text"]


async def generate_images(prompt, n_images=4):
    r = await openai.Image.acreate(prompt=prompt, n=n_images, size="512x512")
    image_urls = [item.url for item in r.data]
    return image_urls

# is_content_acceptable
async def is_content_acceptable(prompt):
    r = await openai.Moderation.acreate(input=prompt)
    return not all(r.results[0].categories.values())
