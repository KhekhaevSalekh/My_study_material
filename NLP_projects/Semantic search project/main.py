import os
import pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from pdf_loader import PDFLoader
from chat_with_local_llm import Chat

load_dotenv()

openai_base = os.environ.get("OPENAI_BASE")
openai_model = os.environ.get("OPENAI_MODEL")
pinecone_key = os.environ.get("PINECONE_API_KEY")
pinecone_env = os.environ.get("PINECONE_ENVIRONMENT")
INDEX_NAME = os.environ.get("INDEX_NAME")
NAMESPACE = os.environ.get("NAMESPACE")
SYSTEM_PROMPT = 'You are a helpful Q/A bot that can only use reference material from a knowledge base.If a user asks anything that is not "from the knowledge base", say that you cannot answer.'.strip()

model_for_embeddings = SentenceTransformer(
    "sentence-transformers/multi-qa-mpnet-base-cos-v1"
)

if pinecone_key:
    pinecone.init(api_key=pinecone_key, environment=pinecone_env)

    if not INDEX_NAME in pinecone.list_indexes():
        pinecone.create_index(
            INDEX_NAME,
            dimension=768,
            metric="cosine",
            pod_type="p1",
        )
        print("Pinecone index created")
    else:
        print("Pinecone index already exists")

    # Store the index as a variable
    index = pinecone.Index(INDEX_NAME)

pdf_loader = PDFLoader(
    "red-cap.pdf",
    model=model_for_embeddings,
    index=index,
    namespace=NAMESPACE,
    max_tokens=20,
    show_progress=True,
)
pdf_loader.load_to_pinecone()

chat = Chat(
    system_prompt=SYSTEM_PROMPT,
    index=index,
    engine=model_for_embeddings,
    openai_model=openai_model,
    openai_base=openai_base,
    namespace=NAMESPACE,
)

user_input = input("Введите вопрос: ")
while user_input != "exit":
    print(chat.user_turn(user_input))
    user_input = input("Введите вопрос: ")

print("Вывожу весь диалог:")
chat.display_conversation()
