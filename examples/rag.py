from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

texts = ["harrison went to kensho"]
embeddings = OpenAIEmbeddings()
db = FAISS.from_texts(texts, embeddings)

retriever = db.as_retriever()

from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    """Answer the question "{question}"  based on the following context: {context}"""
)

from langchain.schema.messages import AIMessage, AnyMessage, FunctionMessage

from permchain import Channel, Pregel
from permchain.channels import Topic

from langchain.chat_models import ChatOpenAI

model = ChatOpenAI()

chain = (
    Channel.subscribe_to(["question"])
    | {
        "context": (lambda x: x["question"])
        | Channel.write_to(
            messages=lambda _input: AIMessage(
                content="",
                additional_kwargs={
                    "function_call": "retrieval",
                    "arguments": {"question": _input},
                },
            )
        )
        | retriever
        | Channel.write_to(
            messages=lambda documents: FunctionMessage.construct(
                content=documents,  # function message requires content to be str
                name="retrieval",
            )
        ),
        "question": lambda x: x["question"],
    }
    | prompt
    | model
    | Channel.write_to(messages=lambda message: [message])
)

app = Pregel(
    chains={"chain": chain},
    channels={"messages": Topic(AnyMessage)},
    input=["question"],
    output=["messages"],
)

for s in app.stream({"question": "where did harrison go"}):
    print(s)
