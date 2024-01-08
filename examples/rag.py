from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.messages import AIMessage, AnyMessage, FunctionMessage
from langchain_core.prompts import PromptTemplate

from langgraph.channels import Topic
from langgraph.pregel import Channel, Pregel

texts = ["harrison went to kensho"]
embeddings = OpenAIEmbeddings()
db = FAISS.from_texts(texts, embeddings)

retriever = db.as_retriever()


prompt = PromptTemplate.from_template(
    """Answer the question "{question}"  based on the following context: {context}"""
)

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
