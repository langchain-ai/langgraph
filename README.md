# GigaGraph

⚡ Разработка языковых агентов в виде графов ⚡

## Описание

GigaGraph — это библиотека, дающая возможность работать с LLM (большие языковые модели) для создания приложений, которые используют множество взаимодействующих цепочек (акторов) и сохраняют данные о состоянии.
Так как в основе GigaGraph лежит [GigaChain](https://github.com/ai-forever/gigachain), предполагается совместное использование обоих библиотек.

Основной сценарий использования GigaGraph — добавление циклов в приложения с LLM. Для этого библиотека добавляет в [LangChain Expression Language](https://python.langchain.com/docs/expression_language/) возможность работать с множеством цепочек на каждой из итераций вычислительного цикла.
Использование циклов позволяет реализовать поведение агента, когда приложению нужно многократно вызывать LLM и спрашивать, какое действие нужно выполнить следующим.

Следует отметить, что GigaGraph не предназначена для создания *DAG* (ориентированного ациклического графа).
Для решения этой задачи используйте стандартные возможности LangChain Expression Language.

## Установка

Для установки используйте менеджер пакетов pip:

```shell
pip install gigagraph
```

## Быстрый старт

Ниже приводится пример разработки агента, использующего несколько моделей и вызов функций.
Агент отображает каждое свое состояние в виде отдельных сообщений в списке

Для работы агента потребуется установить некоторые пакеты GigaChain и использовать в качестве демонстрации сервис [Tavily](https://app.tavily.com/sign-in):

State in LangGraph can be pretty general, but to keep things simpler to start, we'll show off an example where the graph's state is limited to a list of chat messages using the built-in `MessageGraph` class. This is convenient when using LangGraph with LangChain chat models because we can directly return chat model output.

First, install the GigaChain OpenAI integration package:

```python
pip install gigachain_openai
```

We also need to export some environment variables:

```shell
export OPENAI_API_KEY=sk-...
```

And now we're ready! The graph below contains a single node called `"oracle"` that executes a chat model, then returns the result:

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessageGraph

model = ChatOpenAI(temperature=0)

graph = MessageGraph()

graph.add_node("oracle", model)
graph.add_edge("oracle", END)

graph.set_entry_point("oracle")

runnable = graph.compile()
```

Let's run it!

```python
runnable.invoke(HumanMessage("What is 1 + 1?"))
```

```
[HumanMessage(content='What is 1 + 1?'), AIMessage(content='1 + 1 equals 2.')]
```

So what did we do here? Let's break it down step by step:

1. First, we initialize our model and a `MessageGraph`.
2. Next, we add a single node to the graph, called `"oracle"`, which simply calls the model with the given input.
3. We add an edge from this `"oracle"` node to the special string `END` (`"__end__"`). This means that execution will end after the current node.
4. We set `"oracle"` as the entrypoint to the graph.
5. We compile the graph, translating it to low-level [pregel operations](https://research.google/pubs/pregel-a-system-for-large-scale-graph-processing/) ensuring that it can be run.

Then, when we execute the graph:

1. LangGraph adds the input message to the internal state, then passes the state to the entrypoint node, `"oracle"`.
2. The `"oracle"` node executes, invoking the chat model.
3. The chat model returns an `AIMessage`. LangGraph adds this to the state.
4. Execution progresses to the special `END` value and outputs the final state.

And as a result, we get a list of two chat messages as output.

### Interaction with LCEL

As an aside for those already familiar with LangChain - `add_node` actually takes any function or [runnable](https://python.langchain.com/docs/expression_language/interface/) as input. In the above example, the model is used "as-is", but we could also have passed in a function:

```python
def call_oracle(messages: list):
    return model.invoke(messages)

graph.add_node("oracle", call_oracle)
```

Just make sure you are mindful of the fact that the input to the [runnable](https://python.langchain.com/docs/expression_language/interface/) is the **entire current state**. So this will fail:

```python
# This will not work with MessageGraph!
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant named {name} who always speaks in pirate dialect"),
    MessagesPlaceholder(variable_name="messages"),
])

chain = prompt | model

# State is a list of messages, but our chain expects a dict input:
#
# { "name": some_string, "messages": [] }
#
# Therefore, the graph will throw an exception when it executes here.
graph.add_node("oracle", chain)
```

## Conditional edges

Now, let's move onto something a little bit less trivial. LLMs struggle with math, so let's allow the LLM to conditionally call a `"multiply"` node using [tool calling](https://python.langchain.com/docs/modules/model_io/chat/function_calling/).

We'll recreate our graph with an additional `"multiply"` that will take the result of the most recent message, if it is a tool call, and calculate the result.
We'll also [bind](https://api.python.langchain.com/en/latest/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html#langchain_openai.chat_models.base.ChatOpenAI.bind_tools) the calculator's schema to the OpenAI model as a tool to allow the model to optionally use the tool necessary to respond to the current state:

```python
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

@tool
def multiply(first_number: int, second_number: int):
    """Multiplies two numbers together."""
    return first_number * second_number

model = ChatOpenAI(temperature=0)
model_with_tools = model.bind_tools([multiply])

builder = MessageGraph()

builder.add_node("oracle", model_with_tools)

tool_node = ToolNode([multiply])
builder.add_node("multiply", tool_node)

builder.add_edge("multiply", END)

builder.set_entry_point("oracle")
```

Now let's think - what do we want to have happened?

- If the `"oracle"` node returns a message expecting a tool call, we want to execute the `"multiply"` node
- If not, we can just end execution

We can achieve this using **conditional edges**, which call a function on the current state and routes execution to a node the function's output.

Here's what that looks like:

```python
from typing import Literal

def router(state: List[BaseMessage]) -> Literal["multiply", "__end__"]:
    tool_calls = state[-1].additional_kwargs.get("tool_calls", [])
    if len(tool_calls):
        return "multiply"
    else:
        return "__end__"

builder.add_conditional_edges("oracle", router)
```

If the model output contains a tool call, we move to the `"multiply"` node. Otherwise, we end execution.

Great! Now all that's left is to compile the graph and try it out. Math-related questions are routed to the calculator tool:

```python
runnable = builder.compile()

runnable.invoke(HumanMessage("What is 123 * 456?"))
```

```

[HumanMessage(content='What is 123 * 456?'),
 AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_OPbdlm8Ih1mNOObGf3tMcNgb', 'function': {'arguments': '{"first_number":123,"second_number":456}', 'name': 'multiply'}, 'type': 'function'}]}),
 ToolMessage(content='56088', tool_call_id='call_OPbdlm8Ih1mNOObGf3tMcNgb')]
```

While conversational responses are outputted directly:

```python
runnable.invoke(HumanMessage("What is your name?"))
```

```
[HumanMessage(content='What is your name?'),
 AIMessage(content='My name is Assistant. How can I assist you today?')]
```

## Cycles

Now, let's go over a more general cyclic example. We will recreate the `AgentExecutor` class from LangChain. The agent itself will use chat models and tool calling.
This agent will represent all its state as a list of messages.

We will need to install some GigaChain packages, as well as [Tavily](https://app.tavily.com/sign-in) to use as an example tool.

```shell
pip install -U gigachain gigachain_openai tavily-python
```

Также для доступа к OpenAI и Tavily API понадобится задать переменные среды:

```shell
export OPENAI_API_KEY=sk-...
export TAVILY_API_KEY=tvly-...
```

При желании вы можете использовать [LangSmith](https://docs.smith.langchain.com/):

```shell
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY=ls__...
```

### Подготовьте инструменты

В первую очередь определите инструменты (`tools`), которые будет использовать приложение.
В качестве примера в этом разделе используется поиск, встроенный в Tavily, но вы также можете использовать собственные инструменты.
Подробнее об том как создавать свои инструменты — в [документации](https://python.langchain.com/docs/modules/agents/tools/custom_tools).


```python
from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=1)]
```

Оберните инструменты в GigaGraph `ToolExecutor` — класс, который принимает объекты запуска инструмента `ToolInvocation`, вызывает инструмент и возвращает ответ.
Объект `ToolInvocation` — произвольный класс с атрибутами `tool` и `tool_input`.

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)
```

### Задайте модель

Подключите модель, которую будет использовать приложение.
Для демонстрации в описываемом примере модель должна:

* поддерживать списки сообщений. Каждое свое состояние агент будет возвращать в виде сообщений, поэтому модель должна хорошо работать со списками сообщений.
* предоставлять интерфейсы вызова функций, аналогичные моделям OpenAI.


```python
from langchain_openai import ChatOpenAI
from langgraph.checkpoint import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

# Параметр streaming=True включает потоковую передачу токенов
# Подробнее в разделе Потоковая передача.
model = ChatOpenAI(temperature=0, streaming=True)
```

После подключения убедитесь, что модель знает, какие инструменты доступны ей.
Для этого преобразуйте инструменты GigaGraph в формат OpenAI-функций и привяжите их к классу модели.

```python
model = model.bind_tools(tools)
```

### Определите состояние агента

Основным графом `gigagraph` является `StatefulGraph`.
Этот граф параметризован объектом состояния, который он передает каждой вершине.
В свою очередь каждая вершина возвращает операции для обновления состояния.
Операции могут либо задавать (SET) определенные атрибуты состояния (например, переписывать существующие значения), либо добавлять ()ADD данные к существующим атрибутам.
Будет операция задавать или добавлять данные, определяется аннотациями объекта состояния, который используется для создания графа.

В приведенном примере отслеживаемое состояние представлено в виде списка сообщений.
Поэтому нужно чтобы каждая вершина добавляла сообщения в список.

Для этого используйте `TypedDict` с одним ключом (`messages`) и аннотацией, указывающей на то, что в атрибут `messages` можно только добавлять данные.

```python
from typing import TypedDict, Annotated

def add_messages(left: list, right: list):
    """Add-don't-overwrite."""
    return left + right

class AgentState(TypedDict):
    # The `add_messages` function within the annotation defines
    # *how* updates should be merged into the state.
    messages: Annotated[list, add_messages]
```

### Определите вершины графа

Теперь нужно определить несколько разных вершин графа.
В `langgraph` вершина может быть представлена в виде функции или [исполняемого интерфейса](https://python.langchain.com/docs/expression_language/).
Для описываемого примера понадобятся две основных вершины:

* Агент, который принимает решения когда и какие действия нужно выполнять.
* Функция для вызова инструментов. Если агент решает совершить действие, эта вершина его выполнит.

Также нужно определить ребра графа.
Часть ребер могут зависеть от условий (*условные ребра*).
Это связанно с тем, что в зависимости от вывода вершины могут быть реализованы различные пути развития событий.
При этом неизвестно какой путь будет выбран до момента обращения к вершине.
Какой путь выбрать LLM решает самостоятельно.

Разница между обычным и условным ребром графа:

* В случае условного ребра, после вызова агента:

  * если агент решает предпринять действие, нужно вызвать функцию для обращения к инструментам;
  * если агент решает, что действие завершено, операции должны быть прекращены.

* В случае обычного ребра после обращения к инструментам, нужно всегда возвращаться к агенту, чтобы он определил дальнейшие действия.


Определите вершины и функцию, которая будет решать какое из условных ребер выполнять.

model = ChatOpenAI(temperature=0).bind_tools(tools)

# Задайте функцию, которая определяет нужно продолжать или нет
def should_continue(state):
    messages = state['messages']
    last_message = messages[-1]
    # Приложение останавливается, если нет вызова функции
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    # В противном случае выполнение продолжается
    else:
        return "continue"

# Задайте функцию, которая будет обращаться к модели
def call_model(state):
    messages = state['messages']
    response = model.invoke(messages)
    # Возвращается список, который будет добавлен к существующему списку сообщений
    return {"messages": [response]}

# Задайте функцию, которая будет вызывать инструменты
def call_tool(state):
    messages = state['messages']
    # Благодаря условию continue
    # приложение знает, что последнее сообщение содержит вызов функции
    last_message = messages[-1]
    # Создание ToolInvocation из function_call
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(last_message.additional_kwargs["function_call"]["arguments"]),
    )
    # Вызов tool_executor и получение ответа
    response = tool_executor.invoke(action)
    # Использование ответа для создания сообщения FunctionMessage
    function_message = FunctionMessage(content=str(response), name=action.tool)
    # Возвращение списка, который будет добавлен к существующему списку сообщений
    return {"messages": [function_message]}
```

### Определите граф

```python
from langgraph.graph import StateGraph, END
# Задайте новый граф
workflow = StateGraph(AgentState)

# Задайте две вершины, которые будут работать в цикле
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Задайте точку входа `agent`
# Точка входа указывает вершину, котора будет вызвана в первую очередь
workflow.set_entry_point("agent")

# Создайте условное ребро
workflow.add_conditional_edges(
    # Определите начальную вершину. В этом примере используется вершина `agent`.
    # Это задает ребра, которые будут использованы после вызова вершины `agent`.
    "agent",
    # Передайте функцию, которая определяет какую вершину вызвать дальше.
    should_continue,
    # Передайте структуру (map), в которой ключами будут строки, а значениями другие вершины.
    # END — зарезервированная вершна, указываящая на то, что граф должен завершиться.
    # После вызова `should_continue` вывод функции сравнивается с ключами в структуре.
    # После чего вызывается соотвествующая выводу вершина.
    {
        # If `tools`, then we call the tool node.
        # Если значение `tools`, вызывается вершина, ответственная за обращение к интрсументам.
        "continue": "action",
        # В противном случае граф заканчивается.
        "end": END
    }
)

# Добавьте обычное ребро, соединяющее вершины `tools` и `agent`.
# Ребро задает путь при котором после вызова вершины `tools`, вызывается вершина `agent`.
workflow.add_edge('action', 'agent')

# Скомпилируйте все предыдущие этапы в исполняемый интерфейс GigaChain.
# Теперь граф можно использовать также, как и другие исполняемые интерфейсы.
app = workflow.compile()
```

### Использование

Скомпилированный исполняемый интерфейс принимает на вход список сообщений:

```python
from langchain_core.messages import HumanMessage

inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}
app.invoke(inputs)
```

Работа интерфейса занимает некоторое время.
Чтобы наблюдать за результатом работы в прямом эфире, вы можете включить потоковую передачу.

## Потоковая передача

GigaGraph поддерживает несколько разных способов потоковой передачи.

### Потоковая передача вывода вершины

GigaGraph предоставляет возможность потоковой передачи результата вызова каждой из вершин графа по мере обращения к ним.

```python
inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}
for output in app.stream(inputs):
    # stream() возвращает словари с парами `Вершина графа — вывод`
    for key, value in output.items():
        print(f"Output from node '{key}':")
        print("---")
        print(value)
    print("\n---\n")
```

```
'The current weather in San Francisco is as follows:\n- Temperature: 60.1°F (15.6°C)\n- Condition: Partly cloudy\n- Wind: 5.6 mph (9.0 kph) from SSW\n- Humidity: 83%\n- Visibility: 9.0 miles (16.0 km)\n- UV Index: 4.0\n\nFor more details, you can visit [Weather API](https://www.weatherapi.com/).'
```

### Потоковая передача токенов модели

Библиотека дает доступ к потоковой передачи токенов модели по мере их возникновения на каждой из вершин.
В приведенном примере только вершина `agent` может возвращать токены модели.
Для работы этой функциональность нужно чтобы модель поддерживала работу в режиме потоковой передачи токенов.

```python
final_state = app.invoke(
    {"messages": [HumanMessage(content="what about ny")]},
    config={"configurable": {"thread_id": 42}}
)
final_state["messages"][-1].content
```

```
'The current weather in New York is as follows:\n- Temperature: 20.3°C (68.5°F)\n- Condition: Overcast\n- Wind: 2.2 mph from the north\n- Humidity: 65%\n- Cloud Cover: 100%\n- UV Index: 5.0\n\nFor more details, you can visit [Weather API](https://www.weatherapi.com/).'
```

## Область применения

Используйте библиотеку если вам нужна поддержка циклов.

Если обычной работы с цепочками для решения ваших задач достаточно, используйте основные возможности [LangChain Expression Language](https://python.langchain.com/docs/expression_language/).

## How-to Guides

These guides show how to use LangGraph in particular ways.

### Async

If you are running LangGraph in async workflows, you may want to create the nodes to be async by default.
For a walkthrough on how to do that, see [this documentation](https://github.com/langchain-ai/langgraph/blob/main/examples/async.ipynb)

### Streaming Tokens

Sometimes language models take a while to respond and you may want to stream tokens to end users.
For a guide on how to do this, see [this documentation](https://github.com/langchain-ai/langgraph/blob/main/examples/streaming-tokens.ipynb)

### Persistence

LangGraph comes with built-in persistence, allowing you to save the state of the graph at point and resume from there.
For a walkthrough on how to do that, see [this documentation](https://github.com/langchain-ai/langgraph/blob/main/examples/persistence.ipynb)

### Human-in-the-loop

LangGraph comes with built-in support for human-in-the-loop workflows. This is useful when you want to have a human review the current state before proceeding to a particular node.
For a walkthrough on how to do that, see [this documentation](https://github.com/langchain-ai/langgraph/blob/main/examples/human-in-the-loop.ipynb)

### Visualizing the graph

Agents you create with LangGraph can be complex. In order to make it easier to understand what is happening under the hood, we've added methods to print out and visualize the graph.
This can create both ascii art and pngs.
For a walkthrough on how to do that, see [this documentation](https://github.com/langchain-ai/langgraph/blob/main/examples/visualization.ipynb)

### "Time Travel"

With "time travel" functionality you can jump to any point in the graph execution, modify the state, and rerun from there.
This is useful for both debugging workflows, as well as end user-facing workflows to allow them to correct the state.
For a walkthrough on how to do that, see [this documentation](https://github.com/langchain-ai/langgraph/blob/main/examples/time-travel.ipynb)


## Примеры

### ChatAgentExecutor: with function calling

### Исполнитель чат-агента с возможностью вызывать функции

Пример приложения-исполнителя принимает на вход список сообщений и также возвращает список сообщений на выходе.
Состояние агента также представлено в виде списка сообщений.
Представленный пример использует вызов функций OpenAI.


- [Getting Started Notebook](https://github.com/langchain-ai/langgraph/blob/main/examples/chat_agent_executor_with_function_calling/base.ipynb). Базовый пример, демонстрирующий пошаговое создание приложения исполнителя агентов.
- [High Level Entrypoint](https://github.com/langchain-ai/langgraph/blob/main/examples/chat_agent_executor_with_function_calling/high-level.ipynb). Пример демонстрирует как можно использовать высокоуровневую точку входа для исполнителя чат-агента.

**Вариации примеров**

We also have a lot of examples highlighting how to slightly modify the base chat agent executor. These all build off the [getting started notebook](https://github.com/langchain-ai/langgraph/blob/main/examples/chat_agent_executor_with_function_calling/base.ipynb) so it is recommended you start with that first.

- [Human-in-the-loop](https://github.com/langchain-ai/langgraph/blob/main/examples/chat_agent_executor_with_function_calling/human-in-the-loop.ipynb). Пример демонстрирует как реализовать подход «человек-в-цикле».
- [Принудительный вызов инструмента](https://github.com/langchain-ai/langgraph/blob/main/examples/chat_agent_executor_with_function_calling/force-calling-a-tool-first.ipynb). Пример демонстрирует как всегда вызывать определенный инструмент в первую очередь.
- [Ответ в заданном формате](https://github.com/langchain-ai/langgraph/blob/main/examples/chat_agent_executor_with_function_calling/respond-in-format.ipynb). Пример демонстрирует, как принудительно получить ответ агента в заданном формате.
- [Динамический вывод результата использования инструмента](https://github.com/langchain-ai/langgraph/blob/main/examples/chat_agent_executor_with_function_calling/dynamically-returning-directly.ipynb). Пример демонстрирует, как агент может самостоятельно решать возвращать результат использования инструмента пользователю или нет.
- [Управление этапами работы агента](https://github.com/langchain-ai/langgraph/blob/main/examples/chat_agent_executor_with_function_calling/managing-agent-steps.ipynb). Пример демонстрирует, как можно более детально управлять промежуточными этапами работы агента.

### Исполнители агентов

Примеры приложений-исполнителей, использующих агенты LangChain.

- [Getting Started Notebook](https://github.com/langchain-ai/langgraph/blob/main/examples/agent_executor/base.ipynb). Базовый пример, демонстрирующий пошаговое создание приложения исполнителя агентов.
- [High Level Entrypoint](https://github.com/langchain-ai/langgraph/blob/main/examples/agent_executor/high-level.ipynb). Пример демонстрирует как можно использовать высокоуровневую точку входа для исполнителя чат-агента.

**Вариации примеров**

Примеры небольших изменений, которые можно сделать при разработке исполнителя чат-агента.
Приведенные вариации основаны на примере [Getting Started Notebook](https://github.com/langchain-ai/langgraph/blob/main/examples/agent_executor/base.ipynb).

- [Human-in-the-loop](https://github.com/langchain-ai/langgraph/blob/main/examples/agent_executor/human-in-the-loop.ipynb). Пример демонстрирует как реализовать подход «человек-в-цикле».
- [Принудительный вызов инструмента](https://github.com/langchain-ai/langgraph/blob/main/examples/agent_executor/force-calling-a-tool-first.ipynb). Пример демонстрирует как всегда вызывать определенный инструмент в первую очередь.
- [Управление этапами работы агента](https://github.com/langchain-ai/langgraph/blob/main/examples/agent_executor/managing-agent-steps.ipynb). Пример демонстрирует, как можно более детально управлять промежуточными этапами работы агента.

### Planning Agent Examples

The following notebooks implement agent architectures prototypical of the "plan-and-execute" style, where an LLM planner decomposes a user request into a program, an executor executes the program, and an LLM synthesizes a response (and/or dynamically replans) based on the program outputs.

- [Plan-and-execute](https://github.com/langchain-ai/langgraph/blob/main/examples/plan-and-execute/plan-and-execute.ipynb): a simple agent with a **planner** that generates a multi-step task list, an **executor** that invokes the tools in the plan, and a **replanner** that responds or generates an updated plan. Based on the [Plan-and-solve](https://arxiv.org/abs/2305.04091) paper by Wang, et. al.
- [Reasoning without Observation](https://github.com/langchain-ai/langgraph/blob/main/examples/rewoo/rewoo.ipynb): planner generates a task list whose observations are saved as **variables**. Variables can be used in subsequent tasks to reduce the need for further re-planning. Based on the [ReWOO](https://arxiv.org/abs/2305.18323) paper by Xu, et. al.
- [LLMCompiler](https://github.com/langchain-ai/langgraph/blob/main/examples/llm-compiler/LLMCompiler.ipynb): planner generates a **DAG** of tasks with variable responses. Tasks are **streamed** and executed eagerly to minimize tool execution runtime. Based on the [paper](https://arxiv.org/abs/2312.04511) by Kim, et. al.

### Reflection / Self-Critique

When output quality is a major concern, it's common to incorporate some combination of self-critique or reflection and external validation to refine your system's outputs. The following examples demonstrate research that implement this type of design.

- [Basic Reflection](./examples/reflection/reflection.ipynb): add a simple "reflect" step in your graph to prompt your system to revise its outputs.
- [Reflexion](./examples/reflexion/reflexion.ipynb): critique missing and superflous aspects of the agent's response to guide subsequent steps. Based on [Reflexion](https://arxiv.org/abs/2303.11366), by Shinn, et. al.
- [Giga Reflexion](./examples/reflexion_giga/reflexion.ipynb): реализация Reflexion на GigaChat
- [Language Agent Tree Search](./examples/lats/lats.ipynb): execute multiple agents in parallel, using reflection and environmental rewards to drive a Monte Carlo Tree Search. Based on [LATS](https://arxiv.org/abs/2310.04406/LanguageAgentTreeSearch/), by Zhou, et. al.

### Multi-agent Examples
### Примеры с несколькими агентами

- [Совместная работа нескольких агентов](https://github.com/langchain-ai/langgraph/blob/main/examples/multi_agent/multi-agent-collaboration.ipynb). Пример демонстрирует как создать двух агентов, которые работают вместе для решения задачи.
- [Несколько агентов с «руководителем»](https://github.com/langchain-ai/langgraph/blob/main/examples/multi_agent/agent_supervisor.ipynb). Пример демонстрирует как организовать работу агентов используя LLM в роли «руководителя», который решает как распределять работу.
- [Иерархичные команды агентов](https://github.com/langchain-ai/langgraph/blob/main/examples/multi_agent/hierarchical_agent_teams.ipynb): пример демонстрирует как организовать «команды» агентов, которые будут взаимодействовать для решения задачи, в виде вложенных графов.

### Симуляция для оценки чат-бота

Оценка работы чат-бота в многоэтапных сценариях может вызывать трудности. Для решения таких задач вы можете использовать симуляции.

- [Оценка чат-бота с помощью симуляции взаимодействия нескольких агентов.](https://github.com/langchain-ai/langgraph/blob/main/examples/chatbot-simulation-evaluation/agent-simulation-evaluation.ipynb). В примере показано как симулировать диалог «виртуального пользователя» с чат-ботом.

### Асинхронная работа

При работе с асинхронными процессами вам может потребоваться создать с помощью GigaGraph граф с вершинами, которые будут асинхронными по умолчанию.
[Пример](https://github.com/langchain-ai/langgraph/blob/main/examples/async.ipynb).

- [Chat bot evaluation as multi-agent simulation](https://github.com/langchain-ai/langgraph/blob/main/examples/chatbot-simulation-evaluation/agent-simulation-evaluation.ipynb): how to simulate a dialogue between a "virtual user" and your chat bot
- [Evaluating over a dataset](./examples/chatbot-simulation-evaluation/langsmith-agent-simulation-evaluation.ipynb): benchmark your assistant over a LangSmith dataset, which tasks a simulated customer to red-team your chat bot.

Ответ модели может занимать продолжительное время и вам может потребоваться на лету отображать пользователям результат работы модели.
[Пример](https://github.com/langchain-ai/langgraph/blob/main/examples/streaming-tokens.ipynb).

### Устойчивость

GigaGraph позволяет сохранять состояние графа в определенный момент времени и потом возобновлять работу с этого состояния.
[Пример](https://github.com/langchain-ai/langgraph/blob/main/examples/persistence.ipynb).

### Человек-в-цикле

GigaGraph поддерживает процесс, при котором необходимо участие человека, проверяющего текущее состояние графа перед переходом к следующей вершине.
Пример такого подхода — в [документации](https://github.com/langchain-ai/langgraph/blob/main/examples/human-in-the-loop.ipynb).

## Справка

GigaGraph предоставляет доступ к нескольким новым интерфейсам.

### StateGraph

Основная точка входа — класс `StateGraph`.

```python
from langgraph.graph import StateGraph
```

Класс ответственный за создание графа.
Этот граф параметризован объектом состояния, который он передает каждой вершине.

#### `__init__`

```python
    def __init__(self, schema: Type[Any]) -> None:
```

При создании графа нужно передать схему состояния.
Каждая вершина будет возращать операции для обновления этого состояния.
Операции могут либо задавать (SET) определенные атрибуты состояния (например, переписывать существующие значения), либо добавлять ()ADD данные к существующим атрибутам.
Будет операция задавать или добавлять данные, определяется аннотациями объекта состояния, который используется для создания графа.

Схему состояния рекомендуется задавать с помощью типизированного словаря: `from typing import TypedDict`

После создания схемы вы можете аннотировать атрибуты с помощью `from typing imoport Annotated`.
Сейчас поддерживается только одна аннотация — `import operator; operator.add`.
Аннотация указывает, что каждая вершина, которая возвращает этот атрибут добавляет новые данные к существующему значению.

Пример состояния:

## Documentation

class AgentState(TypedDict):
   # Входная строка
   input: str
   # The outcome of a given call to the agent
   # Needs `None` as a valid type, since this is what this will start as
   # Результат вызова агента
   # Должен принимать `None` в качестве валидного типа, так как это начальное значение
   agent_outcome: Union[AgentAction, AgentFinish, None]
   # Список действий и соответвтующих шагов
   # Аннотация `operator.add` указывает что состояние должно дополняться (ADD) новыми данными,
   # а не перезаписываться
   intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

```

Пример использования:

```python
# Инициализируйте StateGraph с помощью состояния AgentState
graph = StateGraph(AgentState)
# Создайте вершины и ребра
...
# Скомпилируйте граф
app = graph.compile()

# На вход должен передаваться словарь, так как состояние создано как TypedDict
inputs = {
   # Пример входный данныъ
   "input": "hi"
   # Предположим, что `agent_outcome` задается графом как некоторая точка
   # Передавать значение не нужно, по умолчанию оно будет None
   # Предположим, что граф со временем наполняет `intermediate_steps`
   # Передавать значение не нужно, по умолчанию список будет пустым
   # Список `intermediate_steps` будет представлен в виде пустого списка, а не None потому,
   # что он аннотирован с помощью `operator.add`
}
```

#### `.add_node`

```python
    def add_node(self, key: str, action: RunnableLike) -> None:
```

This method adds a node to the graph.
Добавляет вершину графа.
Принимает два параметра:

* `key` — Уникальная строка с названием вершины.
* `action` — действие, которое выполняется при вызове вершины. Выражается в виде функции или исполняемого интерфейса.

#### `.add_edge`

```python
    def add_edge(self, start_key: str, end_key: str) -> None:
```

Создает ребро графа, соединяющее начальную и конечную вершины.
Вывод начальной вершины передается в конечную.
Принимает два параметра:

- `start_key` — строка с названием начальной вершины. Название вершины должно быть зарегистрировано в графе.
- `end_key` — строка с названием конечной вершины. Название вершины должно быть зарегистрировано в графе.

#### `.add_conditional_edges`

```python
    def add_conditional_edges(
        self,
        start_key: str,
        condition: Callable[..., str],
        conditional_edge_mapping: Dict[str, str],
    ) -> None:
```

Создает условное ребро.
Позволяет задавать пути развития событий в зависимости от результата вызова начальной вершины.
Принимает три параметра:

- `start_key` — строка с названием начальной вершины. Название вершины должно быть зарегистрировано в графе.
- `condition` — функция, которая вызывается для определения пути развития событий. На вход принимает результат вызова начальной вершины. Возвращает строку, зарегистрированную в структуре `conditional_edge_mapping`, которая указывает в соответствии с каким ребром будут развиваться события.
- `conditional_edge_mapping` — структура (map) строка-строка. В качестве ключа задается название ребра, которое может вернуть `condition`. В качестве значения задается вершина, которые будет вызваны если `condition` вернет соответствующее название ребра.

#### `.set_entry_point`

```python
    def set_entry_point(self, key: str) -> None:
```
Точка входа в граф.
Задает вершину, которая будет вызвана в самом начале.
Принимает один параметр:

- `key` — название вершины, которую нужно вызывать в первую очередь.

#### `.set_conditional_entry_point`

```python
    def set_conditional_entry_point(
        self,
        condition: Callable[..., str],
        conditional_edge_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
```

This method adds a conditional entry point.
What this means is that when the graph is called, it will call the `condition` Callable to decide what node to enter into first.

- `condition`: A function to call to decide what to do next. The input will be the input to the graph. It should return a string that is present in `conditional_edge_mapping` and represents the edge to take.
- `conditional_edge_mapping`: A mapping of string to string. The keys should be strings that may be returned by `condition`. The values should be the downstream node to call if that condition is returned.

#### `.set_finish_point`

```python
    def set_finish_point(self, key: str) -> None:
```

This is the exit point of the graph.
When this node is called, the results will be the final result from the graph.
It only has one argument:

Точка выхода из графа.
При вызове заданной вершины, результат ее работы будет итоговым для графа.
Принимает один параметр:

- `key` — название вершины, результат вызова который будет считаться итоговым результатом работы графа.

Вершину не нужно вызывать если на предыдущих шагах графа было создано ребро (условное или обычное) ведущее к зарезервированной вершине `END`.

### Graph

```python
from langgraph.graph import Graph

graph = Graph()
```

Класс предоставляет доступ к интерфейсу `StateGraph`, но отличается тем, что объект состояния не обновляется со временем, а класс передает все состояние целиком на каждом этапе.
Это означает, что данные, которые возвращаются в результате работы одной вершины, передаются на вход при вызове другой вершины в исходном состоянии.

### `END`

```python
from langgraph.graph import END
```

This is a special node representing the end of the graph.
This means that anything passed to this node will be the final output of the graph.
It can be used in two places:

Зарезервированная вершина указывающая на завершение работы графа.
Все данные, которые передаются вершине при вызове будут считаться результатом работы графа.
Вершину можно использовать в двух случая:

- В качестве ключа `end_key` в `add_edge`.
- В качестве значения в структуре `conditional_edge_mapping`, передаваемой `add_conditional_edges`.

## Готовые примеры

Представленные примеры содержат несколько методов, облегчающих работу с распространенными, готовыми графами и компонентами.

### ToolExecutor

```python
from langgraph.prebuilt import ToolExecutor
```

Вспомогательный класс для вызова инструментов.
В качестве параметров класс принимает список инструментов.

```python
tools = [...]
tool_executor = ToolExecutor(tools)
```

После инициализации класс дает доступ к [исполняемому интерфейсу](https://python.langchain.com/docs/expression_language/interface).
Используйте класс для вызова инструментов. Передайте [AgentAction](https://python.langchain.com/docs/modules/agents/concepts#agentaction) для автоматического определения подходящего инструмента и входных данных.

### chat_agent_executor.create_function_calling_executor

```python
from langgraph.prebuilt import chat_agent_executor
```

Вспомогательная функция для создания графа, который работает с генеративной моделью и может вызывать функции.
Для использования функции передайте на вход модель и список инструментов.
Модель должна поддерживать интерфейс вызова функций аналогичный OpenAI.

```python
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import chat_agent_executor
from langchain_core.messages import HumanMessage

tools = [TavilySearchResults(max_results=1)]
model = ChatOpenAI()

app = chat_agent_executor.create_function_calling_executor(model, tools)

inputs = {"messages": [HumanMessage(content="какая погода в саратове")]}
for s in app.stream(inputs):
    print(list(s.values())[0])
    print("----")
```

### chat_agent_executor.create_tool_calling_executor

```python
from langgraph.prebuilt import chat_agent_executor
```

This is a helper function for creating a graph that works with a chat model that utilizes tool calling.
Can be created by passing in a model and a list of tools.
The model must be one that supports OpenAI tool calling.

```python
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import chat_agent_executor
from langchain_core.messages import HumanMessage

tools = [TavilySearchResults(max_results=1)]
model = ChatOpenAI()

app = chat_agent_executor.create_tool_calling_executor(model, tools)

inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}
for s in app.stream(inputs):
    print(list(s.values())[0])
    print("----")
```

### create_agent_executor

```python
from langgraph.prebuilt import create_agent_executor
```

Вспомогательная функция для работы с [агентами LangChain](https://python.langchain.com/docs/modules/agents/).
Для использования функции передайте на вход агента и список инструментов.

```python
from langgraph.prebuilt import create_agent_executor
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=1)]

# Подключите шаблон промпта. Вы можете выбрать любой шаблон
prompt = hub.pull("hwchase17/openai-functions-agent")

# Выберите модель, с которой будет работать агент
llm = ChatOpenAI(model="gpt-3.5-turbo-1106")

# Создайте агента OpenAI Functions
agent_runnable = create_openai_functions_agent(llm, tools, prompt)

app = create_agent_executor(agent_runnable, tools)

inputs = {"input": "what is the weather in sf", "chat_history": []}
for s in app.stream(inputs):
    print(list(s.values())[0])
    print("----")
```
