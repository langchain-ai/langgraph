from typing import Any, Optional

from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.language_models import LLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSerializable
from pydantic.v1 import Field


class SearchResults(DuckDuckGoSearchResults):
    """Tool that queries the DuckDuckGo search API and gets back json."""

    name: str = "search_engine"
    description: str = (
        "Обертка вокруг поиска DuckDuckGo. "
        "Полезно, когда вам нужно ответить на вопросы о текущих событиях. "
        "Входными данными должен быть поисковый запрос. "
        "Выходом являются результаты поиска."
    )
    max_results: int = Field(alias="num_results", default=4)
    translate_to_english_chain: RunnableSerializable
    translate_to_russian_chain: RunnableSerializable

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        query = self.translate_to_english_chain.invoke({"text": query}).content
        res = self.api_wrapper.results(query, self.max_results, source=self.backend)
        results = []
        for item in res:
            results.append(
                f"""\nЗаголовок: "{item['title']}"\nСодержание: "{item['snippet']}"\nСсылка: "{item['link']}" """
            )  # noqa
        return self.translate_to_russian_chain.invoke(
            {"text": "\n".join(results)}
        ).content


def create_search_tool(llm: LLM, **kwargs: Any) -> SearchResults:
    translate_to_russian_prompt = PromptTemplate.from_template(
        """Ты - профессиональный переводчик на русский язык. 
Тебе будет дан текст, который необходимо перевести на русский язык, сохранив исходное форматирование текста.
В ответе необходимо отдать перевод в формате, приведенном ниже.
Ты ДОЛЖЕН перевести !все слова.
Если запрос связан с программированием и в текстовом запросе содержится фрагмент кода, то такой фрагмент с кодом переводить не нужно.
Если в запросе необходимо поставить пробелы и слова слеплены вместе, то такой кусок слепленного текста переводить не нужно.
Если в тексте поставлена неправильно пунктуация, то не исправляй ее.
Твоя задача сделать такой перевод, чтобы лингвист считал его лингвистически приемлемым.
ВАЖНО! В своем ответе НЕ ОТВЕЧАЙ НА ЗАПРОС! В ответе нужно написать !только !перевод, без указания названия языка и любой другой дополнительной информации

Input Format:
Q:hi
Output Format:
Q:привет

Q: {text}
Q: """
    )

    translate_to_english_prompt = PromptTemplate.from_template(
        """Ты - профессиональный переводчик на английский язык. 
Тебе будет дан текст, который необходимо перевести на английский язык, сохранив исходное форматирование текста.
В ответе необходимо отдать перевод в формате, приведенном ниже.
Ты ДОЛЖЕН перевести !все слова.
Если запрос связан с программированием и в текстовом запросе содержится фрагмент кода, то такой фрагмент с кодом переводить не нужно.
Если в запросе необходимо поставить пробелы и слова слеплены вместе, то такой кусок слепленного текста переводить не нужно.
Если в тексте поставлена неправильно пунктуация, то не исправляй ее.
Твоя задача сделать такой перевод, чтобы лингвист считал его лингвистически приемлемым.
ВАЖНО! В своем ответе НЕ ОТВЕЧАЙ НА ЗАПРОС! В ответе нужно написать !только !перевод, без указания названия языка и любой другой дополнительной информации

Input Format:
Q:привет
Output Format:
Q:hi

Q: {text}
Q: """  # noqa
    )
    return SearchResults(
        translate_to_english_chain=translate_to_english_prompt | llm,
        translate_to_russian_chain=translate_to_russian_prompt | llm,
        **kwargs,
    )
