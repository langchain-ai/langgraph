import logging
from typing import List, Any

from langchain.retrievers import WebResearchRetriever
from async_html import AsyncHtmlLoader
from langchain_community.document_transformers.html2text import Html2TextTransformer
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class GigaWebResearchRetriever(WebResearchRetriever):
    retrieve_kwargs: Any = {}
    download_num: int = 5

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Search Google for documents related to the query input.

        Args:
            query: user query

        Returns:
            Relevant documents from all various urls.
        """
        # Get urls
        logger.info("Searching for relevant urls...")
        urls_to_look = []
        search_results = self.search_tool(query, self.num_search_results)
        logger.info("Searching for relevant urls...")
        logger.info(f"Search results: {search_results}")
        for res in search_results:
            if res.get("link", None):
                urls_to_look.append(res["link"])

        # Relevant urls
        urls = set(urls_to_look)

        # Check for any new urls that we have not processed
        new_urls = list(urls.difference(self.url_database))[: self.download_num]

        logger.info(f"New URLs to load: {new_urls}")
        # Load, split, and add new urls to vectorstore
        if new_urls:
            loader = AsyncHtmlLoader(
                new_urls,
                ignore_load_errors=True,
                verify_ssl=self.verify_ssl,
                requests_kwargs={"timeout": 3},
            )
            html2text = Html2TextTransformer()
            logger.info("Indexing new urls...")
            docs = loader.load()
            docs = list(html2text.transform_documents(docs))
            docs = self.text_splitter.split_documents(docs)
            if len(docs) == 0:
                return []
            self.vectorstore.add_documents(docs)
            if len(docs) > 0:
                self.url_database.extend(new_urls)
            else:
                raise ValueError("No documents were loaded")

        # Search for relevant splits
        # TODO: make this async
        logger.info("Grabbing most relevant splits from urls...")
        docs = []
        docs.extend(self.vectorstore.similarity_search(query, **self.retrieve_kwargs))

        # Get unique docs
        unique_documents_dict = {
            (doc.page_content, tuple(sorted(doc.metadata.items()))): doc for doc in docs
        }
        unique_documents = list(unique_documents_dict.values())
        return unique_documents
