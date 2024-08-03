
urls = [
    'https://midas.app/terms-and-conditions',
    'https://docs.ondo.finance/legal/terms-of-service',
    'https://docs.mountainprotocol.com/legal/terms-and-conditions',
    'https://www.hashnote.com/terms-and-conditions',
    'https://www.ftinstitutional.com/terms-of-use',
    'https://www.aktionariat.com/terms-of-service',
    'https://www.angle.money/terms',
    'https://www.blackrock.com/corporate/compliance/terms-and-conditions',
]

from typing import List
import csv


from backend.lib.text_content_analysis import crawl_data_from_website
from langchain_text_splitters import HTMLSectionSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

headers_to_split_on = [
    ("h1", "Header 1"), 
    ("h2", "Header 2"), 
    ("h3", "Header 3"), 
    ("h4", "Header 4")
    ]

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
USING_OPEN_AI = False

def write_csv(filepath, dicts):
    with open(filepath,'w') as f:
        w = csv.writer(f)
        for dict in dicts:
            w.writerows(dict.items())


def main(urls: List[str]):
    outputs = []
    for url in urls:
        data = crawl_data_from_website(url)
        if data:
            html_splitter = HTMLSectionSplitter(headers_to_split_on=headers_to_split_on)
            html_header_splits = html_splitter.split_text(data)
            # Split
            chunk_size = 500
            chunk_overlap = 0
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            splits = text_splitter.split_documents(html_header_splits)
            retriever = BM25Retriever.from_documents(splits)
            result = retriever.invoke("fee cost")
            location = [doc.metadata for doc in result]
            text = [doc.page_content for doc in result]

            if USING_OPEN_AI:
                template = """Please explain shortly the information about the fees or cost \\
                    or financial obligations which users might incur by using the service \\
                        or product of that website with the following context {context}"""

                prompt = PromptTemplate.from_template(template)

                context = "\n".join(text)
                llm = OpenAI(openai_api_key="YOUR_API_KEY")
                llm_chain = prompt | llm
                explanation = llm_chain.invoke(context)
            else:
                explanation = "\n".join(text)
                

            output = {
                "result" : len(result)>0,
                "explain" : explanation,
                "location" : location,
                "text" : text,
            }
        else:
            output = {
                "result" : False,
                "explain" : None,
                "location" : [],
                "text" : [],
            }
        outputs.append(output)
        write_csv("./output/output_P1.csv", outputs)
    return outputs

if __name__ == "__main__":
    main(urls)