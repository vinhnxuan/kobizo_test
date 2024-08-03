from abc import ABC, abstractmethod

KW_CONSTANTS = {
    "cost": ["cost"],
    "fee": ["fee"],
    "fin_obli": ["financ", "obligation"],
}

import re


###### SearchEngine Classs #####
class SearchEngine(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def search(self, text:str):
        pass
    
###### TextSearchEngine Classs #####
class KWSearchEngine(ABC):
    def __init__(self) -> None:
        super().__init__()

    def _get_kw():
        return KW_CONSTANTS["cost"] + KW_CONSTANTS["fee"] + KW_CONSTANTS["fin_obli"]

    def search(self, text:str):
        kws = self.get_kw()
        for kw in kws:
            found = re.search(r"%s" % kw, text, re.IGNORECASE)
            if found:
                return True
        return False
    
    
    ##### NER-based Searching ###########
    ##### Find the similar named entity####
    def __bm25_search(self, text:str):
        return self.__kw_search (text)
    
# class TextContentAnalyzer():
#     def __init__(self) -> None:
#         self.search_cost_engine = CostSearchEngine()
#         self.search_fee_engine = FeeSearchEngine()
#         self.search_fin_engine = FinSearchEngine()

#     def analyze(self, text:str):
#         output ={}
#         self.search_cost_engine


def crawl_data_from_website (url:str):
    import requests
    # Send a GET request to the specified URL and store the response in 'resp'
    resp = requests.get(url)

    # Print the HTTP status code of the response to check if the request was successful
    if resp.status_code ==200:
        return resp.text
    else:
        print ("The URL %s is not reachable"%url)
        return None
    
