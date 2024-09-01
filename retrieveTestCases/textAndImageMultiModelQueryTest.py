import asyncio
import os
import sys
from typing import Any, Dict, List

# 假设 multiModelsEmbedding.py 文件在项目的根目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType, VectorizedQuery
from dotenv import load_dotenv
from openai import AzureOpenAI

from multiModelsEmbedding import get_picture_embedding
from pictureFormatProcess import download_and_save_as_pdf
from pictureOcrProcess import analyze_document, get_image_caption_byCV

# Configure environment variables  
load_dotenv()  
azure_search_service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT") 
azure_search_index_name = os.getenv("AZURE_SEARCH_INDEX") 
azure_search_key = os.getenv("AZURE_COGNITIVE_SEARCH_KEY") 
azure_search_credential = AzureKeyCredential(azure_search_key)


azureOpenAIClient = AzureOpenAI(
  api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version = "2024-02-01",
  azure_endpoint =os.getenv("AZURE_OPENAI_BASE") 
)

azure_openAI_embedding_deployment = os.getenv("EMBEDDING_MODEL_DEPLOYMENT")


import logging
import os
from typing import Any, Dict, List

import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure environment variables are loaded
from dotenv import load_dotenv

load_dotenv(verbose=True)

azure_computer_vision_endpoint = "https://cv-hu-test-westus.cognitiveservices.azure.com/"
azure_computer_vision_key = "76fd3ae5ca8346dfa266636d8afc5478"

def get_text_embedding_by_computer_vision(text: str) -> List[float]:
    logging.info(f"Getting text embedding for {text}")
    
    url = azure_computer_vision_endpoint + "computervision/retrieval:vectorizeText?api-version=2024-02-01&model-version=2023-04-15"
    headers = {
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": azure_computer_vision_key
    }
    body = {
        "text": text
    }

    response = requests.post(url, headers=headers, json=body)
    if response.status_code == 200:
        data = response.json()
        return data['vector']
    else:
        raise Exception(f"Error getting text embedding: {response.status_code} - {response.text}")

pdf_dir = "docs/pdf"


if __name__ == "__main__":

    query_image_url="https://img2.tapimg.com/moment/etag/FvhNYMQT78nnCjAvBqHvY40FcH46.jpeg"
    query_text = "DNF手游伤害为什么是黄字？"

    # # generate ocr content by form recognizer service
    # pdfFileLocalPath =  asyncio.run(download_and_save_as_pdf(query_image_url,pdf_dir))
    # ocrContent = asyncio.run(analyze_document(pdfFileLocalPath))
    # captionByCV = asyncio.run(get_image_caption_byCV(query_image_url))

    # query = ocrContent + captionByCV
    
    aoaiResponse = azureOpenAIClient.embeddings.create(input = query_text,model = azure_openAI_embedding_deployment)  
    aoai_embedding_query = aoaiResponse.data[0].embedding
    #print(aoai_embedding_query)

    cv_embedding_query = asyncio.run(get_picture_embedding(query_image_url))
    #print(cv_embedding_query)

    search_client = SearchClient(azure_search_service_endpoint, azure_search_index_name, AzureKeyCredential(azure_search_key))

    aoai_embedding_query = VectorizedQuery(vector=aoai_embedding_query, 
                                k_nearest_neighbors=3, 
                                fields="contentVector,captionVector,ocrContentVecotor")

    azure_cv_embedding_query = VectorizedQuery(vector=cv_embedding_query, 
                                k_nearest_neighbors=3, 
                                fields="imageVecotor")

    results = search_client.search(  
        search_text=query_text,
        search_fields=["caption","content","ocrContent"],
        query_language="zh-cn",
        scoring_profile="firstProfile",   
        vector_queries=[aoai_embedding_query,azure_cv_embedding_query],
        query_type=QueryType.SEMANTIC, 
        semantic_configuration_name='default', 
        select=["id","caption", "content","imageUrl","ocrContent"],
        top=3
    )
    print("####################Results####################")
    
    for result in results:
        print(f"Reranker Score: {result['@search.reranker_score']}")
        print(f"Score: {result['@search.score']}")  
        print(f"Captions: {result['@search.captions']}")  
        print(f"Highlights: {result['@search.highlights']}")  
        print(f"Content: {result['caption']}\n")  
        print(f"imageUrl: {result['imageUrl']}\n")  
        print("###############################")
