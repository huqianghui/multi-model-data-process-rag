import argparse
import asyncio
import dataclasses
import os
import time

from azure.core.credentials import AzureKeyCredential
from azure.identity import AzureDeveloperCliCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    AIServicesVisionParameters,
    AIServicesVisionVectorizer,
    AzureOpenAIParameters,
    AzureOpenAIVectorizer,
    CorsOptions,
    HnswAlgorithmConfiguration,
    ScoringProfile,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    TextWeights,
    VectorSearch,
    VectorSearchProfile,
)
from dotenv import load_dotenv
from tqdm import tqdm

# 加载 .env 文件中的环境变量
load_dotenv()

from data_utils import process_images_records


def create_search_index(index_name, index_client):
    print(f"Ensuring search index {index_name} exists")
    if index_name not in index_client.list_index_names():
        scoring_profile = ScoringProfile(
            name="firstProfile",
            text_weights=TextWeights(weights={
                                        "caption": 5, 
                                        "content": 1, 
                                        "ocrContent": 2}),
            function_aggregation="sum")
        scoring_profiles = []
        scoring_profiles.append(scoring_profile)
        
        index = SearchIndex(
            name=index_name,
            cors_options = CorsOptions(allowed_origins=["*"], max_age_in_seconds=600),
            scoring_profiles = scoring_profiles,
            fields=[
                SimpleField(name="id", type=SearchFieldDataType.String, key=True,searchable=False, filterable=True, sortable=True, facetable=True),
                SearchableField(name="caption", type=SearchFieldDataType.String, analyzer_name="zh-Hans.microsoft"),# caption of the picture
                SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="zh-Hans.microsoft"),# context of the picture from gpt-4o
                SimpleField(name="imageUrl", type=SearchFieldDataType.String,Searchable=False,filterable=False, sortable=True, facetable=True),# url of the picture
                SearchableField(name="ocrContent", type=SearchFieldDataType.String,analyzer_name="zh-Hans.microsoft"), # context of the picture from document intelligence
                SearchField(name="captionVector", 
                            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                            hidden=False, 
                            searchable=True, 
                            filterable=False, 
                            sortable=False, 
                            facetable=False,
                            vector_search_dimensions=1536, 
                            vector_search_profile_name="azureOpenAIHnswProfile"), #  the caption's vector of the picture
                SearchField(name="contentVector", 
                            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                            hidden=False, 
                            searchable=True, 
                            filterable=False, 
                            sortable=False, 
                            facetable=False,
                            vector_search_dimensions=1536, 
                            vector_search_profile_name="azureOpenAIHnswProfile"),  # content vector of the picture from gpt-4o
                SearchField(name="ocrContentVecotor", 
                            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                            hidden=False, 
                            searchable=True, 
                            filterable=False, 
                            sortable=False, 
                            facetable=False,
                            vector_search_dimensions=1536, 
                            vector_search_profile_name="azureOpenAIHnswProfile"),  # content vector of the picture from document intelligence
                SearchField(name="imageVecotor", 
                            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                            hidden=False, 
                            searchable=True, 
                            filterable=False, 
                            sortable=False, 
                            facetable=False,
                            vector_search_dimensions=1024, 
                            vector_search_profile_name="azureComputerVisionHnswProfile")  # content vector of the picture from computer vision
            ],
            semantic_search=SemanticSearch(
                configurations=[
                    SemanticConfiguration(
                        name="default",
                        prioritized_fields=SemanticPrioritizedFields(
                            title_field=SemanticField(field_name="caption"),
                            content_fields=[
                                SemanticField(field_name="content")
                            ],
                            keywords_fields=[
                                SemanticField(field_name="ocrContent")
                            ]
                        ),
                    )
                ]
            ),
            vector_search=VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="myHnsw")
                ],
                profiles=[
                    VectorSearchProfile(
                        name="azureOpenAIHnswProfile",
                        algorithm_configuration_name="myHnsw",
                        vectorizer="azureOpenAIVectorizer"),
                    VectorSearchProfile(
                        name="azureComputerVisionHnswProfile",
                        algorithm_configuration_name="myHnsw",
                        vectorizer="azureComputerVisionVectorizer")],
                vectorizers=[
                    AzureOpenAIVectorizer(
                        name="azureOpenAIVectorizer",
                        azure_open_ai_parameters=AzureOpenAIParameters(
                            resource_uri=os.environ.get("AZURE_OPENAI_ENDPOINT"),
                            deployment_id="text-embedding-ada-002",
                            model_name="text-embedding-ada-002",
                            api_key=os.environ.get("AZURE_OPENAI_API_KEY"))),
                    AIServicesVisionVectorizer(
                        name="azureComputerVisionVectorizer",
                        ai_services_vision_parameters=AIServicesVisionParameters(
                            resource_uri=os.environ.get("AZURE_COMPUTER_VISION_ENDPOINT"),
                            api_key=os.environ.get("AZURE_COMPUTER_VISION_KEY"),
                            model_version="2023-04-15"))
                ]
            )
        )
        print(f"Creating {index_name} search index")
        index_client.create_index(index)
    else:
        print(f"Search index {index_name} already exists")


def upload_documents_to_index(docs, search_client, upload_batch_size=50):
    to_upload_dicts = []

    for document in docs:
        d = dataclasses.asdict(document)
        # add id to documents
        d.update({"@search.action": "upload", "id": str(d["id"])})

        if "captionVector" in d and d["captionVector"] is None:
            del d["captionVector"]
        if "contentVector" in d and d["contentVector"] is None:
            del d["contentVector"]
        if "ocrContentVecotor" in d and d["ocrContentVecotor"] is None:
            del d["ocrContentVecotor"]
        if "imageVecotor" in d and d["imageVecotor"] is None:
            del d["imageVecotor"]

        to_upload_dicts.append(d)

    # Upload the documents in batches of upload_batch_size
    for i in tqdm(
        range(0, len(to_upload_dicts), upload_batch_size), desc="Indexing Chunks..."
    ):
        batch = to_upload_dicts[i : i + upload_batch_size]
        results = search_client.upload_documents(documents=batch)
        num_failures = 0
        errors = set()
        for result in results:
            if not result.succeeded:
                print(
                    f"Indexing Failed for {result.key} with ERROR: {result.error_message}"
                )
                num_failures += 1
                errors.add(result.error_message)
        if num_failures > 0:
            raise Exception(
                f"INDEXING FAILED for {num_failures} documents. Please recreate the index."
                f"To Debug: PLEASE CHECK chunk_size and upload_batch_size. \n Error Messages: {list(errors)}"
            )


def validate_index(index_name, index_client):
    for retry_count in range(5):
        stats = index_client.get_index_statistics(index_name)
        num_chunks = stats["document_count"]
        if num_chunks == 0 and retry_count < 4:
            print("Index is empty. Waiting 60 seconds to check again...")
            time.sleep(60)
        elif num_chunks == 0 and retry_count == 4:
            print("Index is empty. Please investigate and re-index.")
        else:
            print(f"The index contains {num_chunks} chunks.")
            average_chunk_size = stats["storage_size"] / num_chunks
            print(f"The average chunk size of the index is {average_chunk_size} bytes.")
            break


async def create_and_populate_index(index_name:str, index_client:SearchIndexClient,search_client:SearchClient):
    # create or update search index with compatible schema
    create_search_index(index_name, index_client)

    # process records
    # print("insert data...")
    recordResult = await process_images_records(file_path= "multi-models/image_captions/ima_files_2_test.txt")

    if len(recordResult.documentList) == 0:
        raise Exception("No records found. Please check the data path and records.")

    print(f"Processed {recordResult.totalRecords} records")
    print(f"records with errors: {len(recordResult.failedImageList)} records")
    print(f"valid records: {len(recordResult.documentList)} documents")

    # upload documents to index
    print("Uploading documents to index...")
    upload_documents_to_index(recordResult.documentList, search_client)

    # check if index is ready/validate index
    print("Validating index...")
    validate_index(index_name, index_client)
    print("Index validation completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare documents by extracting content from PDFs, splitting content into sections and indexing in a search index.",
        epilog="Example: prepdocs.py --searchservice mysearch --index myindex",
    )
    parser.add_argument(
        "--tenantid",
        required=False,
        default="",
        help="Optional. Use this to define the Azure directory where to authenticate)",
    )
    parser.add_argument(
        "--searchservice",
        default="",
        help="Name of the Azure Cognitive Search service where content should be indexed (must exist already)",
    )
    parser.add_argument(
        "--index",
        default="",
        help="Name of the Azure Cognitive Search index where content should be indexed (will be created if it doesn't exist)",
    )
    parser.add_argument(
        "--searchkey",
        required=False,
        default="",
        help="Optional. Use this Azure Cognitive Search account key instead of the current user identity to login (use az login to set current user for Azure)",
    )

    args = parser.parse_args()

    # Use the current user identity to connect to Azure services unless a key is explicitly set for any of them
    azd_credential = (
        AzureDeveloperCliCredential()
        if args.tenantid == None
        else AzureDeveloperCliCredential(tenant_id=args.tenantid, process_timeout=60)
    )
    default_creds = azd_credential if args.searchkey == None else None
    search_creds = (
        default_creds if args.searchkey == None else AzureKeyCredential(args.searchkey)
    )
    
    print("Data preparation script started")
    print("Preparing data for index:", args.index)
    search_endpoint = f"https://{args.searchservice}.search.windows.net/"
    index_client = SearchIndexClient(endpoint=search_endpoint, credential=search_creds)

    search_client = SearchClient(
        endpoint=search_endpoint, credential=search_creds, index_name=args.index
    )

    asyncio.run(create_and_populate_index(args.index, index_client,search_client))
    print("Data preparation for index", args.index, "completed")
