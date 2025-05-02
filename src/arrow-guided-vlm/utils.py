from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import AzureChatOpenAI

from .configuration import ArrowGuidedVLMConfiguration


def init_model(config: ArrowGuidedVLMConfiguration) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment=config.azure_openai_model_name,
        api_version=config.azure_openai_api_version,
        max_retries=config.max_retries,
        temperature=0,
        rate_limiter=InMemoryRateLimiter(),
    )


def init_document_analysis_client(
    config: ArrowGuidedVLMConfiguration,
) -> DocumentAnalysisClient:
    return DocumentAnalysisClient(
        endpoint=config.azure_document_intelligence_endpoint,
        credential=AzureKeyCredential(
            config.azure_document_intelligence_key.get_secret_value()
        ),
    )
