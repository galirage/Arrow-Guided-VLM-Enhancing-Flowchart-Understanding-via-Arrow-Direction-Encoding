from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig, ensure_config
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env file into environment variables before BaseSettings reads them
load_dotenv()


class ArrowGuidedVLMConfiguration(BaseSettings):
    """Configuration for the Arrow Guided VLM workflow."""

    # BaseSettings will now read from environment variables loaded by load_dotenv()
    # env_file in model_config might become redundant but doesn't hurt
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Azure Document Intelligence Settings
    azure_document_intelligence_endpoint: str = Field(
        ...,  # Indicates this field is required, should be set via .env
        description="The endpoint for Azure Document Intelligence service.",
        validation_alias="AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",  # Maps to env variable
    )
    azure_document_intelligence_key: SecretStr = Field(
        ...,  # Indicates this field is required, should be set via .env
        description="The API key for Azure Document Intelligence service.",
        validation_alias="AZURE_DOCUMENT_INTELLIGENCE_KEY",  # Maps to env variable
    )

    # Azure OpenAI Settings
    azure_openai_endpoint: str = Field(
        ...,
        description="The endpoint for Azure OpenAI service.",
        validation_alias="AZURE_OPENAI_ENDPOINT",
    )
    azure_openai_api_key: SecretStr = Field(
        ...,
        description="The API key for Azure OpenAI service.",
        validation_alias="AZURE_OPENAI_API_KEY",
    )
    azure_openai_model_name: str = Field(
        default="gpt-4o",
        description="The Azure OpenAI model name to use (e.g., gpt-4o).",
    )
    azure_openai_api_version: str = Field(
        default="2025-01-01-preview", description="The Azure OpenAI API version to use."
    )
    max_retries: int = Field(
        default=3,
        description="The maximum number of retries for the API call.",
    )
    # Workflow Settings
    detection_ocr_match_threshold: float = Field(
        default=0.5,
        description="Threshold for matching detection bounding boxes with OCR results.",
    )

    @classmethod
    def from_runnable_config(
        cls, config: RunnableConfig | None = None
    ) -> "ArrowGuidedVLMConfiguration":
        """Load configuration w/ defaults for the given invocation.

        Loads base settings from environment variables/.env first,
        then overrides with values from the RunnableConfig if provided.
        """
        # 1. Load base configuration from environment variables (already loaded by load_dotenv)
        try:
            instance = cls()  # type: ignore
        except Exception as e:
            print(f"Error loading base configuration from environment: {e}")
            raise

        # 2. Ensure RunnableConfig and extract configurable fields
        runnable_config = ensure_config(config)
        configurable = runnable_config.get("configurable") or {}

        # 3. Override with values from RunnableConfig if they exist
        if configurable:
            instance_dict = instance.model_dump()
            _fields = cls.model_fields.keys()
            updated = False
            for k, v in configurable.items():
                if k in _fields:
                    # Special handling for SecretStr if needed, though Pydantic might handle it
                    instance_dict[k] = v
                    updated = True

            # Re-validate if any values were updated
            if updated:
                try:
                    instance = cls.model_validate(instance_dict)
                except Exception as e:
                    print(f"Error applying RunnableConfig overrides: {e}")
                    # Decide how to handle validation errors - raise, warn, ignore?
                    raise  # Re-raise for now

        return instance
