from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import ensure_config
from pydantic import BaseModel, Field


class Configuration(BaseModel):
    """The configuration for the evaluator."""

    # Azure OpenAIの設定
    model_name: str = Field(
        default="gpt-4o",
        description="The name of the model to use for the evaluator.",
    )

    api_version: str = Field(
        default="2025-01-01-preview",
        description="The version of the API to use for the evaluator.",
    )

    # リトライの設定
    max_retries: int = Field(
        default=3,
        description="The maximum number of retries for the API call.",
    )

    # レート制限の設定
    rate_limit_max_requests: int = Field(
        default=100,
        description="The maximum number of API requests allowed within the specified interval.",
    )

    # レート制限の間隔
    rate_limit_interval: int = Field(
        default=60,
        description="The time interval in seconds for rate limiting (e.g., 60 for per-minute rate limiting).",
    )

    # 並列処理の設定
    parallel_processing_count: int = Field(
        default=5,
        description=(
            "The number of items to process in a single batch. Adjust based on API "
            "limits and computational resources."
        ),
    )

    @classmethod
    def from_runnable_config(
        cls, config: RunnableConfig | None = None
    ) -> "Configuration":
        """Load configuration w/ defaults for the given invocation."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        # Use model_fields for Pydantic v2+
        _fields = cls.model_fields.keys()
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
