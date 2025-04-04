from pydantic import BaseModel, Field


class LLMOutputSchema(BaseModel):
    """
    LLMによる評価結果を格納するスキーマ定義
    """

    reasoning: str | None = Field(None, description="LLMが評価を下した理由や説明")
    is_correct: bool = Field(..., description="最終的な評価結果")


class EvaluationData(BaseModel):
    """
    評価対象となるデータのスキーマ定義
    """

    question: str = Field(..., description="評価対象の質問")
    image: bytes = Field(..., description="評価対象の画像")
    reference_answer: str = Field(..., description="模範回答")
    model_output: str = Field(..., description="評価対象のモデルが生成した回答")


class EvaluationResult(EvaluationData):
    """
    LLMによる評価結果を格納するスキーマ定義
    """

    reasoning: str | None = Field(None, description="LLMが評価を下した理由や説明")
    is_correct: bool = Field(..., description="最終的な評価結果")


class EvaluationDataset(BaseModel):
    """
    評価を実行するデータセットの定義
    """

    items: list[EvaluationData] = Field(..., description="評価対象のデータ一覧")


class EvaluatedDataset(BaseModel):
    """
    評価が実行された後の結果データセットの定義
    """

    results: list[EvaluationResult] = Field(
        ..., description="LLMによって評価された結果一覧"
    )
