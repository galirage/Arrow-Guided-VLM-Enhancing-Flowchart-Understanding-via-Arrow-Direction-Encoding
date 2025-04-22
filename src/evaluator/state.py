from pydantic import BaseModel, Field


class InputState(BaseModel):
    """評価パイプラインの入力ステート"""

    question: str = Field(..., description="評価対象の質問")
    image_path: str = Field(..., description="評価対象の画像のパス")
    reference_answer: str = Field(..., description="模範回答")
    model_output: str = Field(..., description="評価対象のモデルが生成した回答")


class State(InputState):
    """評価パイプラインのステート"""

    reasoning: str | None = Field(None, description="LLMが評価を下した理由や説明")
    is_correct: bool = Field(..., description="最終的な評価結果")
