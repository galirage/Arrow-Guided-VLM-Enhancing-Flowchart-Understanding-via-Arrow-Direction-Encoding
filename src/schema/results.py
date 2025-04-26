from pydantic import BaseModel, Field, RootModel


class QAEntry(BaseModel):
    """1 行（＝ 1 質問‐回答ペア）のスキーマ"""

    image_path: str = Field(
        ...,
        description="フローチャート画像へのパス（相対・絶対どちらでも可）",
    )
    image_num: int = Field(
        ...,
        description="画像の一意な数値 ID（例: 150）",
    )
    question_id: str = Field(
        ...,
        description="`<image_num>_<index>` 形式の複合 ID（例: '150_0'）",
    )
    question_type: int = Field(
        ...,
        description="質問タイプ（整数で管理。1: next-step など）",
    )
    question: str = Field(
        ...,
        description="LLM へ投げた質問文",
    )
    answer_collect: str = Field(
        ...,
        description="手動アノテーションによる正解",
    )
    answer_from_llm_with_no_dec_ocr: str = Field(
        ...,
        description="検出・OCR 情報なしで生成した LLM の回答",
    )
    answer_from_llm_with_dec_ocr: str = Field(
        ...,
        description="検出・OCR 情報ありで生成した LLM の回答",
    )


class QAEvaluationSet(RootModel):
    """ファイル全体（リスト）のラッパー。
    Pydantic v2 では RootModel を使用して Top-Level の list を表現する。
    """

    root: list[QAEntry]
