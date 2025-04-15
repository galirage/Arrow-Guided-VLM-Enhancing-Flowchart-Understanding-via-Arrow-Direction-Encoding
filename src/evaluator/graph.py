from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableConfig
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph

from ..schema import LLMOutputSchema
from .configuration import Configuration
from .prompts import evaluation_human_message, evaluation_system_prompt
from .state import InputState, State

load_dotenv()


def evaluator(state: State, config: RunnableConfig) -> dict[str, str | bool | None]:
    """評価ノード"""
    configuration = Configuration.from_runnable_config(config)

    rate_limiter = InMemoryRateLimiter()

    model = AzureChatOpenAI(
        azure_deployment=configuration.model_name,
        api_version=configuration.api_version,
        temperature=0,
        max_retries=configuration.max_retries,
        rate_limiter=rate_limiter,
    )

    model_with_structured_output = model.with_structured_output(LLMOutputSchema)

    evaluation_prompt = ChatPromptTemplate(
        messages=[
            ("system", evaluation_system_prompt),
            ("user", evaluation_human_message),
        ],
        input_variables=["question", "reference_answer", "model_output"],
    )

    chain = evaluation_prompt | model_with_structured_output

    result = chain.invoke({
        "question": state.question,
        "reference_answer": state.reference_answer,
        "model_output": state.model_output,
    })

    if isinstance(result, LLMOutputSchema):
        return {
            "reasoning": result.reasoning,
            "is_correct": result.is_correct,
        }
    else:
        raise ValueError("result is not an instance of EvaluationResult")


workflow = StateGraph(
    State, input=InputState, config_schema=Configuration, output=State
)

workflow.add_node("evaluator", evaluator)
workflow.add_edge(START, "evaluator")
workflow.add_edge("evaluator", END)

workflow = workflow.compile()
