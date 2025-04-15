from unittest.mock import MagicMock, patch

import pytest
from langchain_core.runnables import RunnableConfig

from src.evaluator.configuration import Configuration

# Import the function to test and its dependencies
from src.evaluator.graph import evaluator
from src.evaluator.state import State  # Use State for type hint, provide necessary data
from src.schema import EvaluationData, EvaluationResult, LLMOutputSchema


@pytest.fixture
def mock_dependencies():
    """Fixture to mock AzureChatOpenAI and ChatPromptTemplate."""
    # Patch both dependencies within the scope of this fixture
    with patch("src.evaluator.graph.AzureChatOpenAI") as mock_model_cls, patch(
        "src.evaluator.graph.ChatPromptTemplate"
    ) as mock_prompt_cls:
        # --- Configure Mock AzureChatOpenAI ---
        mock_model_instance = MagicMock()
        # Mock the object returned by with_structured_output
        mock_structured_output = MagicMock()
        mock_model_instance.with_structured_output.return_value = mock_structured_output
        # Ensure that when AzureChatOpenAI() is called, our mock instance is returned
        mock_model_cls.return_value = mock_model_instance

        # --- Configure Mock ChatPromptTemplate and Chain ---
        mock_prompt_instance = MagicMock()
        # This is the final chain object resulting from: evaluation_prompt | model_with_structured_output
        mock_chain = MagicMock()

        # Define the expected output object from the final chain's invoke
        expected_result_obj = LLMOutputSchema(
            reasoning="The model correctly identified the capital.",
            is_correct=True,
        )
        # Configure the mock chain's invoke method to return the predefined result
        mock_chain.invoke.return_value = expected_result_obj

        # Mock the piping behavior: evaluation_prompt | model_with_structured_output should yield mock_chain
        # LangChain uses .pipe() or __or__ for chaining Runnables
        # The __or__ method should be mocked on the left-hand side object, which is mock_prompt_instance
        mock_prompt_instance.pipe.return_value = mock_chain
        mock_prompt_instance.__or__.return_value = mock_chain

        # Ensure that when ChatPromptTemplate() is called, our mock instance is returned
        mock_prompt_cls.return_value = mock_prompt_instance

        # Yield the mocks needed for verification in the test
        # We need the mock_chain to verify its invoke call
        yield (
            mock_model_cls,
            mock_prompt_cls,
            mock_chain,
            expected_result_obj,
        )  # Pass expected_result_obj


def test_evaluator_unit(mock_dependencies):
    """Test the evaluator function as a unit."""
    # Unpack the mocks provided by the fixture
    mock_model_cls, mock_prompt_cls, mock_chain, expected_result_obj = (
        mock_dependencies  # Unpack expected_result_obj
    )

    # --- Arrange ---
    # Sample input data for the evaluator function
    eval_data = EvaluationData(
        question="What is the capital of France?",
        image=b"dummy_image_bytes",
        reference_answer="Paris",
        model_output="Paris",
    )
    # The evaluator function expects a State object.
    # We provide the input data and a placeholder for the result field.
    # Create a dummy EvaluationResult for initialization to bypass validation issue
    dummy_result = EvaluationResult(
        question="",
        image=b"",
        reference_answer="",
        model_output="",
        is_correct=False,
        reasoning=None,
    )
    initial_state = State(
        question=eval_data.question,
        image=eval_data.image,
        reference_answer=eval_data.reference_answer,
        model_output=eval_data.model_output,
        reasoning=dummy_result.reasoning,
        is_correct=dummy_result.is_correct,
    )

    # Configuration needed by the evaluator function
    run_config = RunnableConfig(configurable=Configuration(max_retries=1).model_dump())

    # --- Act ---
    # Call the evaluator function directly with the prepared state and config
    result_dict = evaluator(initial_state, config=run_config)

    # --- Assert ---
    # Check the structure and content of the returned dictionary
    assert isinstance(result_dict, dict)
    assert "reasoning" in result_dict  # Check for the keys returned by evaluator
    assert "is_correct" in result_dict  # Check for the keys returned by evaluator
    # Ensure the values match the expected object from the mocked chain
    assert result_dict["is_correct"] == expected_result_obj.is_correct
    assert result_dict["reasoning"] == expected_result_obj.reasoning
    # Optionally, check specific fields in the result object
    # Add type annotation to help the linter
    # evaluated_result = cast(EvaluationResult, result_dict["evaluated_result"]) # The result is a dict now
    # assert evaluated_result.is_correct is True
    # assert evaluated_result.reasoning == "The model correctly identified the capital."

    # Verify that the dependencies were called as expected
    mock_model_cls.assert_called_once()  # Was AzureChatOpenAI instantiated?

    # Get the actual arguments passed to ChatPromptTemplate to ensure they are correct (Optional but good practice)
    # print(mock_prompt_cls.call_args)

    # Verify the chain was invoked correctly
    # Get the object that `invoke` was called on (it's the mock_chain)
    # And verify its call arguments
    mock_chain.invoke.assert_called_once_with({
        "question": eval_data.question,
        "reference_answer": eval_data.reference_answer,
        "model_output": eval_data.model_output,
    })
