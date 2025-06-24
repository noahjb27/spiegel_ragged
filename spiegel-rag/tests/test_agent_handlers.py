import pytest
from src.ui.handlers.agent_handlers import AgentSearchConfig, perform_llm_assisted_search

def test_agent_search_config_initialization():
    with pytest.raises(TypeError):
        AgentSearchConfig(evaluation_temperature=0.5)

def test_perform_llm_assisted_search_invalid_input():
    result = perform_llm_assisted_search("Invalid input")
    assert result == "Error: Invalid input provided."

def test_perform_llm_assisted_search_no_texts_selected():
    result = perform_llm_assisted_search(None)
    assert result == "Keine Texte durch KI-Bewertung ausgewählt."