import pytest

@pytest.fixture
def agent_search_config():
    from src.ui.handlers.agent_handlers import AgentSearchConfig
    return AgentSearchConfig(evaluation_temperature=0.2)

def test_agent_search_config_initialization(agent_search_config):
    assert agent_search_config.evaluation_temperature == 0.2

def test_perform_llm_assisted_search():
    from src.ui.handlers.agent_handlers import perform_llm_assisted_search
    result = perform_llm_assisted_search("valid input")
    assert isinstance(result, tuple)  # Adjust based on expected output structure

def test_perform_llm_assisted_search_invalid_input():
    from src.ui.handlers.agent_handlers import perform_llm_assisted_search
    with pytest.raises(ValueError):
        perform_llm_assisted_search("invalid input")  # Adjust based on expected behavior