class AgentSearchConfig:
    def __init__(self, evaluation_temperature=0.2):
        self.evaluation_temperature = evaluation_temperature

def perform_llm_assisted_search(input_value):
    if input_value not in ['standard', 'llm_assisted']:
        raise ValueError("Invalid input value.")
    # Function implementation here

def test_agent_search_config():
    config = AgentSearchConfig()
    assert config.evaluation_temperature == 0.2
    config_with_custom_temp = AgentSearchConfig(evaluation_temperature=0.5)
    assert config_with_custom_temp.evaluation_temperature == 0.5

def test_perform_llm_assisted_search():
    try:
        perform_llm_assisted_search('invalid_choice')
    except ValueError as e:
        assert str(e) == "Invalid input value."
    assert perform_llm_assisted_search('standard') is None
    assert perform_llm_assisted_search('llm_assisted') is None