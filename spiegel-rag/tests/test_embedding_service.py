class TestAgentSearchConfig:
    def test_initialization_with_evaluation_temperature(self):
        with pytest.raises(TypeError):
            AgentSearchConfig(evaluation_temperature=0.5)

class TestLLMAssistedSearch:
    def test_perform_llm_assisted_search_with_unexpected_value(self):
        result = perform_llm_assisted_search("unexpected_value")
        assert result == "Error: AgentSearchConfig.__init__() got an unexpected keyword argument 'evaluation_temperature'"

    def test_perform_llm_assisted_search_no_texts_selected(self):
        result = perform_llm_assisted_search("standard")
        assert result == "Keine Texte durch KI-Bewertung ausgewählt."

    def test_temperature_setting(self):
        llm_service = LLMService()
        llm_service.set_temperature(0.5)
        assert llm_service.temperature == 0.5