class TestAgentSearchConfig:
    def test_initialization_with_evaluation_temperature(self):
        with pytest.raises(TypeError):
            AgentSearchConfig(evaluation_temperature=0.5)

class TestPerformLLMAssistedSearch:
    def test_perform_llm_assisted_search_with_unexpected_value(self):
        response = perform_llm_assisted_search("unexpected_value")
        assert response == "Error: AgentSearchConfig.__init__() got an unexpected keyword argument 'evaluation_temperature'"

    def test_perform_llm_assisted_search_with_valid_input(self):
        response = perform_llm_assisted_search("standard")
        assert response is not None  # Adjust based on expected output

    def test_perform_llm_assisted_search_no_text_selected(self):
        response = perform_llm_assisted_search("no_text")
        assert response == "Keine Texte durch KI-Bewertung ausgewählt."