from src.rag.prompts import get_rag_prompt

def test_system_prompt_structure():
    """Test if the generated prompt has the correct structure preventing hallucinations."""
    prompt = get_rag_prompt()
    assert len(prompt.messages) == 2
    system_msg = prompt.messages[0].prompt.template
    
    # Check for core instructions
    assert "ONLY use the information" in system_msg
    assert "do not have that information" in system_msg
    assert "professional tone" in system_msg
