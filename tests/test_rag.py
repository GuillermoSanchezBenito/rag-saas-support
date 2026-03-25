from src.rag.prompts import get_rag_prompt

def test_system_prompt_structure():
    # test prompt rules
    prompt = get_rag_prompt()
    assert len(prompt.messages) == 2
    system_msg = prompt.messages[0].prompt.template
    
    # check instructions
    assert "Only use the info" in system_msg
    assert "say you don't know" in system_msg
    assert "professional tone" in system_msg
