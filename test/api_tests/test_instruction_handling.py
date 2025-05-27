#!/usr/bin/env python3
"""Test how instruction/system messages are handled in the ensemble"""

import logging
from ensemblehub.conversation import ConversationTemplate

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_conversation_template():
    """Test ConversationTemplate behavior with and without system message"""
    
    print("Test 1: With system message")
    print("="*60)
    
    convo1 = ConversationTemplate(
        system_prompt="You are a helpful math assistant.",
        initial_question="What is 2+2?"
    )
    
    print("render():")
    print(repr(convo1.render()))
    print("\nrender_dict():")
    print(convo1.render_dict())
    print("\nrender_list():")
    print(convo1.render_list())
    
    print("\n" + "="*60)
    print("Test 2: Without system message (None)")
    print("="*60)
    
    convo2 = ConversationTemplate(
        system_prompt=None,
        initial_question="What is 3+3?"
    )
    
    print("render():")
    print(repr(convo2.render()))
    print("\nrender_dict():")
    print(convo2.render_dict())
    print("\nrender_list():")
    print(convo2.render_list())
    
    print("\n" + "="*60)
    print("Test 3: With empty string system message")
    print("="*60)
    
    convo3 = ConversationTemplate(
        system_prompt="",
        initial_question="What is 4+4?"
    )
    
    print("render():")
    print(repr(convo3.render()))
    print("\nrender_dict():")
    print(convo3.render_dict())
    print("\nrender_list():")
    print(convo3.render_list())
    
    print("\n" + "="*60)
    print("Test 4: API default behavior")
    print("="*60)
    
    # Simulate what happens in API when no system message is provided
    messages = [{"role": "user", "content": "What is 5+5?"}]
    system_messages = [msg for msg in messages if msg["role"] == "system"]
    instruction = system_messages[0]["content"] if system_messages else "You are a helpful assistant."
    
    print(f"No system message → instruction: {repr(instruction)}")
    
    example = {
        "instruction": instruction,
        "input": "What is 5+5?",
        "output": ""
    }
    print(f"Example dict: {example}")

if __name__ == "__main__":
    test_conversation_template()