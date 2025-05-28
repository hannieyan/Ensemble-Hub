"""
Conversation template for dialogue-based generation.
"""

from typing import List, Dict, Optional


class ConversationTemplate:
    """
    A conversation template for constructing dialogue prompts.
    It includes an optional system prompt, a single user question, and accumulated assistant responses.
    """

    def __init__(self, system_prompt: Optional[str] = None, initial_question: str = ""):
        self.system = system_prompt
        self.question = initial_question
        self.assistant_parts: List[str] = []  # Collected assistant responses

    def add_assistant(self, content: str):
        """Append a new assistant response to the prompt context."""
        self.assistant_parts.append(content)

    def render(self) -> str:
        """
        Render the full prompt as a raw string. Includes system prompt if provided.
        """
        lines = []
        if self.system:
            lines.append(f"[SYSTEM] {self.system} [/SYSTEM]")
        lines.append(f"<user>\n{self.question.strip()}\n</user>")
        if self.assistant_parts:
            lines.append(f"<assistant>\n{''.join(self.assistant_parts)}")
        return "".join(lines)

    def render_dict(self) -> Dict[str, str]:
        """
        Render the prompt as a dictionary. Always includes 'instruction' key (empty string if no system prompt).
        """
        output_dict = {
            "instruction": self.system or "",  # Always include instruction key, empty if None
            "input": self.question.strip(),
            "output": "".join(self.assistant_parts)
        }
        return output_dict

    def render_list(self) -> List[Dict[str, str]]:
        """
        Render the prompt as a list of role-based messages.
        """
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": self.question.strip()})
        if self.assistant_parts:
            messages.append({"role": "assistant", "content": "".join(self.assistant_parts)})
        return messages