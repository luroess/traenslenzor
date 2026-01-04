import logging

from ollama import Client

from traenslenzor.supervisor.config import settings

logger = logging.getLogger(__name__)

client = Client(host=settings.llm.ollama_url)

logger = logging.getLogger(__name__)


def optimize_text(text: list[str], user_suggestion: str) -> list[str]:
    system = {
        "role": "system",
        "content": (
            """
            You are a text optimization tool.  

            You are given existing text in this format:
                1: This is an
                2: Example text

            You are also given user suggestions in this format:
                user_suggestions:
                    This is a user suggestion

            Your task is to **incorporate the user suggestions into the existing text**, producing a corrected version **while preserving the original structure and numbering**.
            The **number of indices before and after correction must remain the same**.

            Output must follow the same format as the input:
                1: Corrected line 1
                2: Corrected line 2

            Do not output anything other than the corrected text.
            """
        ),
    }
    existing_text_str = "\n".join(f"{i + 1}: {line}" for i, line in enumerate(text))
    message_content = (
        f"Here is the existing text:\n{existing_text_str}\n\n"
        f"user_suggestions:\n    {user_suggestion}"
    )

    message = {
        "role": "user",
        "content": message_content,
    }

    # Call the LLM
    response = client.chat(model=settings.llm.model, messages=[system, message])

    # Extract text from response and parse into list[str]
    raw_output = response["choices"][0]["message"]["content"]
    lines = raw_output.strip().splitlines()

    # Remove numbering and keep clean text lines (optional)
    corrected_lines = []
    for line in lines:
        # Expect format "1: corrected text"
        if ":" in line:
            _, content = line.split(":", 1)
            corrected_lines.append(content.strip())
        else:
            # fallback if numbering missing
            corrected_lines.append(line.strip())

    return corrected_lines
