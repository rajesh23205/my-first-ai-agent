import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from tools import calculator

# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set")


# --------------------------------------------------
# 2. Create the OpenAI client
# --------------------------------------------------

client = OpenAI(api_key=api_key)


# --------------------------------------------------
# 3. Describe the calculator tool to OpenAI
#
# This tells OpenAI what our Python tool can do.
# OpenAI does NOT execute calculator().
# Our Python code executes it later.
# --------------------------------------------------

calculator_tool = {
    "type": "function",
    "name": "calculator",
    "description": "Performs basic arithmetic calculations.",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {
                "type": "number",
                "description": "The first number.",
            },
            "operation": {
                "type": "string",
                "description": (
                    "The operation to perform: "
                    "add, subtract, multiply, square, or divide."
                ),
            },
            "b": {
                "type": "number",
                "description": ("The second number. Not required for square."),
            },
        },
        "required": ["a", "operation"],
    },
}


# --------------------------------------------------
# 4. Run the AI agent
# --------------------------------------------------


def run_agent(user_question: str, tool_choice: bool = False) -> str:

    # Ask OpenAI to process the user's question.
    response = client.responses.create(
        model="gpt-5.6-luna",
        input=user_question,
        tools=[calculator_tool],
        tool_choice="required" if tool_choice else "auto",
    )

    # --------------------------------------------------
    # 5. Look for a tool call in OpenAI's response
    # --------------------------------------------------

    for item in response.output:
        if item.type != "function_call":
            return response.output_text

        print("Tool:", item.name)
        print("Arguments:", item.arguments)

        # --------------------------------------------------
        # 6. Execute our Python function
        # --------------------------------------------------

        if item.name == "calculator":
            # Convert OpenAI's JSON arguments into a Python dictionary.
            arguments = json.loads(item.arguments)

            # Execute the actual Python calculator function.
            result = calculator(**arguments)

            print("Tool result:", result)

            # --------------------------------------------------
            # 7. Send the tool result back to OpenAI
            # --------------------------------------------------

            tool_output = {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": str(result),
            }

            # --------------------------------------------------
            # 8. Ask OpenAI to generate the final answer
            # --------------------------------------------------

            final_response = client.responses.create(
                model="gpt-5.6-luna",
                input=[
                    {
                        "role": "user",
                        "content": user_question,
                    },
                    {
                        "type": "function_call",
                        "call_id": item.call_id,
                        "name": item.name,
                        "arguments": item.arguments,
                    },
                    tool_output,
                    {
                        "role": "user",
                        "content": (
                            "Using the calculator result, answer "
                            "the user's question in a clear "
                            "natural-language sentence."
                        ),
                    },
                ],
                tools=[calculator_tool],
            )

            return final_response.output_text

    # --------------------------------------------------
    # 9. No tool was called
    # --------------------------------------------------

    return response.output_text


# --------------------------------------------------
# 10. Test multiplication
# --------------------------------------------------

user_question = "What is 25 multiplied by 48?"

answer = run_agent(user_question, tool_choice=True)

print("Final answer:", answer)


# --------------------------------------------------
# 11. Test square
# --------------------------------------------------

user_square_question = "What is square of 25?"

answer = run_agent(user_square_question, tool_choice=True)

print("Final answer:", answer)


# --------------------------------------------------
# 11. Test square
# --------------------------------------------------

open_question = "What is capital of India?"

answer = run_agent(open_question, tool_choice=False)

print("Final answer:", answer)
