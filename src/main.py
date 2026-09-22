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
# 2. Create OpenAI client
# --------------------------------------------------

client = OpenAI(api_key=api_key)


# --------------------------------------------------
# 3. Describe our calculator tool to OpenAI
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
                "description": "The second number. Not required for squre",
            },
        },
        "required": ["a", "operation"],
    },
}


# --------------------------------------------------
# 10. Create a common function
# --------------------------------------------------
def run_agent(user_question: str, force_tool: bool):
    response = client.responses.create(
        model="gpt-5.6-luna",
        input=user_question,
        tools=[calculator_tool],
        tool_choice="required" if force_tool else "auto",
    )

    for item in response.output:
        import json

        if item.type == "function_call":
            print("Tool:", item.name)
            print("Arguments:", item.arguments)

            # --------------------------------------------------
            # 6. Execute the requested Python function
            # --------------------------------------------------

            if item.name == "calculator":
                # import json

                arguments = json.loads(item.arguments)

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
                # 8. Ask OpenAI for the final answer
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
                            "content": "Using the calculator result, answer the user's question in a clear natural-language sentence.",
                        },
                    ],
                    tools=[calculator_tool],
                )

                # --------------------------------------------------
                # 9. Print OpenAI's final answer
                # --------------------------------------------------

                print("Final answer:", final_response.output_text)


user_question = "What is 25 multiplied by 48?"

run_agent(user_question, force_tool=False)

user_square_question = "What is square of 25?"
run_agent(user_square_question, force_tool=True)
