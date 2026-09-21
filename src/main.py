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
                )
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
# 4. Send the user's question to OpenAI
# --------------------------------------------------

user_question = "What is 25 multiplied by 48?"

response = client.responses.create(
    model="gpt-5.6-luna",
    input=user_question,
    tools=[calculator_tool],
)

user_squre_question = "What is square of 25?"

square_response = client.responses.create(
    model="gpt-5.6-luna",
    input=user_squre_question,
    tools=[calculator_tool],
    tool_choice="required",
)

print("Square response:", square_response.output)


# --------------------------------------------------
# 5. Check whether OpenAI wants to use a tool
# --------------------------------------------------

for item in response.output:

    if item.type == "function_call":

        print("Tool:", item.name)
        print("Arguments:", item.arguments)


        # --------------------------------------------------
        # 6. Execute the requested Python function
        # --------------------------------------------------

        if item.name == "calculator":

            import json

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



# --------------------------------------------------
# 5. Check whether OpenAI wants to use a tool
# --------------------------------------------------

for item in square_response.output:

    if item.type == "function_call":

        print("Tool:", item.name)
        print("Arguments:", item.arguments)

        if item.name == "calculator":

            import json

            arguments = json.loads(item.arguments)

            result = calculator(**arguments)

            print("Tool result:", result)

            tool_output = {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": str(result),
            }

            final_response = client.responses.create(
                model="gpt-5.6-luna",
                input=[
                    {
                        "role": "user",
                        "content": user_squre_question,
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