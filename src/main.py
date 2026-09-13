import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from tools import calculator


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set")


client = genai.Client(api_key=api_key)


calculator_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="calculator",
            description="Performs basic arithmetic calculations.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "a": types.Schema(
                        type="NUMBER",
                        description="The first number."
                    ),
                    "b": types.Schema(
                        type="NUMBER",
                        description="The second number."
                    ),
                    "operation": types.Schema(
                        type="STRING",
                        description="The operation to perform: add, subtract, multiply, or divide."
                    ),
                },
                required=["a", "b", "operation"],
            ),
        )
    ]
)


response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="What is 25 multiplied by 48?",
    config=types.GenerateContentConfig(
        tools=[calculator_tool]
    ),
)


print(response)