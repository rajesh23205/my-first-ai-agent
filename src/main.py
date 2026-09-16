import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from tools import calculator


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set")


# --------------------------------------------------
# 2. Create Gemini client
# --------------------------------------------------

client = genai.Client(api_key=api_key)


# --------------------------------------------------
# 3. Describe our calculator tool to Gemini
# --------------------------------------------------

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
                        description="The first number.",
                    ),
                    "b": types.Schema(
                        type="NUMBER",
                        description="The second number.",
                    ),
                    "operation": types.Schema(
                        type="STRING",
                        description=(
                            "The operation to perform: "
                            "add, subtract, multiply, or divide."
                        ),
                    ),
                },
                required=["a", "b", "operation"],
            ),
        )
    ]
)


# --------------------------------------------------
# 4. Send the user's question to Gemini
# --------------------------------------------------

user_question = "What is 25 multiplied by 48?"

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=user_question,
    config=types.GenerateContentConfig(
        tools=[calculator_tool]
    ),
)


# --------------------------------------------------
# 5. Check whether Gemini wants to use a tool
# --------------------------------------------------

for part in response.candidates[0].content.parts:

    if part.function_call:

        function_call = part.function_call

        print("Tool:", function_call.name)
        print("Arguments:", function_call.args)


        # --------------------------------------------------
        # 6. Execute the requested Python function
        # --------------------------------------------------

        if function_call.name == "calculator":

            result = calculator(**function_call.args)

            print("Tool result:", result)


            # --------------------------------------------------
            # 7. Create the tool response
            # --------------------------------------------------

            function_response_part = types.Part.from_function_response(
                name=function_call.name,
                response={
                    "result": result
                },
                # id=function_call.id,
            )


            # --------------------------------------------------
            # 8. Send the tool result back to Gemini
            # --------------------------------------------------

            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text=user_question
                        )
                    ],
                ),

                # Gemini's previous response containing
                # the function call
                response.candidates[0].content,

                # Our Python tool's result
                types.Content(
                    role="user",
                    parts=[
                        function_response_part
                    ],
                ),
            ]


            # --------------------------------------------------
            # 9. Ask Gemini for the final answer
            # --------------------------------------------------

            final_response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=[calculator_tool]
                ),
            )


            # --------------------------------------------------
            # 10. Print Gemini's final answer
            # --------------------------------------------------

            print("Final answer:", final_response.text)