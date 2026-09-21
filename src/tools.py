def calculator(a: float, operation: str, b: float | None = None) -> float:
    if operation == "add":
        return a + b

    if operation == "subtract":
        return a - b

    if operation == "multiply":
        return a * b

    if operation == "square":
        return a * a

    if operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

    raise ValueError(f"Unknown operation: {operation}")


if __name__ == "__main__":
    result = calculator(25, "square")
    print(result)