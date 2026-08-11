import datetime


def time():
    """Return the current time as a string in HH:MM:SS format."""
    return datetime.datetime.now().strftime("%H:%M:%S")


def date():
    """Return the current date as a string in YYYY-MM-DD format."""
    return datetime.datetime.now().strftime("%Y-%m-%d")


def calulator(mathematical_expression: str):
    """Evaluate a mathematical expression and return the result."""
    try:
        result = eval(mathematical_expression)
        return result
    except Exception as e:
        return f"Invalid mathematical expression: {e}"


def stop():
    """Call this function if the user says 'stop', 'quit', or 'exit' to terminate the program."""
    print("Rick and Morty AI Gun is shutting down. Goodbye! - Deekshant Wadhwa")
    exit(0)


misc_tools = [time, date, calulator, stop]
