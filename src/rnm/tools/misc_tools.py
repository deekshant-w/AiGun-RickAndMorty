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


misc_tools = [time, date, calulator]
