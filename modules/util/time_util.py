from datetime import datetime


def get_string_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def generate_default_run_name(training_method) -> str:
    """Generate a default run name from the training method and current timestamp."""
    method = str(training_method).lower().replace(" ", "_")
    return f"{method}_{get_string_timestamp()}"
