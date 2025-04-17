import os


def clean_filename(text: str) -> str:
    text = text.replace(".", "_")
    text = text.replace(" ", "_")
    return text


def create_directory_if_missing(dir_path) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
