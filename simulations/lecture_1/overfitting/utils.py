def clean_filename(filename):
    filename = filename.replace(" ", "_")
    filename = filename.replace(".", "_")
    return filename
