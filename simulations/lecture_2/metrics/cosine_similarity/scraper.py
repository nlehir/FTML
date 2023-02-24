import wikipedia


def get_text(title):
    print("get text")
    wiki = wikipedia.page(title)
    text = wiki.content
    return text


def clean_text(text):
    print("clean text")
    text = text.replace("==", "")
    text = text.replace("\n", "")
    return text


def save_text(title):
    print(f"\n{title}")
    text = get_text(title)
    text = clean_text(text)
    with open(f"texts/{title}.txt", "a") as file1:
        file1.writelines(text)
        file1.close()


save_text("Antoine Griezmann")
save_text("Karim Benzema")
save_text("Wim Wenders")
save_text("Martin Scorcese")
