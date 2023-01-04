def remove_special_characters(text):
    text = re.sub('[^a-zA-z0-9\s]', '', text)
    return text