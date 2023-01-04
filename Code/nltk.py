import nltk
nltk.download('wordnet')
text = 'Vietnam National University of Science'
tokenizer = nltk.tokenize.WhitespaceTokenizer()
tokenizer.tokenize(text)
