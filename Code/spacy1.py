import spacy
nlp = spacy.load('en_core_web_sm')
introduction_file_text = open('sample_data/text.txt').read()
introduction_file_doc = nlp(introduction_file_text)
print ([token.text for token in introduction_file_doc])