import spacy

nlp = spacy.load('en_core_web_sm')

text = 'Vietnam National University of Science'

textArr = nlp(text)

print([token.text for token in textArr])
