import stanfordnlp
stanfordnlp.download('en') 
nlp = stanfordnlp.Pipeline()
doc = nlp("Vietnam National University of Science")
doc.sentences[0].print_dependencies()