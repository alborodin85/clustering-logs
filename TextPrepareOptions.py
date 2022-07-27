class TextPrepareOptions:
    def __init__(self):
        self.strip = True,
        self.lower = False,
        self.clearPunctuation = False,
        self.clearDigits = False,
        self.stopWordsEnglish = False,
        self.stopWordsRussian = False,
        self.lemmatizationEnglish = False,
        self.stemmingEnglish = False,
        self.stemmingRussian = False,
        self.sinonymizeEnglish = False,
