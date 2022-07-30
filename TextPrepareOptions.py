class TextPrepareOptions:
    def __init__(self):
        isMySqlLog = True

        self.nClusters = 3

        self.strip = True
        self.lower = True
        self.clearEmails = False
        self.clearPunctuation = True
        self.clearDigits = True
        self.stopWordsEnglish = False
        self.stopWordsRussian = False
        self.lemmatizationEnglish = False
        self.stemmingEnglish = False
        self.stemmingRussian = False
        self.sinonymizeEnglish = False

        self.countRows = 25_000
        self.minCountRepeat = 3
        self.inAllDocument = True

        self.birchItemsInButch = 1000

        if isMySqlLog:
            self.logPath = r'C:\borodin_admin\Институт\_ВКР\2022-06-14 Приложение\clustering-logs\error.log'
            self.startRowRegExp = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
        else:
            self.logPath = r'C:\borodin_admin\Институт\_ВКР\2022-06-14 Приложение\clustering-logs\application-2022-07-08.log'
            self.startRowRegExp = r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{3}Z)'

        # self.logPath = r'C:\borodin_admin\Институт\_ВКР\2022-06-14 Приложение\clustering-logs\fpm-fcgi-laravel-2022-07-25.log'
        # self.startRowRegExp = r'(\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\])'
