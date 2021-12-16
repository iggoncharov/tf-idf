import math


class CountVectorizer:
    """
    Класс CountVectorizer, имеющий методы:
    fit_transform - Возвращает список списков где подсчитывается число слов
    в тексте из списка уникальных слов
    get_feature_names - возвращает список уникальных слов из всего корпуса
    """

    def __init__(self, lowercase = True):
        self.lowercase = lowercase
        self._vocabulary = []

    def get_feature_names(self) -> list[str]:
        '''
        Возвращает список уникальных слов из всего корпуса
        '''
        return list(self._vocabulary.keys())

    def fit_transform(self, corpus: list[str]) -> list[list[int]]:
        '''
        Функуия принимает на вход корпус(список) состоящий из текстов.
        Возвращает список списков где подсчитывается число слов
        в тексте из списка уникальных слов.
        if corpus = ['one two', 'one']
        out: [[1, 1], [1, 0]]
        '''
        if self.lowercase:
            corpus = [text.lower() for text in corpus]

        vocabulary = []
        corpus_list = []
        for text in corpus:
            words = text.split()
            corpus_list.append(words)
            for word in words:
                if word not in vocabulary:
                    vocabulary.append(word)

        self._vocabulary = {word: index for index, word in enumerate(vocabulary)}

        count_matrix = []
        for text in corpus_list:
            text_matrix = [0]*len(self._vocabulary)
            for word in text:
                text_matrix[self._vocabulary[word]] += 1
            count_matrix.append(text_matrix)
        return count_matrix


class TfIdfTransformer:
    """
    Класс имеющий методы:
    tf_transform - вычисляет сколько раз в тексте встречалось конкретное слово
    idf_transform - вычисляет значение равное log(число документов + 1)/(число документов в которых есть слово)) + 1
    fit_transform - вычисляет значение tf_transform * idf_transform
    """
    def __init__(self):
        self._tf = []
        self._idf = []

    def tf_transform(self, count_matrix: list[list[int]]) -> list[list[float]]:
        for ind, row in enumerate(count_matrix):
            self._tf.append([])
            total_cnt = sum(row)
            for element in row:
                self._tf[ind].append(round(element/total_cnt, 3))
        return self._tf

    def idf_transform(self, count_matrix: list[list[int]]) -> list[float]:
        total = len(count_matrix)
        idf_us = [0] * len(count_matrix[0])
        for text in count_matrix:
            for ind, word in enumerate(text):
                if word > 0:
                    idf_us[ind] += 1
        for i in idf_us:
            self._idf.append(round(math.log((total + 1) / (i + 1)) + 1, 1))
        return self._idf

    def fit_transform(self, count_matrix: list[list[int]]) -> list[list[float]]:
        self.tf_transform(count_matrix)
        self.idf_transform(count_matrix)
        tfidf = []
        for row, _ in enumerate(count_matrix):
            document_tfidf = []
            for col, _ in enumerate(count_matrix[0]):
                document_tfidf.append( round(self._tf[row][col] * self._idf[col], 3) )
            tfidf.append(document_tfidf)
        return tfidf


class TfIdfVectorizer(CountVectorizer):
    """
    Класс имеющий метод fit_transform, который принимает на вход корпус текстов
    и вычисляет tf_transform * idf_transform класса TfIdfTransformer
    """
    def __init__(self):
        super().__init__()
        self.tf_idf_transformer = TfIdfTransformer()

    def fit_transform(self, corpus: list[str]) -> list[list[float]]:
        matrix = super().fit_transform(corpus)
        tf_idf_matrix = self.tf_idf_transformer.fit_transform(matrix)
        return tf_idf_matrix


if __name__ == '__main__':
    corpus = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    ]
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)
    print('Словарь:')
    print(vectorizer.get_feature_names())
    print('fit_tranform:')
    print(count_matrix)
    vectorizer1 = TfIdfTransformer()
    tf_matrix = vectorizer1.tf_transform(count_matrix)
    print('Tf:')
    print(tf_matrix)
    idf_matrix = vectorizer1.idf_transform(count_matrix)
    print('Idf:')
    print(idf_matrix)
    transformer = TfIdfTransformer()
    tfidf_matrix = transformer.fit_transform(count_matrix)
    print('Tf_Idf:')
    print(tfidf_matrix)
    print()

    transformer1 = TfIdfVectorizer()
    fit = transformer1.fit_transform(corpus)
    print('Tf_Idf через TfIdfVectorizer')
    print(fit)