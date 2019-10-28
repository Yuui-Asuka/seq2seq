import jieba
import collections

class Processor():
    __PAD__ = 0
    __GO__ = 1
    __EOS__ = 2
    __UNK__ = 3
    vocab = ['__PAD__', '__UNK__', '__GO__', '__EOS__']

    def __init__(self):
        self.encoderFile = "./question.txt"
        self.decoderFile = './answer.txt'
        self.stopwordsFile = "./tf_data/stopwords.dat"

    def cut_sents(self, sents, vocab):
        words = jieba.lcut(sents.strip())
        vocab.extend(words)
        return ' '.join(words)

    def wordToVocabulary(self, questions, answers, question_segment,
                         answer_segment, vocabFile):
        # stopwords = [i.strip() for i in open(self.stopwordsFile).readlines()]
        # print(stopwords)
        # exit()
        vocabulary = []
        question_sege = open(question_segment, "w",encoding = 'utf-8')
        answer_sege = open(answer_segment, 'w', encoding='utf-8')
        jieba.load_userdict('./tf_data_new/dictionary.txt')
        with open(questions, 'r',encoding = 'utf-8') as q,\
             open(answers, 'r', encoding='utf-8') as a:
            for sent in q.readlines():
                words = self.cut_sents(sent, vocabulary)
                question_sege.write(words + '\n')
            for sent in a.readlines():
                words = self.cut_sents(sent, vocabulary)
                answer_sege.write(words + '\n')
        question_sege.close()
        answer_sege.close()
        # 去重并存入词典
        vocabulary_ = collections.Counter(vocabulary)
        vocabulary = set(vocabulary)
        for key, value in vocabulary_.items():
            if value < 4:
                vocabulary.remove(key)
        #_vocabulary = list(set(vocabulary))
        #_vocabulary.sort(key=vocabulary.index)
        #_vocabulary = self.vocab + _vocabulary
        vocab_file = open(vocabFile, "w", encoding='utf-8')
        for vocab in self.vocab:
            vocab_file.write(vocab + '\n')
        for index, word in enumerate(vocabulary):
            vocab_file.write(word + "\n")
        vocab_file.close()

    def toVec(self, segementFile, vocabFile, doneFile):
        word_dicts = {}
        #vec = []
        with open(vocabFile, "r",encoding = 'utf-8') as dict_f:
            for index, word in enumerate(dict_f.readlines()):
                word_dicts[word.strip()] = index

        f = open(doneFile, "w")
        with open(segementFile, "r",encoding = 'utf-8') as sege_f:
            for sent in sege_f.readlines():
                sents = [i.strip() for i in sent.split(" ")[:-1]]
                #vec.extend(sents)
                for word in sents:
                    f.write(str(word_dicts.get(word, self.__UNK__)) + " ")
                f.write("\n")
        f.close()

    def run(self):
        # 获得字典
        # self.wordToVocabulary(
        #     self.encoderFile, './tf_data_new/enc.vocab', './tf_data_new/enc.segement')
        # self.wordToVocabulary(
        #     self.decoderFile, './tf_data_new/dec.vocab', './tf_data_new/dec.segement')
        # 转向量
        self.wordToVocabulary(self.encoderFile, self.decoderFile, './tf_data_new/enc.segement',
                              './tf_data_new/dec.segement', './tf_data_new/en_de_vocabs')
        self.toVec("./tf_data_new/enc.segement",
                   "./tf_data_new/en_de_vocabs",
                   "./tf_data_new/enc.vec")
        self.toVec("./tf_data_new/dec.segement",
                   "./tf_data_new/en_de_vocabs",
                   "./tf_data_new/dec.vec")

process = Processor()
process.run()
