import re
from wordcloud import WordCloud
from PIL import Image
from nltk import pos_tag as pt
import numpy as np
import matplotlib.pyplot as plt
import spacy
from nltk.stem.porter import *
import string


class Tokenizer:
    def __init__(self):
        self.stammer = PorterStemmer()
        self.nlp = spacy.load('en_core_web_sm')
        self.add_missing_stopwords(self.nlp)

    @staticmethod
    def row_gen():
        with open('crime_and_punishment.txt', encoding='utf-8-sig') as book:
            for row in book:
                yield row

    @staticmethod
    def add_missing_stopwords(nlp):
        extended_stopwords_list = ["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and",
                                   "any", "are", "aren", "aren't", "as", "at", "be", "because", "been", "before",
                                   "being",
                                   "below", "between", "both", "but", "by", "can", "couldn", "couldn't", "d", "de",
                                   "dr",
                                   "did", "didn",
                                   "didn't", "do", "does", "doesn", "doesn't", "doing", "don", "don't", "down",
                                   "during",
                                   "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn",
                                   "hasn't", "have", "haven", "haven't", "having", "he", "her", "here", "hers",
                                   "herself",
                                   "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isn't", "it",
                                   "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more",
                                   "most", "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not",
                                   "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours",
                                   "ourselves",
                                   "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's", "should",
                                   "should've", "shouldn", "shouldn't", "so", "some", "such", "t", "than", "that",
                                   "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these",
                                   "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve",
                                   "very",
                                   "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", "where",
                                   "which", "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn",
                                   "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
                                   "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's", "i'd",
                                   "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll", "that's", "there's",
                                   "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've",
                                   "what's",
                                   "when's", "where's", "who's", "why's", "would", "able", "abst", "accordance",
                                   "according", "accordingly", "across", "act", "actually", "added", "adj", "affected",
                                   "affecting", "affects", "afterwards", "ah", "almost", "alone", "along", "already",
                                   "also", "although", "always", "among", "amongst", "announce", "another", "anybody",
                                   "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere",
                                   "apparently",
                                   "approximately", "arent", "arise", "around", "aside", "ask", "asking", "auth",
                                   "available", "away", "awfully", "b", "back", "became", "become", "becomes",
                                   "becoming",
                                   "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "believe",
                                   "beside", "besides", "beyond", "biol", "brief", "briefly", "c", "ca", "came",
                                   "cannot",
                                   "can't", "cause", "causes", "certain", "certainly", "co", "com", "come", "comes",
                                   "contain", "containing", "contains", "couldnt", "date", "different", "done",
                                   "downwards",
                                   "due", "e", "eh", "ed", "edu", "effect", "eg", "eight", "eighty", "either", "else",
                                   "elsewhere", "end", "ending", "enough", "especially", "et", "etc", "even", "ever",
                                   "every", "everybody", "everyone", "everything", "everywhere", "ex", "except", "f",
                                   "far",
                                   "ff", "fifth", "first", "five", "fix", "followed", "following", "follows", "former",
                                   "formerly", "forth", "found", "four", "furthermore", "g", "gave", "get", "gets",
                                   "getting", "give", "given", "gives", "giving", "go", "goes", "gone", "got", "gotten",
                                   "h", "hm", "ha", "happens", "hardly", "hed", "hence", "hereafter", "hereby",
                                   "herein",
                                   "heres",
                                   "hereupon", "hes", "hi", "hid", "hither", "home", "howbeit", "however", "hundred",
                                   "id",
                                   "ie", "im", "immediate", "immediately", "importance", "important", "inc", "indeed",
                                   "index", "information", "instead", "invention", "inward", "itd", "it'll", "j", "k",
                                   "keep", "keeps", "kept", "kg", "km", "know", "known", "knows", "l", "la", "largely",
                                   "last",
                                   "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "lets",
                                   "like",
                                   "liked", "likely", "line", "little", "'ll", "look", "looking", "looks", "ltd",
                                   "made",
                                   "mainly", "make", "makes", "many", "may", "maybe", "mean", "means", "meantime",
                                   "meanwhile", "merely", "mg", "might", "million", "miss", "ml", "moreover", "mostly",
                                   "mr", "mrs", "much", "mug", "must", "n", "na", "nt", "name", "namely", "nay", "nd",
                                   "near",
                                   "nearly", "necessarily", "necessary", "need", "needs", "neither", "never",
                                   "nevertheless", "new", "next", "nine", "ninety", "nobody", "non", "none",
                                   "nonetheless",
                                   "noone", "normally", "nos", "noted", "nothing", "nowhere", "obtain", "obtained",
                                   "obviously", "often", "oh", "ok", "okay", "old", "omitted", "one", "ones", "onto",
                                   "ord",
                                   "others", "otherwise", "outside", "overall", "owing", "p", "page", "pages", "part",
                                   "particular", "particularly", "past", "per", "perhaps", "placed", "please", "plus",
                                   "poorly", "possible", "possibly", "potentially", "pp", "predominantly", "present",
                                   "previously", "primarily", "probably", "promptly", "proud", "provides", "put", "q",
                                   "que", "quickly", "quite", "qv", "r", "ran", "rather", "rd", "readily", "really",
                                   "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related",
                                   "relatively", "research", "respectively", "resulted", "resulting", "results",
                                   "right",
                                   "run", "said", "saw", "say", "saying", "says", "sec", "section", "see", "seeing",
                                   "seem",
                                   "seemed", "seeming", "seems", "seen", "self", "selves", "sent", "seven", "several",
                                   "shall", "shed", "shes", "show", "showed", "shown", "showns", "shows", "significant",
                                   "significantly", "similar", "similarly", "since", "six", "slightly", "somebody",
                                   "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat",
                                   "somewhere", "soon", "sorry", "specifically", "specified", "specify", "specifying",
                                   "still", "stop", "strongly", "sub", "substantially", "successfully", "sufficiently",
                                   "suggest", "sup", "sure", "take", "taken", "taking", "tell", "tends", "th", "thank",
                                   "thanks", "thanx", "thats", "that've", "thence", "thereafter", "thereby", "thered",
                                   "therefore", "therein", "there'll", "thereof", "therere", "theres", "thereto",
                                   "thereupon", "there've", "theyd", "theyre", "think", "thou", "though", "thoughh",
                                   "thousand", "throug", "throughout", "thru", "thus", "til", "tip", "together", "took",
                                   "toward", "towards", "tried", "tries", "truly", "try", "trying", "ts", "twice",
                                   "two",
                                   "u", "un", "unfortunately", "unless", "unlike", "unlikely", "unto", "upon", "ups",
                                   "us",
                                   "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "v",
                                   "value", "various", "'ve", "via", "viz", "vol", "vols", "vs", "w", "wo", "want",
                                   "wants",
                                   "wasnt", "way", "wed", "welcome", "went", "werent", "whatever", "what'll", "whats",
                                   "whence", "whenever", "whereafter", "whereas", "whereby", "wherein", "wheres",
                                   "whereupon", "wherever", "whether", "whim", "whither", "whod", "whoever", "whole",
                                   "who'll", "whomever", "whos", "whose", "widely", "willing", "wish", "within",
                                   "without",
                                   "wont", "words", "world", "wouldnt", "www", "x", "yes", "yet", "ye", "youd", "youre",
                                   "z",
                                   "zero", "a's", "ain't", "allow", "allows", "apart", "appear", "appreciate",
                                   "appropriate", "associated", "best", "better", "c'mon", "c's", "cant", "changes",
                                   "clearly", "concerning", "consequently", "consider", "considering", "corresponding",
                                   "course", "currently", "definitely", "described", "despite", "entirely", "exactly",
                                   "example", "going", "greetings", "hello", "help", "hopefully", "ignored", "inasmuch",
                                   "indicate", "indicated", "indicates", "inner", "insofar", "it'd", "keep", "keeps",
                                   "novel", "presumably", "reasonably", "second", "secondly", "sensible", "serious",
                                   "seriously", "sure", "t's", "third", "thorough", "thoroughly", "three", "well",
                                   "wonder"]
        for stop_word in extended_stopwords_list:
            nlp.vocab[stop_word].is_stop = True

    # @staticmethod
    # def remove_punctuation(row):
    #     return row.translate(str.maketrans('', '', string.punctuation + '’‘' + '“' + '”'))

    @staticmethod
    def process_token(token):
        if token.is_punct:
            return None
        stripped_tok = token.string.strip().lower()
        return stripped_tok if len(stripped_tok) > 1 else None

    def tokenize(self, remove_stop_words=False, stem_words=False):
        occurrences = {}
        for row in self.row_gen():
            for token in self.nlp.tokenizer(row):
                word = self.process_token(token)
                if word and word:
                    if remove_stop_words and self.nlp.vocab[word].is_stop:
                        continue
                    if stem_words:
                        word = self.stammer.stem(word)
                    if word not in occurrences:
                        occurrences[word] = 0
                    occurrences[word] += 1
        return occurrences


def print_top_20_tokens(occur_dict):
    sorted_dict = {k: v for k, v in sorted(occur_dict.items(), key=lambda item: item[1])[::-1]}
    print([token for token in sorted_dict.keys()][:20])


def plot_results(occur_dict, title='results'):
    plt.figure(title)
    plt.title(title)
    plt.xlabel('log(rank)')
    plt.ylabel('log(frequency)')
    occurrences = list(occur_dict.values())
    occurrences.sort()
    log_rank = np.log(np.arange(1, len(occurrences) + 1))
    log_freq = np.log(occurrences[::-1])
    plt.plot(log_rank, log_freq)
    plt.show()


def get_all_words():
    """
    breaks rows of text (using a generator) into structually significant piece
    example - 'I, don't think it's safe breaks' to ['I', ',', 'do', 'n\'t', 'think', 'it', '\'s', 'safe']
    :returns: a numpy array of strings
    """
    res = []
    sp = spacy.load('en_core_web_sm')
    for row in Tokenizer.row_gen():
        line = row.strip()
        for token in sp.tokenizer(line):
            w = token.string.strip(' ')
            if not w:
                continue
            res.append(w)
    return res


def process_word(token):
    if token.is_punct:
        return None
    stripped_tok = token.string.strip()
    return stripped_tok if len(stripped_tok) > 1 else None


def get_POS_tags(words):
    """
    uses the NLTK library to take a list of strings and give them PoS tags
    :param words: np array of shape (N,) of strings
    :returns: np array of shame (2,N) where res[0, :] is the original list of strings and res[1, :] is the list of tags
    """
    sp = spacy.load('en_core_web_sm')
    res = pt(words)
    return remove_punct(np.array(res).T)


def remove_punct(pos_tags):
    """
    removes punctuation for pos tagging
    """
    index = np.char.strip(pos_tags[0, :], string.punctuation + '”' + '’' + '“' + '‘')
    # these single punctuations aren't the typical " and ' seen on the keyboard and appear in the pdf
    # they are removed "manually"
    locs = []
    for i in range(index.shape[0]):
        if len(index[i]) != 0:
            locs.append(i)
    res = np.row_stack((pos_tags[0, :][locs], pos_tags[1, :][locs]))
    return res


def make_dict(pos_tags):
    """
    given a list of words and positional arguments returns a dict of each words use
    :param pos_tags: a np array of shape (2,N) where 1st row is strings and 2nd row is their PoS tag
    :returns: a dict with keys being words (lower case) and values are sets of PoS tags of the key word
    """
    res = {}
    for i in np.arange(pos_tags.shape[1]):
        key = pos_tags[0, i].strip()
        if not key:
            continue
        res.setdefault(key.lower(), set())
        res[key.lower()].add(pos_tags[1, i])
    return res


def tag_adj(pos_tags):
    """
    :param pos_tags: a np array of shape (2,N) where 1st row is strings and 2nd row is their PoS tag
    :returns: an np array of shape (M,) of the indexes of adjectives
    """
    data = pos_tags[1, :]
    index = np.where(np.char.startswith(data, 'JJ') == 1)[0]
    return np.array(index)


def find_adj_noun(pos_tags):
    """
    :param pos_tags: a np array of shape (2,N) where 1st row is strings and 2nd row is their PoS tag
    :param adj_ind: an np array of shape (M,) of the indexes of adjectives
    :returns: a set of all adjective + noun phrases (numltiple adjectives followed by multiple nouns)
    """
    i = 0
    res = set()
    adj_ind = tag_adj(pos_tags)
    while i < adj_ind.shape[0]:
        j = 1
        flag = False
        start = adj_ind[i]
        if start == pos_tags.shape[0] - 1:
            break
        while True:
            while pos_tags[1, adj_ind[i] + 1].startswith('JJ'):
                i += 1
                if adj_ind[i] + j == pos_tags.shape[0] - 1:
                    flag = True
                    break
            if flag:
                break
            while pos_tags[1, adj_ind[i] + j].startswith('NN'):
                j += 1
                if adj_ind[i] + j >= pos_tags.shape[0] - 1:
                    break
            if j > 1:
                txt = ""
                for k in range(start, adj_ind[i] + j):
                    txt += pos_tags[0, k] + " "
                res.add(txt[:len(txt) - 1])
            i += 1
            break
    return res, count_uniques(res)


def count_uniques(str_list):
    """
    counts occurrences of item in a list and returns a dict of the results
    """
    res = {}
    for w in str_list:
        res.setdefault(w.lower(), 0)
        res[w.lower()] += 1
    return res


def get_homographs(pos_dict):
    """
    Q4- p.g
    :param pos_dict: a dictionary of the text, where keys are the words and
            the pos tags are the values.
    :return: Outputs the text's 10 top and 10 bottoms homographs as 2
            lists. each list contains of sub list where the first element is the word,
            and the second element is it's POS list.
    """
    pd = pos_dict.copy()
    highest, lowest = [], []
    for i in range(10):
        high = max(pd.keys(), key=(lambda k: len(pd[k])))
        highest.append([high, pd[high]])
        del pd[high]
        low = min(pd.keys(), key=(lambda k: len(pd[k])))
        lowest.append([low, pd[low]])
        del pd[low]
    return [highest, lowest]


def tag_proper_noun(pos_tags):
    """
    :param pos_tags: a np array of shape (2,N) where 1st row is strings and 2nd row is their PoS tag
    :returns: an np array of shape (M,) of the indexes of proper nouns
    """
    data = pos_tags[1, :]
    index = np.where(np.char.startswith(data, 'NNP') == 1)[0]
    return np.array(index)


def make_nnp_dict(pos_tags):
    """
    :param pos_tags: a np array of shape (2,N) where 1st row is strings and 2nd row is their PoS tag
    :returns: dict with keys being words  (in lower case) and values are number of occurances as proper nouns
    """
    index = tag_proper_noun(pos_tags)
    words = pos_tags[0, :][index]
    res = {}
    for w in words:
        res.setdefault(w.lower(), 0)
        res[w.lower()] += 1
    return res


def word_cloud(pos_tags, mask_path):
    """
    Q4- p.h
    creates & saves a word cloud out of the text's NNP or NNPS words.
    :param pos_dict: a dictionary of the text, where keys are the NNP or NNPS words and
        the values are their occurrences in the text.
    :param mask_path: the mask's path
    """
    new_dict = make_nnp_dict(pos_tags)
    mask_im = np.array(Image.open(mask_path))
    cloud = WordCloud(background_color="black", relative_scaling=0.5,
                      mask=mask_im).generate_from_frequencies(new_dict)
    cloud.to_file("wordCloud.png")


def find_cons_rep_words():
    """
    Q4- p.i:
    finds all two consecutive repeated words in the text,
    and the row from which they came.
    :return: returns a list of lists, where every inner list holds
    the consecutive word (string) at idx 0, and the row (as string) at idx 1.
    """
    res = []
    last_word = ""
    for row in Tokenizer.row_gen():
        row = last_word + " " + row
        a = re.findall(r'\b([A-Za-z]+)\W?\s+\1\b', row)
        if a:
            res.append([a[0], row])

        # maintain last word:
        s_row = row.split()
        if s_row:
            last_word = s_row[-1]
        else:
            last_word = ""

    return res


if __name__ == "__main__":
    tok = Tokenizer()
    occur_dict = tok.tokenize()
    plot_results(occur_dict, 'Tokenizing results, with stopwords, no stemming.')
    print_top_20_tokens(occur_dict)
    occur_dict = tok.tokenize(remove_stop_words=True)
    plot_results(occur_dict, 'Tokenizing results, no stopwords, no stemming.')
    print_top_20_tokens(occur_dict)
    occur_dict = tok.tokenize(remove_stop_words=True, stem_words=True)
    plot_results(occur_dict, 'Tokenizing results, no stopwords, with stemming.')
    print_top_20_tokens(occur_dict)


    words = get_all_words()
    tags = get_POS_tags(words)
    adj = tag_adj(tags)
    an_list, an_dict = find_adj_noun(tags)
    print_top_20_tokens(an_dict)
    plot_results(an_dict, 'Tokenizing results for adjective and noun combinations')

    pos_dict = make_dict(tags)

    # Q4 p.i:
    w = find_cons_rep_words()
    print("rep words:\n")
    print(np.array(w)[:, 0])
    for i in w:
        print(i)
    print(len(w))

    # Q4 p.g:
    h, l = get_homographs(pos_dict)
    print("\nhighest homo:")
    for i in h:
        print(i)
    print("\nlowest homo:")
    for i in l:
        print(i)

    # Q4 p.h:
    img_path = "man_mask.png"
    word_cloud(tags, img_path)
