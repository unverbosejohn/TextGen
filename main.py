import nltk
import random
import collections
from collections import defaultdict
import re
import os
from string import Template
from io import StringIO
import argparse


# Message initialization
msg00 = 'Index Error. Please input an integer that is in the range of the corpus.'
msg01 = 'Please enter a number corresponding to one of the actions below.'
msg02 = 'Corpus statistics'
msg03 = 'All tokens: '
msg04 = 'Unique tokens: '
msg05 = 'Index Error. Please input a value that is not greater than the number of all bigrams.'
msg06 = 'Number of bigrams: '
msg07 = 'Key Error. The requested word is not in the model. Please input another word.'
msg08 = 'Please select the book to use as base corpus:'
msg09 = 'Please enter one of the values above:\n'
msg10 = 'all: All books. This may (and probably will) lead to strange results.'
msg11 = Template('$i. $d')
msg12 = Template('import: Import your own text file. Make sure it is in "$c"')
msg13 = 'This may take some time, are you sure? (y/n)'
msg14 = 'How many sentences to construct?'
msg15 = 'Welcome to TexGen, a text generation program.\n' \
        '' \
        'TextGen uses corpora from the Gutenberg library found in nltk.\n' \
        'You can find more corpora here: https://www.nltk.org/nltk_data/\n'
msg16 = 'What will we do today?'
msg17 = '1. Quick generation (from random corpus)'
msg18 = '2. Start custom generation'
msg19 = '3. Export generated text'
exit_msg = '4. Exit'
msg20 = 'Let\'s generate some text first, shall we?'
msg21 = 'File name:'
msg22 = Template('The log has been saved in $f')
msg23 = 'Corpus statistics'
msgline = '---------------------------------------------------------------'
# Message initialization end

# Logging and argument initialization
log_file = StringIO()
parser = argparse.ArgumentParser(description='This is a flashcard program.')
parser.add_argument('--import_from', default=None)
parser.add_argument('--export_to', default=None)
args = parser.parse_args()


def logger(msg, out=True):
    """
    :param msg: The message to log (input or output)
    :param out: Bool, defaults to True. If False, the message is treated as input from the user
    :return: Nothing by default, the msg if out = False
    """
    if not out:
        log_file.write(f'> {msg}\n')
        return
    log_file.write(f'{msg}\n')
    return msg


class Corpus:
    def __init__(self, f_name):
        self.f_name = f_name
        self.name = ''
        self.tokens = []
        self.bigrams, self.freq_bigram = self.build()
        self.heads, self.tails = self.get_bigrams()

    def get_bigrams(self):
        """
        Creates a bi(tri)gram from the corpus.
        :return: list: heads, list: tokens
        """
        heads = []
        tails = []
        last_index = len(self.tokens) - 1
        heads.append(' '.join([self.tokens[0], self.tokens[1]]))
        tails.append(self.tokens[2])
        for i in range(1, last_index - 2):
            heads.append(' '.join([self.tokens[i], self.tokens[i + 1]]))
            tails.append(self.tokens[i + 2])
        return heads, tails

    def build(self):
        """
        Bigram/Trigram constructor
        :return: dict: Heads, dict: Frequency of tails
        """
        f = open(self.f_name, 'r')
        self.name = f.readlines(1)[0].lstrip('[').rstrip(']\n')
        wst = nltk.WhitespaceTokenizer()
        for line in f:
            for word in wst.tokenize(line):
                self.tokens.append(word)
        f.close()
        heads, tails = self.get_bigrams()
        freq_bigram = defaultdict(dict)
        bigrams = defaultdict(list)
        for head, tail in zip(heads, tails):
            bigrams[head].append(tail)
        for key, value in bigrams.items():
            freq_bigram[key] = collections.Counter(value)
        return bigrams, freq_bigram


def is_end(test_tail):
    """
    Checks if a word (or a coupld of words) can be the end of a sentence
    :param test_tail: str: One or two words.
    :return: True/False
    """
    candidate = re.match(r'^[a-zA-Z]+ ?[a-zA-Z]*[!?.]+$', test_tail)
    if candidate:
        return True
    return False


def is_starting_head(test_head):
    """
    Checks if the test head (2 words) can be at the start of a sentence
    :param test_head: str: 2 words
    :return: True/False
    """
    candidate = re.match(r'[A-Z][a-z]*[^.\\!:,\';?] [a-z]+[^.\\!:,\';?]$', test_head)
    if candidate:
        return True
    return False


# def is_head(test_head):
#     candidate = re.match(r'^[A-Z][a-z]*[^.\\!:,\';?] [a-z]+[^.\\!:,\';?]$', test_head)
#     if candidate:
#         return True
#     return False


def get_tails(freq_dict, head):
    """
    Looks for tails for the given head
    :param freq_dict: The frequency of all heads/tails
    :param head: str: Two words
    :return: list: The most probable tails for the given head, or False.
    """
    try:
        if len(freq_dict[head]) == 0:
            return False
        return collections.Counter.most_common(freq_dict[head])
    except KeyError:
        return False


def get_tail(corpus, previous_words):
    """
    Tries to find the most suitable tail for the given head (previous_words)
    :param corpus: The corpus used.
    :param previous_words: str: Two words (the head)
    :return: str: A tail (one word)
    """
    previous_words = ' '.join(previous_words)
    try:
        candidate = random.choices(list(corpus.freq_bigram[previous_words].keys()),
                                   weights=list(corpus.freq_bigram[previous_words].values()))
        if candidate:
            return candidate[0]
    except IndexError:
        try:
            previous_word = previous_words[previous_words.index(' ') + 1:]
            for key, value in corpus.freq_bigram.items():
                try:
                    if key[key.index(' ') + 1:] == previous_word:
                        return random.choices(list(value.keys()), list(value.values()))[0]
                except IndexError:
                    continue
        except ValueError:
            pass
    return random.choice(corpus.heads)


def build_master_corpus(path):
    """
    Combines all txt files in path, in one big file.
    :param path: The path to the corpora directory
    :return: str: The path/filename.txt
    """
    master = open(path + 'all.txt', 'w', encoding='utf-8')
    master.write('[All books combined]\n')
    list_files = os.listdir(path)
    for file in list_files:
        if file != 'README':
            with open(path + file, 'r') as f:
                master.write(f.read())
                print(f'Copied {file} to master.')
    master.close()
    print(f'Done copying {len(list_files)} to all.txt')
    return path + 'all.txt'


def select_corpus(rand=False):
    """
    Prompts the user to select a corpus from a list. Will select randomly if rand=True
    :param rand: Defaults False, will select randomly if True
    :return: str: The path/filename.txt
    """
    cwd = os.getcwd()
    results = sorted(os.listdir(cwd + '\\corpora\\'))
    if rand:
        selection = results[random.randint(1, len(results))]
        logger(cwd + '\\corpora\\' + selection)
        return cwd + '\\corpora\\' + selection

    for i, result in enumerate(results):
        if result != 'README':
            with open(cwd + '\\corpora\\' + result, 'r') as f:
                title = str(f.readlines()[0].rstrip(']\n').replace('[', ''))
            print(msg11.substitute(i=i, d=title))
    # Removed option to construct a master corpus, with all the corpora combined.
    # print(msg10)
    print(msg12.substitute(c=cwd))
    print(msg08)
    while True:
        selection_no = input('> ')
        if selection_no == 'all':
            if 'all.txt' not in results:
                print(msg13)
                confirmation = input('> ')
                if confirmation in ['y', 'Y']:
                    return build_master_corpus(cwd + '\\corpora\\')
                else:
                    print(msg09)
                break
            else:
                return cwd + '\\corpora\\' + 'all.txt'
        elif selection_no == 'import':
            pass
            break
        else:
            try:
                selection_no = int(selection_no)
                break
            except ValueError:
                print(msg09)
            except IndexError:
                print(msg09)
    selection = results[selection_no]
    logger(cwd + '\\corpora\\' + selection)
    return cwd + '\\corpora\\' + selection


def pseudosentence(corpus, n):
    """
    Constructs a sentence of n words, using the corpus
    :param corpus: A corpus object
    :param n: int: How many words to make
    :return: A sentence of n words
    """
    starting_heads = []
    sentence = []
    for head in corpus.heads:
        if is_starting_head(head):
            starting_heads.append(head)
    word = random.choice(starting_heads)
    sentence.append(word[:word.index(' ')])
    sentence.append(word[word.index(' ') + 1:])
    for rep in range(n - 1):
        if rep == n - 2:
            while True:
                word = get_tail(corpus, word)
                if is_end(word):
                    sentence.append(str(word))
                    break
            return sentence
        last_words = sentence[len(sentence) - 2:]
        word = get_tail(corpus, last_words)
        sentence.append(word)
        if len(sentence) >= (n / 2) and is_end(word):
            return sentence
    return sentence


def log():
    """
    Will ask for a file name and export all generated text to that file.
    :return: Nothing
    """
    print(msg21)
    fname = input('> ')
    with open(fname, 'w', encoding='utf-8') as fn:
        print(log_file.getvalue(), file=fn)
    print(msg22.substitute(f=fname))


def start():
    print('{:^24s}'.format(msg15))
    while True:
        print(msgline)
        print(msg16)
        print(msg17)
        print(msg18)
        try:
            if text_list is not None:
                print(msg19)
        except NameError:
            pass
        print(exit_msg)
        selection = input('> ')
        try:
            selection = int(selection)
        except ValueError:
            pass

        if selection == 1:
            global random_corpus
            random_corpus = Corpus(select_corpus(rand=True))
            number_of_words = random.randint(8, 12)
            text_list = pseudosentence(random_corpus, number_of_words)
            print(f'From "{random_corpus.name}", around {number_of_words} words:')
            print(logger(' '.join(text_list)))
            print(msgline)
        elif selection == 2:
            text_file = select_corpus()
            break
        elif selection == 3:
            try:
                if text_list is not None:
                    log()
            except NameError:
                print(msg20)
        elif selection == 4:
            return
        else:
            print(msg01)
    print(msg14)
    while True:
        try:
            no_of_sentences = int(input('> '))
            break
        except ValueError:
            print(msg01)
    global corpus01
    corpus01 = Corpus(text_file)
    for _n in range(no_of_sentences):
        text_list = pseudosentence(corpus01, random.randint(8, 12))
        print(logger(' '.join(text_list)))
    print(msgline)


if __name__ == "__main__":
    start()
