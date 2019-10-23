import copy
import html
import re
from concurrent.futures import ProcessPoolExecutor
from itertools import islice, tee
from multiprocessing import cpu_count

import spacy
import unidecode
from num2words import num2words
from spacy.attrs import ORTH
from spacy.pipeline import SentenceSegmenter

lang_ext = {'pt': 'portuguese', 'en': 'english'}
num2words_lang_ext = {'pt': 'pt_br', 'en': 'en'}

NUMBERS_PATTERN = r'(-?\d+)(?:\.|,)?(\d*)'
CURRENCY_PATTERN = r'(R\$|\$|€|¥)\s*' + NUMBERS_PATTERN
ORDINAL_PATTERN = r'(\d+)\s*(?:os|o|º|st|nd|rd|th)'


def partition(iterable, parts):
    return (islice(it, i, None, parts) for i, it in enumerate(tee(iterable, parts)))


def rm_ellipsis(sent):
    """Remove ellipses"""
    return re.sub(r'\.\.\.', ' ', sent)


def rm_dashes(sent):
    """ Remove dashes surrounded by spaces (e.g. phrase - phrase) """
    return re.sub(r'\s-+\s', ' ', sent)


def rm_double_dashes(sent):
    """ Remove dashes between words with no spaces (e.g. word--word) and dashes between numbers """
    return re.sub(r'([A-Za-z0-9])\-\-([A-Za-z0-9])', r'\1 \2', sent)


def rm_dash_at_end(sent):
    """ Remove dash at a word end (e.g. three- to five-year) """
    return re.sub(r'(\w)-\s', r'\1 ', sent)


def rm_double_single_quotes(sent):
    """ Remove double single-quotes """
    sent = re.sub(r'\'\'\s', ' ', sent)
    return re.sub(r'\s\'\'', ' ', sent)


def rm_quotation_marks(sent):
    """ Remove single quotes used as quotation marks (e.g. some 'phrase in quotes')
    Remove double quotes used as quotation marks (e.g. some "phrase in quotes" or
    ``phrase in quotes'')
    """
    sent = re.sub(r"\s'([\w\s]+[\w])'\s", r' \1 ', sent)
    return re.sub(r'["“”‘’]', r' ', sent)


def unicode_transliteration(sent):
    """ convert unicode sent to ASCII that is the closest approximation.
    """
    return unidecode.unidecode(sent)


def cast_num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def rm_leading_spaces(sent):
    """ Remove extra spaces and leading spaces """
    sent = re.sub(r' +', ' ', sent)
    return re.sub(r'^ +', '', sent)


def rm_bet_brackets(sent):
    """ Remove everything between brackets """
    return re.sub(r'\(.*\)', '', sent)


def spec_add_spaces(sent):
    # Add spaces around / and # in `t`.
    sent = re.sub(r'([/#])', r' \1 ', sent)
    # Add space between numbers separate by dash
    return re.sub(r'([0-9]+)(-+)([0-9]+)', r'\1 \2 \3', sent)


def fix_html(x):
    "List of replacements from html strings in `x`."
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
            '<br />', "\n").replace('\\"', '"').replace(' @.@ ', '.').replace(' @-@ ', '-').replace(
                '\\', ' \\ ')
    return re.sub(r' +', ' ', html.unescape(x))


def text2lower(x):
    return x.lower()


def filter_by_vocab(vocab):
    def _filter(tokens):
        if vocab is None:
            return tokens

        return list(filter(lambda t: all(c in vocab for word in t for c in word), tokens))

    return _filter


def rm_punct(tokens):
    return [list(filter(lambda x: x not in '!,-.:;?', t)) for t in tokens]


class BaseTokenizer:
    "Basic class for a tokenizer function."

    def __init__(self, lang):
        self.lang = lang

    def tokenizer(self, text):
        return text.split(' ')

    def add_special_cases(self, tokens):
        pass

    def pre_rules(self, text):
        raise NotImplementedError

    def post_rules(self, tokens):
        raise NotImplementedError

    def __repr__(self):
        return 'BaseTokenizer'


class SpacyTokenizer(BaseTokenizer):
    "Wrapper around a spacy tokenizer to make it a `BaseTokenizer`."

    def __init__(self, lang):
        self.nlp = spacy.load(lang, disable=['parser', 'tagger', 'ner'])
        sbd = SentenceSegmenter(self.nlp.vocab)
        self.nlp.add_pipe(sbd)

    def tokenizer(self, text):
        return [[t.text for t in sent] for sents in filter(lambda x: len(x), text.split('\n'))
                for sent in self.nlp(sents.strip()).sents]

    def add_special_cases(self, tokens):
        for w in tokens:
            self.nlp.tokenizer.add_special_case(w, [{ORTH: w}])


class PTBRTokenizer(SpacyTokenizer):
    # symbol to word
    s2w = {
        'R$': (('real', 'reais'), ('centavo', 'centavos')),
        '$': (('dólar', 'dólares'), ('centavo', 'centavos')),
        '€': (('euro', 'euros'), ('centavo', 'centavos')),
        '¥': (('iene', 'ienes'), ('centavo', 'centavos'))
    }

    conversions = [
        (r'([0-9]+(?:\.[0-9]+)?)\s*(\%)', r'\1 por cento'),  # change % to `por cento`
    ]

    def __init__(self,
                 to_lower=True,
                 min_tokens=None,
                 vocab=None,
                 keep_numbers=False,
                 transliterate=False):
        self._pre_rules = [
            fix_html, spec_add_spaces, rm_bet_brackets, rm_ellipsis, rm_dashes, rm_double_dashes,
            rm_dash_at_end, rm_double_single_quotes, rm_quotation_marks, self.convert,
            rm_leading_spaces
        ]

        if not keep_numbers:
            self._pre_rules += [self.convert_currency, self.convert_numbers]

        if transliterate:
            self._pre_rules += [unicode_transliteration]

        if to_lower:
            self._pre_rules += [text2lower]

        self._post_rules = [
            rm_punct,
            lambda x: list(filter(lambda x: len(x) >= min_tokens if min_tokens else x, x)),
            filter_by_vocab(vocab)
        ]

        super().__init__('pt')

        from .lang.pt.tokenizer_exceptions import TOKENIZER_EXCEPTIONS
        for unicode_string, substrings in TOKENIZER_EXCEPTIONS.items():
            self.nlp.tokenizer.add_special_case(unicode_string, substrings)

    def convert(self, sent):
        for c in self.conversions:
            sent = re.sub(c[0], c[1], sent)
        return sent

    def convert_currency(self, sent):
        """ Convert currency without the cents
        """

        def _currency2words(match):
            symbol = match.group(1)
            integer_part = match.group(2)
            decimal_part = match.group(3)
            val = cast_num(integer_part + ('.' + decimal_part if decimal_part else ''))

            curr_name = self.s2w[symbol][0]
            cents_name = self.s2w[symbol][1]

            str_val = num2words(val, lang='pt_BR', to='currency')
            str_val = re.sub('(?:de\s)?rea(?:l|is)', curr_name[0 if int(integer_part) <= 1 else 1],
                             str_val)

            if decimal_part:
                str_val = re.sub('centavo(?:s)', cents_name[0 if decimal_part == '01' else 1],
                                 str_val)

            return str_val

        return re.sub(CURRENCY_PATTERN, _currency2words, sent)

    def convert_numbers(self, sent):

        sent = re.sub(
            ORDINAL_PATTERN,
            lambda x: num2words(cast_num(x.group(1).replace(',', '.')), lang='pt_BR', to='ordinal'),
            sent)

        try:
            sent = re.sub(NUMBERS_PATTERN,
                          lambda x: num2words(cast_num(x.group(1).replace(',', '.')), lang='pt_BR'),
                          sent)
        except OverflowError:
            pass
        finally:
            return sent

    def pre_rules(self, text):
        for rule in self._pre_rules:
            text = rule(text)
        return text.strip()

    def post_rules(self, tokens):
        for rule in self._post_rules:
            tokens = rule(tokens)
        return tokens

    def __repr__(self):
        res = f'Tokenizer {self.__class__.__name__} in PT_BR with the following rules:\n'
        for rule in self._pre_rules:
            res += f' - {rule.__name__}\n'
        for rule in self._post_rules:
            res += f' - {rule.__name__}\n'
        return res


class Tokenizer:
    """Tokenizer class.
    Based on fastai class
    Put together rules, a tokenizer function and a language to tokenize text with multiprocessing.
    """

    def __init__(self, tok, n_cpus=min(cpu_count() // 2, 1)):
        self.tok = tok
        self.n_cpus = n_cpus

    def process_text(self, text, tok=None):
        "Processe one text `t` with tokenizer `tok`."
        tok = tok or self.tok
        text = tok.pre_rules(text)
        tokens = tok.tokenizer(text)
        tokens = tok.post_rules(tokens)
        return tokens

    def _process_all(self, texts):
        "Process a list of `texts` in one process."
        tok = copy.deepcopy(self.tok)

        return [self.process_text(t, tok) for t in texts]

    def process_all(self, texts):
        "Process a list of `texts`."
        if self.n_cpus <= 1:
            return self._process_all(texts)

        with ProcessPoolExecutor(self.n_cpus) as e:
            return sum(e.map(self._process_all, partition(texts, self.n_cpus)), [])

    def __repr__(self):
        return repr(self.tok)
