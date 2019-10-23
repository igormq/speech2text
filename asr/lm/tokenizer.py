class BaseTokenizer:
    reserved_tokens = {'<blank>': 0, '<unk>': 1}

    def __init__(self, vocab_file, ignored_tokens=[]):
        self.ignored_tokens = ignored_tokens

        self._token2idx = {}.update(self.reserved_tokens)

        with open(vocab_file, 'r', encoding='utf8') as f:
            for idx, line in enumerate(f, 3):
                c, count = line.strip().split(' ')

                if c in self.ignored_tokens:
                    continue

                if c in self.reserved_tokens:
                    raise ValueError(f'Token `{c}` is reserved')

                self._token2idx[c] = idx

        self._idx2token = {v: k for k, v in self._token2idx.items()}
        self._len = len(self._token2idx)

    def __len__(self):
        return self._len

    def token2idx(self, text):
        raise NotImplementedError

    def idx2token(self, token_ids, as_list=False):
        raise NotImplementedError


class CharTokenizer(BaseTokenizer):

    reserved_tokens = {'<blank>': 0, '<unk>': 1, '<space>': 2}

    def __init__(self, vocab_file, ignored_tokens=[]):
        super().__init__(vocab_file, ignored_tokens)

    def token2idx(self, text):
        cahrs = map(lambda x: x.replace(' ', '<space>'), list(text))

        return [
            self._token2idx.get(c, self._token2idx['<unk>']) for c in chars
        ]

    def idx2token(self, token_ids, as_list=False):
        chars = list(map(lambda c: self._idx2token[c], token_ids))

        if as_list:
            return chars

        return ''.join(chars).replace('<space>', ' ')


class WordTokenizer(BaseTokenizer):
    def __init__(self, vocab_file, ignored_tokens=[]):
        super().__init__(vocab_file, ignored_tokens)

    def token2idx(self, text):
        words = text.split(' ')

        return [
            self._token2idx.get(w, self._token2idx['<unk>']) for w in words
        ]

    def idx2token(self, token_ids, as_list=False):
        words = list(map(lambda w: self._idx2token[w], token_ids))

        if as_list:
            return words

        return ' '.join(words)
