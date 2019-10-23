""" Rewriting
"""
from spacy.symbols import ORTH

_exc = {
    "às": [{
        ORTH: "às"
    }],
    "ao": [{
        ORTH: "ao"
    }],
    "aos": [{
        ORTH: "aos"
    }],
    "àquele": [{
        ORTH: "àquele"
    }],
    "àquela": [{
        ORTH: "àquela"
    }],
    "àqueles": [{
        ORTH: "àqueles"
    }],
    "àquelas": [{
        ORTH: "àquelas"
    }],
    "àquilo": [{
        ORTH: "àquilo",
    }],
    "aonde": [{
        ORTH: "aonde"
    }]
}

# Contractions

_per_pron = ["ele", "ela", "eles", "elas"]
_dem_pron = [
    "este", "esta", "estes", "estas", "isto", "esse", "essa", "esses", "essas", "isso", "aquele",
    "aquela", "aqueles", "aquelas", "aquilo"
]
_und_pron = ["outro", "outra", "outros", "outras"]
_adv = ["aqui", "aí", "ali", "além"]

for orth in _per_pron + _dem_pron + _und_pron + _adv:
    _exc["d" + orth] = [{ORTH: "d" + orth}]

for orth in _per_pron + _dem_pron + _und_pron:
    _exc["n" + orth] = [{ORTH: "n" + orth}]

for orth, ext_orth in [("Adm.", "Administrador"), ("Dr.", "Doutor"), ("Dra.", "Doutora"),
                       ('a.c.', 'antes de cristo'), ('d.c.', 'depois de cristo'), ("e.g.",
                                                                                 "por exemplo"),
                       ("E.g.", None), ("E.G.", None), ("Gen.", None), ("Gov.", None),
                       ("i.e.", "isto é"), ("I.e.", None), ("I.E.", None), ("Jr.", None),
                       ("km.", 'quilômetro'), ("Ltd.", None), ("p.m.", None), ("Ph.D.", None),
                       ("Rep.", None), ("Rev.", None), ("Sen.", None), ("Sr.", "Senhor"),
                       ("Sra.", "Senhora"), ("vs.", "versus"), ("tel.", "telefone"),
                       ("pág.", "página"), ("pag.", None)]:
    _exc[orth] = [{ORTH: orth if ext_orth is None else ext_orth}]

TOKENIZER_EXCEPTIONS = _exc
