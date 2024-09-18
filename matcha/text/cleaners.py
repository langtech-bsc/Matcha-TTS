""" from https://github.com/keithito/tacotron

Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
"""

import logging
import re

import phonemizer
import piper_phonemize
from unidecode import unidecode

# To avoid excessive logging we set the log level of the phonemizer package to Critical
critical_logger = logging.getLogger("phonemizer")
critical_logger.setLevel(logging.CRITICAL)

# Intializing the phonemizer globally significantly reduces the speed
# now the phonemizer is not initialising at every call
# Might be less flexible, but it is much-much faster
global_phonemizer = phonemizer.backend.EspeakBackend(
    language="en-us",
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
    logger=critical_logger,
)

global_phonemizer_cat = phonemizer.backend.EspeakBackend(
    language="ca",  # 'ca' if catalan
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
    logger=critical_logger,
)

global_phonemizer_es_lat = phonemizer.backend.EspeakBackend(
    language="es-419",  # spanish latam
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
    logger=critical_logger,
)

global_phonemizer_es = phonemizer.backend.EspeakBackend(
    language="es",  # spanish
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
    logger=critical_logger,
)

global_phonemizer_cat_bal = phonemizer.backend.EspeakBackend(
    language="ca-ba",
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
    logger=critical_logger,
)

global_phonemizer_cat_occ = phonemizer.backend.EspeakBackend(
    language="ca-nw",
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
    logger=critical_logger,
)

global_phonemizer_cat_val = phonemizer.backend.EspeakBackend(
    language="ca-va",
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
    logger=critical_logger,
)

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]

# Regular expression to match either <cat>...</cat> or <es>...</es>
cs_tags = re.compile(r'<(cat|es)>(.*?)<\/\1>')

def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def none_cleaners(text):
    return text


def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners2(text):
    """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
    text = text.replace('\u0303', '')
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = global_phonemizer.phonemize([text], strip=True, njobs=1)[0]
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def catalan_cleaners(text):
    """Pipeline for Catalan text, including abbreviation expansion. + punctuation + stress"""
    # text = convert_to_ascii(text)
    text = lowercase(text)
    # text = expand_abbreviations(text)
    phonemes = global_phonemizer_cat.phonemize([text], strip=True, njobs=1)[0]
    phonemes = collapse_whitespace(phonemes)
    # print(phonemes)  # check punctuations!!
    return phonemes

def catalan_balear_cleaners(text):
    """Pipeline for Catalan text, including abbreviation expansion. + punctuation + stress"""
    # text = convert_to_ascii(text)
    text = lowercase(text)
    # text = expand_abbreviations(text)
    phonemes = global_phonemizer_cat_bal.phonemize([text], strip=True, njobs=1)[0]
    phonemes = collapse_whitespace(phonemes)
    # print(phonemes)  # check punctuations!!
    return phonemes

def catalan_occidental_cleaners(text):
    """Pipeline for Catalan text, including abbreviation expansion. + punctuation + stress"""
    # text = convert_to_ascii(text)
    text = lowercase(text)
    # text = expand_abbreviations(text)
    phonemes = global_phonemizer_cat_occ.phonemize([text], strip=True, njobs=1)[0]
    phonemes = collapse_whitespace(phonemes)
    # print(phonemes)  # check punctuations!!
    return phonemes

def catalan_valencia_cleaners(text):
    """Pipeline for Catalan text, including abbreviation expansion. + punctuation + stress"""
    # text = convert_to_ascii(text)
    text = lowercase(text)
    # text = expand_abbreviations(text)
    phonemes = global_phonemizer_cat_val.phonemize([text], strip=True, njobs=1)[0]
    phonemes = collapse_whitespace(phonemes)
    # print(phonemes)  # check punctuations!!
    return phonemes

def spanish_latam_cleaners(text):
    """Pipeline for spanish text + punctuation + stress"""
    # text = convert_to_ascii(text)
    text = lowercase(text)
    # text = expand_abbreviations(text)
    phonemes = global_phonemizer_es_lat.phonemize([text], strip=True, njobs=1)[0]
    phonemes = collapse_whitespace(phonemes)
    # print(phonemes)  # check punctuations!!
    return phonemes

def spanish_cleaners(text):
    """Pipeline for spanish text + punctuation + stress"""
    # text = convert_to_ascii(text)
    text = lowercase(text)
    # text = expand_abbreviations(text)
    phonemes = global_phonemizer_es.phonemize([text], strip=True, njobs=1)[0]
    phonemes = collapse_whitespace(phonemes)
    # print(phonemes)  # check punctuations!!
    return phonemes

def code_switch_cleaners(text):
    # Find all matches
    matches = cs_tags.findall(text)

    # Process the matches
    ordered_sentences = []
    for lang, sentence in matches:
        ordered_sentences.append((lang, sentence))

    phonemes_all = ""

    for lang, sentence in ordered_sentences:
        if lang == "es":
                text = lowercase(sentence)
                phonemes = global_phonemizer_es.phonemize([text], strip=True, njobs=1)[0]
                phonemes = collapse_whitespace(phonemes)
                phonemes_all += phonemes
        elif lang == "cat":
                text = lowercase(sentence)
                phonemes = global_phonemizer_cat.phonemize([text], strip=True, njobs=1)[0]
                phonemes = collapse_whitespace(phonemes)
                phonemes_all += phonemes
    
    return phonemes_all

def english_cleaners_piper(text):
    """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = "".join(piper_phonemize.phonemize_espeak(text=text, voice="en-US")[0])
    phonemes = collapse_whitespace(phonemes)
    return phonemes
