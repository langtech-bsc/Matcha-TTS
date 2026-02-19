""" from https://github.com/keithito/tacotron

Defines the set of symbols used in text input to the model.
"""
_pad = "_"
# _punctuation = ';:,.!?¡¿—…"«»“”() '  # added '()' Matcha v2
_punctuation = ';:,.!?¡¿—…"«»“”()- '  # added '()' and '-' Matcha graphemes
#_punctuation = ';:,.!?¡¿—…"«»“” ' # Matcha v1
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)
_letters_accented = "àáèéìíòóùú·üïöñ’#´"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
symbols_graphemes = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa) + list(_letters_accented)
# symbols_graphemes = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

print("symbols vocabulary size: ", len(symbols_graphemes))
# Special symbol ids
SPACE_ID = symbols.index(" ")
