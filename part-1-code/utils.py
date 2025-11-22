import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    
    import random, re
    from nltk.corpus import wordnet
    from nltk import word_tokenize
    from nltk.tokenize.treebank import TreebankWordDetokenizer

    text = example["text"]
    words = word_tokenize(text)
    detok = TreebankWordDetokenizer()

    
    p_synonym = 0.35
    p_typo = 0.15
    p_delete = 0.12
    p_lower = 0.6
    p_shuffle = 0.15
    p_strip_punct = 0.6
    p_filler = 0.4

    fillers_start = [
        "Honestly,", "To be fair,", "If you ask me,", "Frankly speaking,", "In my honest opinion,"
    ]
    fillers_end = ["you know?", "just saying.", "if that makes sense.", "that's all."]

    new_tokens = []
    for w in words:
        lw = w.lower()

        # Random deletion (skip non-critical tokens)
        if random.random() < p_delete and len(w) > 3 and w.isalpha():
            continue

        new_w = w

        # Synonym replacement (adjectives, verbs, nouns)
        if random.random() < p_synonym and w.isalpha():
            syns = wordnet.synsets(lw)
            if syns:
                lemmas = [l.name().replace("_", " ") for l in syns[0].lemmas()]
                lemmas = [s for s in lemmas if s.lower() != lw and s.isalpha()]
                if lemmas:
                    new_w = random.choice(lemmas)

        # Typo (swap, drop, or double a character)
        if random.random() < p_typo and len(new_w) > 3 and new_w.isalpha():
            chars = list(new_w)
            choice = random.choice(["swap", "drop", "double"])
            if choice == "swap":
                i = random.randint(0, len(chars) - 2)
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
            elif choice == "drop":
                i = random.randint(0, len(chars) - 1)
                chars.pop(i)
            else:  # double
                i = random.randint(0, len(chars) - 1)
                chars.insert(i, chars[i])
            new_w = "".join(chars)

        new_tokens.append(new_w)

    # Random adjacent word swap (minor shuffle)
    if random.random() < p_shuffle and len(new_tokens) > 4:
        i = random.randint(0, len(new_tokens) - 2)
        new_tokens[i], new_tokens[i + 1] = new_tokens[i + 1], new_tokens[i]

    new_text = detok.detokenize(new_tokens)

    # Punctuation stripping
    if random.random() < p_strip_punct:
        new_text = re.sub(r"[^\w\s]", "", new_text)

    # Lowercasing (important for cased models)
    if random.random() < p_lower:
        new_text = new_text.lower()

    # Add neutral filler phrase (start or end)
    if random.random() < p_filler:
        if random.random() < 0.5:
            filler = random.choice(fillers_start)
            new_text = f"{filler} {new_text}"
        else:
            filler = random.choice(fillers_end)
            new_text = f"{new_text} {filler}"

    example["text"] = new_text.strip()
    return example
