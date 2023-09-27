# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

import math
import random
import re
import string

from iso639 import Lang

# for the first time you use wordnet
# import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet, stopwords
from typing import Union

__VALID_CHARS = string.ascii_lowercase + " "
__RANDOM_SEED = 12

# FIXME: Inconsistency issue with the augmented sentences generated \
#        and the number of augmented sentences when 0 < n < 1.
random.seed(__RANDOM_SEED)


def get_only_chars(line: str) -> str:
    """
    Remove invalid characters from a line.

    Args:
        line (str): input line.

    Returns:
        clean_line (str): line with invalid characters removed.
    """
    line = (
        line.replace("’", "")
        .replace("'", "")
        .replace("-", " ")
        .replace("\t", " ")
        .replace("\n", " ")
        .lower()
    )

    clean_line = "".join(char if (char in __VALID_CHARS) else " " for char in line)

    clean_line = re.sub(" +", " ", clean_line).strip()  # delete extra spaces
    return clean_line


########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################


def synonym_replacement(words: list, n: int, language: str) -> list:
    """
    Replace n words in the sentence with synonyms from wordnet.

    Args:
        words (list): list of words in the sentence.
        n (int): number of words to be replaced.
        language (str): language of the input sentence e.g. eng, deu, ind, etc.

    Returns:
        new_words (list): list of words with n words replaced with synonyms.
    """
    new_words = words.copy()
    stop_words = stopwords.words(Lang(language).name.lower())
    random_word_list = list(set(word for word in words if word not in stop_words))
    random.shuffle(random_word_list)
    num_replaced = 0

    for random_word in random_word_list:
        synonyms = get_synonyms(random_word, language=language)

        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1

        if num_replaced >= n:  # only replace up to n words
            break

    # this is stupid but we need it, trust me
    sentence = " ".join(new_words)
    new_words = sentence.split(" ")

    return new_words


def get_synonyms(word: str, language: str) -> list:
    """
    Get synonyms of a word from wordnet.

    Args:
        word (str): input word.
        language (str): language of the input sentence e.g. eng, deu, ind, etc.

    Returns:
        synonyms (list): list of synonyms of the input word.
    """
    synonyms = set()
    for syn in wordnet.synsets(word, lang=language):
        for lemma in syn.lemmas(lang=language):
            synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in __VALID_CHARS])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################


def random_deletion(words: list, probability: float) -> list:
    """
    Randomly delete words from the sentence with probability p.

    Args:
        words (list): list of words in the sentence.
        probability (float): probability of deleting a word.

    Returns:
        new_words (list): list of words with some words deleted.
    """
    # obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    # randomly delete words with probability p
    new_words = [word for word in words if random.uniform(0, 1) > probability]

    # if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]

    return new_words


########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################


def random_swap(words: list, n: int) -> list:
    """
    Randomly swap two words in the sentence n times.

    Args:
        words (list): list of words in the sentence.
        n (int): number of times to swap words.

    Returns:
        new_words (list): list of words with n pairs of words swapped.
    """
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words


def swap_word(new_words: list) -> list:
    """
    Randomly swap two words in the sentence.

    Args:
        new_words (list): list of words in the sentence.

    Returns:
        new_words (list): list of words with two words swapped.
    """
    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words) - 1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = (
        new_words[random_idx_2],
        new_words[random_idx_1],
    )
    return new_words


########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################


def random_insertion(words: list, n: int, language: str) -> list:
    """
    Randomly insert n words into the sentence.

    Args:
        words (list): list of words in the sentence.
        n (int): number of words to be inserted.
        language (str): language of the input sentence e.g. eng, deu, ind, etc.

    Returns:
        new_words (list): list of words with n words inserted.
    """
    new_words = words.copy()
    for _ in range(n):
        try:
            add_word(new_words, language=language)
        except ValueError as e:
            print(f"{new_words=}")
            raise e
    return new_words


def add_word(new_words: list, language: str):
    """
    Randomly insert a word into the sentence.

    Args:
        new_words (list): list of words in the sentence.
        language (str): language of the input sentence e.g. eng, deu, ind, etc.

    Returns:
        new_words (list): list of words with a word inserted.
    """
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words) - 1)]
        synonyms = get_synonyms(random_word, language=language)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words) - 1)
    new_words.insert(random_idx, random_synonym)


########################################################################
# main data augmentation function
########################################################################


def get_num_per_technique(num_aug: Union[int, float], flags: list) -> list:
    """
    Get number of augmented sentences per augmentation technique.

    Args:
        num_aug (int or float): number of augmented sentences to generate per original sentence.
        flags (list): list of flags for each augmentation technique.

    Returns:
        num_new_per_technique (list): number of augmented sentences per augmentation technique.
    """
    num_methods = 4

    if not flags:
        flags = [1] * num_methods

    # num_new_per_technique = [int(num_aug / 4)] * 4
    num_new_per_technique = [
        int(num_aug / num_methods) if flags[pos] else 0 for pos in range(num_methods)
    ]

    # If there's a remainder, add 1 to a random technique
    # for _ in range(math.ceil(num_aug % 4)):
    #     num_new_per_technique[random.randint(0, 4 - 1)] += 1
    remainder = math.ceil(num_aug % num_methods)
    while remainder > 0:
        position = random.randint(0, num_methods - 1)
        if flags[position]:
            num_new_per_technique[position] += 1
            remainder -= 1

    return num_new_per_technique


def eda(
    sentence: str,
    language: str = "eng",
    num_aug: Union[int, float] = 9,
    alpha_sr: float = 0.1,
    alpha_ri: float = 0.1,
    alpha_rs: float = 0.1,
    alpha_rd: float = 0.1,
    randomize: bool = False,
) -> list:
    """
    Perform EDA on sentence.

    Args:
        sentence (str): input sentence to be augmented.
        language (str): language of the input sentence e.g. eng, deu, ind, etc. (default: eng).
        num_aug (int or float): number of augmented sentences to generate per original sentence (default: 9).
        alpha_sr (float): percent of words in each sentence to be replaced by synonyms (0 <= alpha_sr <= 1).
        alpha_ri (float): percent of words in each sentence to be inserted (0 <= alpha_sr <= 1).
        alpha_rs (float): percent of words in each sentence to be swapped (0 <= alpha_sr <= 1).
        alpha_rd (float): percent of words in each sentence to be deleted (0 <= alpha_sr <= 1).
        randomize (bool): whether to shuffle the augmented sentences (default: False).

    Returns:
        augmented_sentences (list): list of augmented sentences.
    """

    # REVIEW: Consider cleaning the sentence for sr and ri only to preserve the originality.
    clean_sentence = get_only_chars(sentence)

    words = clean_sentence.split(" ")
    words = [word for word in words if word != ""]
    num_words = len(words)

    # ? If cleaned sentence is empty, simply return the original sentence
    if num_words == 0:
        return [sentence]

    num_new_per_technique = get_num_per_technique(
        num_aug, flags=[alpha_sr, alpha_ri, alpha_rs, alpha_rd]
    )
    augmented_sentences = []

    # sr / synonym replacement
    if alpha_sr > 0:
        num_sr = max(1, int(alpha_sr * num_words))
        for _ in range(num_new_per_technique[0]):
            a_words = synonym_replacement(words, num_sr, language=language)
            augmented_sentences.append(" ".join(a_words))

    # ri / insertion
    if alpha_ri > 0:
        num_ri = max(1, int(alpha_ri * num_words))
        for _ in range(num_new_per_technique[1]):
            a_words = random_insertion(words, num_ri, language=language)
            augmented_sentences.append(" ".join(a_words))

    # rs / swap
    if alpha_rs > 0:
        num_rs = max(1, int(alpha_rs * num_words))
        for _ in range(num_new_per_technique[2]):
            a_words = random_swap(words, num_rs)
            augmented_sentences.append(" ".join(a_words))

    # rd / deletion
    if alpha_rd > 0:
        for _ in range(num_new_per_technique[3]):
            a_words = random_deletion(words, alpha_rd)
            augmented_sentences.append(" ".join(a_words))

    # Clean up augmented sentences
    augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]

    if randomize:
        random.shuffle(augmented_sentences)

    # trim so that we have the desired number of augmented sentences
    # REVIEW: Possibly fixed with get_num_per_technique(). Consider removing the if block.
    if num_aug >= 1 and len(augmented_sentences) > int(num_aug):
        print(
            f"cutting off {len(augmented_sentences) - int(num_aug)} augmented sentences "
            f"(from {len(augmented_sentences)}) // {num_new_per_technique}"
        )
        augmented_sentences = augmented_sentences[: int(num_aug)]

    elif num_aug > 0:  # 0 < num_aug < 1
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [
            aug_sentence
            for aug_sentence in augmented_sentences
            if random.uniform(0, 1) < keep_prob
        ]

    # append the original sentence
    # augmented_sentences.append(sentence)
    return [sentence] + augmented_sentences
