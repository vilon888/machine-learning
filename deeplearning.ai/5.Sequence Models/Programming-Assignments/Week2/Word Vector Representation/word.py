import numpy as np
from w2v_utils import *


def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similariy between u and v

    Arguments:
        u -- a word vector of shape (n,)
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """

    cosine_similarity = np.dot(u, v) / (np.sqrt(np.sum(np.square(u))) * np.sqrt(np.sum(np.square(v))))
    return cosine_similarity


def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    Performs the word analogy task as explained above: a is to b as c is to ____.

    Arguments:
    word_a -- a word, string
    word_b -- a word, string
    word_c -- a word, string
    word_to_vec_map -- dictionary that maps words to their corresponding vectors.

    Returns:
    best_word --  the word such that v_b - v_a is close to v_best_word - v_c, as measured by cosine similarity
    """
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]

    best_word = None
    max_cosine_sim = -900

    for w in word_to_vec_map.keys():
        if w in [word_a, word_b, word_c]:
            continue

        cosine_sim = cosine_similarity(e_b - e_a, word_to_vec_map[w] - e_c)

        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w

    return best_word


def neutralize(word, g, word_to_vec_map):
    """
    Removes the bias of "word" by projecting it on the space orthogonal to the bias axis.
    This function ensures that gender neutral words are zero in the gender subspace.

    Arguments:
        word -- string indicating the word to debias
        g -- numpy-array of shape (50,), corresponding to the bias axis (such as gender)
        word_to_vec_map -- dictionary mapping words to their corresponding vectors.

    Returns:
        e_debiased -- neutralized word vector representation of the input "word"
    """
    e_bias_component = (np.dot(word_to_vec_map[word], g) * g) / np.sum(g ** 2)
    e_debiased = word_to_vec_map[word] - e_bias_component

    return e_debiased


def equalize(pair, bias_axis, word_to_vec_map):
    """
    Debias gender specific words by following the equalize method described in the figure above.

    Arguments:
    pair -- pair of strings of gender specific words to debias, e.g. ("actress", "actor")
    bias_axis -- numpy-array of shape (50,), vector corresponding to the bias axis, e.g. gender
    word_to_vec_map -- dictionary mapping words to their corresponding vectors

    Returns
    e_1 -- word vector corresponding to the first word
    e_2 -- word vector corresponding to the second word
    """

    w1, w2 = pair[0], pair[1]
    e_w1, e_w2 = word_to_vec_map[w1], word_to_vec_map[w2]

    mu = (e_w1 + e_w2) / 2

    mu_B = np.dot(mu, bias_axis) * bias_axis / np.sum(bias_axis ** 2)
    mu_orth = mu - mu_B

    e_w1B = np.dot(e_w1, bias_axis) * bias_axis / np.sum(bias_axis ** 2)
    e_w2B = np.dot(e_w2, bias_axis) * bias_axis / np.sum(bias_axis ** 2)

    corrected_e_w1B = np.sqrt(np.abs(1 - np.sum(mu_orth ** 2))) * (e_w1B - mu_B) / np.abs(e_w1 - mu_orth - mu_B)
    corrected_e_w2B = np.sqrt(np.abs(1 - np.sum(mu_orth ** 2))) * (e_w2B - mu_B) / np.abs(e_w2 - mu_orth - mu_B)

    e1 = corrected_e_w1B + mu_orth
    e2 = corrected_e_w2B + mu_orth

    return e1, e2


words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'),
                 ('small', 'smaller', 'large')]
for triad in triads_to_try:
    print('{} -> {} :: {} -> {}'.format(*triad, complete_analogy(*triad, word_to_vec_map)))















