import csv
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from encode import encode
import operator

CROP_LENGTH = 500

np.seterr(all='ignore')

def decode(full_ciphertext, has_breakpoint, true_plaintext=None, debug=False):
    n_to_stop = None
    # print("********")
    # print("** ciphertext: {}".format(ciphertext))
    # print("********")

    np.random.seed(seed=124)
    N = int(1e4) # num_iterations
    with open('data/alphabet.csv', 'rb') as f:
        reader = csv.reader(f)
        alphabet = list(reader)[0]
    m = len(alphabet) # alphabet size
    M = np.loadtxt(open("data/letter_transition_matrix.csv", "rb"), delimiter=",")
    P = np.loadtxt(open("data/letter_probabilities.csv", "rb"), delimiter=",")

    # if true_plaintext is None:
    #     # with open('test_plaintext.txt', 'r') as file:
    #     # with open('data/plaintext.txt', 'r') as file:
    #     # with open('data/plaintext_patrick.txt', 'r') as file:
    #     # with open('data/plaintext_meghan.txt', 'r') as file:
    #     # with open('data/plaintext_short.txt', 'r') as file:
    #     # with open('data/plaintext_warandpeace.txt', 'r') as file:
    #     # with open('data/plaintext_feynman.txt', 'r') as file:
    #     # with open('data/plaintext_paradiselost.txt', 'r') as file:
    #         # true_plaintext = file.read().rstrip('\n')[:CROP_LENGTH] # remove trailing \n
    #         true_plaintext = file.read().rstrip('\n') # remove trailing \n

    a = np.empty((N+1))
    pyf = np.empty((N+1))
    log_pyf = np.empty((N+1))
    accept = np.empty((N+1), dtype=bool)
    ciphertext_len = len(full_ciphertext)
    log_l_new = np.empty((N+1))
    log_l_old = np.empty((N+1))
    log_likelihood = np.empty((N+1))

    # initialize f
    # f_inv[n]:    letter y: index in x -- {'a':6, 'b':27, ...}
    f_inv = [{} for n in range(N+1)] 

    if has_breakpoint:
        breakpt = np.empty((N+1), dtype=int)
        f_inv2 = [{} for n in range(N+1)]
        early_ciphertext = full_ciphertext[:500]
        late_ciphertext = full_ciphertext[-500:]
        early_period_ind, early_space_ind, late_period_ind, late_space_ind, early_breakpt, late_breakpt = find_space_and_period_with_breakpoint(early_ciphertext, late_ciphertext, full_ciphertext, alphabet)
        early_ciphertext = full_ciphertext[:early_breakpt]
        late_ciphertext = full_ciphertext[late_breakpt:]
        f_inv[0] = initialize_f_inv(early_ciphertext, alphabet, P)
        f_inv2[0] = initialize_f_inv(late_ciphertext, alphabet, P)
        f_inv[0] = set_letters_correctly([".", " "], [early_period_ind, early_space_ind], alphabet, f_inv[0])
        f_inv2[0] = set_letters_correctly([".", " "], [late_period_ind, late_space_ind], alphabet, f_inv2[0])
        breakpt[0] = (early_breakpt + late_breakpt) / 2
        early_alphabet = deepcopy(alphabet)
        early_alphabet.remove(alphabet[early_period_ind])
        early_alphabet.remove(alphabet[early_space_ind])
        late_alphabet = deepcopy(alphabet)
        late_alphabet.remove(alphabet[late_period_ind])
        late_alphabet.remove(alphabet[late_space_ind])

    else:
        period_ind, space_ind = find_space_and_period(full_ciphertext, alphabet)
        f_inv[0] = initialize_f_inv(full_ciphertext, alphabet, P)
        f_inv[0] = set_letters_correctly([".", " "], [period_ind, space_ind], alphabet, f_inv[0])

    vowel_inds = [alphabet.index(vowel) for vowel in ['a','e','i','o','u']]

    # with open('data/cipher_function.csv', 'rb') as f:
    #     reader = csv.reader(f)
    #     cipher_function = list(reader)[0]
    # print(cipher_function)
    # for i, letter in enumerate(cipher_function):
    #     # REMOVE LATER -- JUST TO TEST WITH THE RIGHT F
    #     f_inv[0][letter] = i
    # # print("correct fn:", f_inv[0])
    # f_inv_orig = deepcopy(f_inv[0])
    # # remappings = [["c","d"], ["e","f"], ["a","b"], ["g","h"]]
    # for kk in range(4):
    #     i, j = np.random.choice(alphabet, 2, replace=False)
    #     # i,j = remappings[kk]
    #     f_inv[0][i] = f_inv_orig[j]
    #     f_inv[0][j] = f_inv_orig[i]

    # print("Initial state:")
    # print("f_inv[0]: {}".format(f_inv[0]))
    # print("f_inv2[0]: {}".format(f_inv2[0]))

    # print(decode_ciphertext(full_ciphertext[:200], f_inv[0], alphabet))
    # if has_breakpoint:
    #     print(decode_ciphertext(full_ciphertext[-200:], f_inv2[0], alphabet))

    # print('-----')
    # print('-----')

    # Run the Markov Chain N steps so it converges to desired distribution
    for n in range(1,N+1):
        # print("-------")
        # print("Step: {}".format(n))

        f_inv[n] = deepcopy(f_inv[n-1])
        if has_breakpoint:
            f_inv2[n] = deepcopy(f_inv2[n-1])
            breakpt[n] = deepcopy(breakpt[n-1])

        # if n % 100 == 0:
        #     # try swapping vowels
        #     vowel_keys = []
        #     for key in f_inv[n-1].keys():
        #         if f_inv[n-1][key] in vowel_inds:
        #             vowel_keys.append(key)
        #     # i, j = np.random.choice(vowel_keys, 2, replace=False)
        #     num_swaps = 1
        #     swaps = np.random.choice(vowel_keys, 2*num_swaps, replace=False)
        #     i, j = swaps
        #     print("Random swap of: ({},{})".format(i,j))
        #     print("we think those are: ({},{})".format(alphabet[f_inv[n-1][i]], alphabet[f_inv[n-1][j]]))
        #     for swap in range(num_swaps):
        #         i, j = swaps[2*swap:2+2*swap]
        #         f_inv[n][i] = f_inv[n-1][j]
        #         f_inv[n][j] = f_inv[n-1][i]
        

        if has_breakpoint:
            if n % 3 == 0:
                # print("proposed new breakpt")
                breakpt[n] = (2*np.random.binomial(n=1,p=0.5)-1) + breakpt[n-1]
            else:
                if n % 3 == 1:
                    # print("proposed new 1st half")
                    mapping = f_inv
                    ciphertext = early_ciphertext
                    this_alphabet = early_alphabet
                else:
                    # print("proposed new 2nd half")
                    mapping = f_inv2
                    ciphertext = late_ciphertext
                    this_alphabet = late_alphabet
                num_swaps = 1
                swaps = np.random.choice(this_alphabet, 2*num_swaps, replace=False)
                for swap in range(num_swaps):
                    i, j = swaps[2*swap:2+2*swap]
                    mapping[n][i] = mapping[n-1][j]
                    mapping[n][j] = mapping[n-1][i]
        else:
            # Sample two indices from the previous cipher mapping
            # i, j = np.random.choice(alphabet, 2, replace=False)
            num_swaps = 1
            swaps = np.random.choice(alphabet, 2*num_swaps, replace=False)
            for swap in range(num_swaps):
                i, j = swaps[2*swap:2+2*swap]
                f_inv[n][i] = f_inv[n-1][j]
                f_inv[n][j] = f_inv[n-1][i]
            # print("Random swap of: ({},{})".format(i,j))


        # propose new cipher mapping by swapping at those 2 sampled indices
        # f_inv[n][i] = f_inv[n-1][j]
        # f_inv[n][j] = f_inv[n-1][i]
        


        # if n % 100 == 0:
        #     print(decode_ciphertext(ciphertext, f_inv[n-1], alphabet)[:500])
        #     print('--')
        #     print(decode_ciphertext(ciphertext, f_inv[n], alphabet)[:500])

        # compute log_likelihoods of the ciphertext under the new and old mappings

        if has_breakpoint:
            log_pyf[n], log_l_new[n], log_l_old[n] = compute_likelihood_breakpt(P, M, mapping[n], mapping[n-1], ciphertext, this_alphabet)
        else:
            log_pyf[n], log_l_new[n], log_l_old[n] = compute_likelihood(P, M, f_inv[n], f_inv[n-1], full_ciphertext, alphabet)

        if np.isinf(log_l_old[n]):
            pyf[n] = 0.
        else:
            pyf[n] = np.power(2, log_pyf[n])

        a[n] = np.min([1., pyf[n]])

        # print("log_l_new[n]: {}".format(log_l_new[n]))
        # print("log_l_old[n]: {}".format(log_l_old[n]))
        # print("log_pyf[n]: {}".format(log_pyf[n]))
        # print("pyf[n]: {}".format(pyf[n]))
        # print("a[n]: {}".format(a[n]))

        # sample from bernoulli whether to accept or reject new mapping
        accept[n] = np.random.binomial(n=1, p=a[n])
        # print("accept[n]: {}".format(accept[n]))

        # if reject, copy old mapping to new mapping (reject new sample fn)
        if accept[n] == False:
            # print("reject")
            if has_breakpoint:
                mapping[n] = deepcopy(mapping[n-1])
            else:
                f_inv[n] = deepcopy(f_inv[n-1])
            log_likelihood[n] = log_l_old[n] # reject new mapping, so use old mapping to compute log_likelihood
        else:
            # print("Accept!")
            log_likelihood[n] = log_l_new[n] # accept the new mapping, so use it to compute log_likelihood
            # if n_to_stop is None:
            #     n_to_stop = n + 20
        
        if n == n_to_stop:
            break
        
        ### Debug Msgs
        if n % 10 == 0 and debug:
            print("Step: {}".format(n))
            # print("Acc: {}".format(compute_accuracy(true_plaintext, ciphertext, f_inv[n], alphabet)))
            print("log_l_new[n]: {}".format(log_l_new[n]))
            print("log_l_old[n]: {}".format(log_l_old[n]))
            print("log_pyf[n]: {}".format(log_pyf[n]))
            # print('--')
            # print("previous: {}".format(decode_ciphertext(full_ciphertext[:200], f_inv[n-1], alphabet)))
            # print("proposed: {}".format(decode_ciphertext(full_ciphertext[:200], f_inv[n], alphabet)))
            print("{}".format(decode_ciphertext(full_ciphertext[:200], f_inv[n], alphabet)))
            # print('--')
            if has_breakpoint:
                # print("breakpoint: {}".format(breakpt[n]))
                # print('--')
                # print("previous: {}".format(decode_ciphertext(full_ciphertext[-200:], f_inv2[n-1], alphabet)))
                # print("proposed: {}".format(decode_ciphertext(full_ciphertext[-200:], f_inv2[n], alphabet)))
                print("{}".format(decode_ciphertext(full_ciphertext[-200:], f_inv2[n], alphabet)))

            # print("accept[n]: {}".format(accept[n]))
            # if accept[n]:
            #     assert(0)
            print("-----")

    # after MCMC has converged to proper distribution, use it to decode
    if has_breakpoint:
        plaintext = decode_ciphertext(full_ciphertext[:breakpt[n]], f_inv[n], alphabet) + decode_ciphertext(full_ciphertext[breakpt[n]:], f_inv2[n], alphabet)
    else:
        plaintext = decode_ciphertext(full_ciphertext, f_inv[n], alphabet)
    # f[n] = finv_to_f(f_inv[n])

    # plot_log_likelihood(log_likelihood[:n+1])
    # plot_acceptance_rate(accept[:n+1], T=20)
    # # accuracy = plot_accuracy(true_plaintext, ciphertext, f_inv[:n+1], alphabet)
    # plot_log_likelihood_per_symbol(log_likelihood[:n+1], len(ciphertext))
    # plt.show()

    if debug:
        print("final answer....")
        print(plaintext[:100] + " ... " + plaintext[-100:])

    # print(plaintext)
    accuracy = 0
    # return plaintext, accuracy
    return plaintext

def set_letters_correctly(letters_to_get_correct, letter_inds, alphabet, f_inv):
    # only set space and . to be the correct mapping
    for k in range(len(letters_to_get_correct)):
        letter_to_get_correct = letters_to_get_correct[k]
        letter_alphabet_ind = alphabet.index(letter_to_get_correct) # where is the letter in the normal alphabet
        letter_inv = alphabet[letter_inds[k]] # letter the letter should be encoded as
        # print("the letter: '{}' should be encoded as '{}'".format(letter_to_get_correct, letter_inv))
        # print("but currently we decode '{}' as '{}'.".format(letter_inv, alphabet[f_inv[letter_inv]]))
        for i, letter in enumerate(f_inv.keys()):
            if f_inv[letter] == letter_alphabet_ind:
                period_letter = letter
        f_inv[period_letter] = f_inv[letter_inv]
        f_inv[letter_inv] = letter_alphabet_ind
        # print("After fixing.....")
        # print("and we now decode '{}' as '{}'.".format(letter_inv, alphabet[f_inv[letter_inv]]))
    return f_inv

def get_ciphertext_pairs(ciphertext):
    pairs = {}
    for k in range(1, len(ciphertext)):
        pair = ciphertext[k:k+2]
        if pair in pairs:
            pairs[pair] += 1
        else:
            pairs[pair] = 1

def compute_likelihood_breakpt(P, M, mapping_new, mapping_old, ciphertext, alphabet):
    log_pyf = 0
    # log_pyf = np.log2(P[mapping_new[ciphertext[0]]] / P[mapping_old[ciphertext[0]]])
    log_new = 0
    log_old = 0
    for k in range(1, len(ciphertext)):
        num = np.log2(M[mapping_new[ciphertext[k]], mapping_new[ciphertext[k-1]]])
        den = np.log2(M[mapping_old[ciphertext[k]], mapping_old[ciphertext[k-1]]])
        if np.isinf(num) and np.isinf(den):
            num = -100
        # if np.isinf(den):
        #     den = -1e5
        log_new += num
        log_old += den
        # if num == 0 or den == 0:
        # print("new mapping: {},{}".format(alphabet[map_new[ciphertext[k]]], alphabet[map_new[ciphertext[k-1]]]))
        #     print("prev mapping: {},{}".format(alphabet[f_inv[ciphertext[k]]], alphabet[f_inv[ciphertext[k-1]]]))
        # print("num: {}".format(num))
        #     print("den: {}".format(den))
        ratio = num - den
        # ratio = den - num
        log_pyf += ratio
        # print(true_plaintext[k], ciphertext[k], alphabet[f_inv[ciphertext[k]]], alphabet[f_inv[ciphertext[k-1]]], num, den, ratio, log_pyf)
        # print(ciphertext[k], num, den, ratio, pyf)
    return log_pyf, log_new, log_old

def compute_likelihood(P, M, f_inv_new, f_inv, ciphertext, alphabet):
    # with open('data/plaintext_short.txt', 'r') as file:
    #     true_plaintext = file.read().rstrip('\n') # remove trailing \n
    log_pyf = np.log2(P[f_inv_new[ciphertext[0]]] / P[f_inv[ciphertext[0]]])
    log_new = 0
    log_old = 0
    # print("first letter prob: {}".format(pyf))
    for k in range(1, len(ciphertext)):
        num = np.log2(M[f_inv_new[ciphertext[k]], f_inv_new[ciphertext[k-1]]])
        den = np.log2(M[f_inv[ciphertext[k]], f_inv[ciphertext[k-1]]])
        if np.isinf(num) and np.isinf(den):
            num = den = -50
        log_new += num
        log_old += den
        # if num == 0 or den == 0:
        #     print("new mapping: {},{}".format(alphabet[f_inv_new[ciphertext[k]]], alphabet[f_inv_new[ciphertext[k-1]]]))
        #     print("prev mapping: {},{}".format(alphabet[f_inv[ciphertext[k]]], alphabet[f_inv[ciphertext[k-1]]]))
        #     print("num: {}".format(num))
        #     print("den: {}".format(den))
        ratio = num - den
        # ratio = den - num
        log_pyf += ratio
        # print(true_plaintext[k], ciphertext[k], alphabet[f_inv[ciphertext[k]]], alphabet[f_inv[ciphertext[k-1]]], num, den, ratio, log_pyf)
        # print(ciphertext[k], num, den, ratio, pyf)
    return log_pyf, log_new, log_old

def initialize_f_inv(ciphertext, alphabet, P):
    p = {}
    letter_counts = {}
    f_inv = {}
    for i, letter in enumerate(alphabet):
        letter_counts[letter] = 0
        p[letter] = P[i]
    for letter in ciphertext:
        letter_counts[letter] += 1
    sorted_letters = sorted(letter_counts.items(), key=operator.itemgetter(1))
    p_sorted_letters = sorted(p.items(), key=operator.itemgetter(1))
    letters = [l[0] for l in sorted_letters]
    p_letters = [l[0] for l in p_sorted_letters]
    for i in range(len(p_letters)):
        f_inv[letters[i]] = alphabet.index(p_letters[i])
    return f_inv

def find_space_and_period_with_breakpoint(early_ciphertext, late_ciphertext, full_ciphertext, alphabet):
    for ciphertext in [early_ciphertext, late_ciphertext]:
        mat = np.zeros((len(alphabet), len(alphabet)))
        for k in range(1, len(ciphertext)):
            mat[alphabet.index(ciphertext[k-1]), alphabet.index(ciphertext[k])] += 1
        top_n_pair_counts = np.sort(np.partition(mat, -2)[:,-2:])
        pos_and_zero_inds = np.where(np.logical_and(top_n_pair_counts[:,0] == 0, top_n_pair_counts[:,1] > 0))
        pos_and_zero_mat = top_n_pair_counts[pos_and_zero_inds,:]
        # print("mat: {}".format(mat))
        # print("top_n_pair_counts: {}".format(top_n_pair_counts))
        # print("pos_and_zero_inds: {}".format(pos_and_zero_inds))
        # print("pos_and_zero_mat: {}".format(pos_and_zero_mat))
        # try:

        # _, period_ind_, _ = np.unravel_index(pos_and_zero_mat.argmax(), pos_and_zero_mat.shape)
        # period_ind = pos_and_zero_inds[0][period_ind_]
        # space_ind = mat[period_ind].argmax()
        # print("old method...")
        # print("period_ind_: {}".format(period_ind_))
        # print("period_ind: {}".format(period_ind))
        # print("space_ind: {}".format(space_ind))
        
        space_inds = np.argmax(mat[pos_and_zero_inds], axis=1)
        num_spaces = np.sum(mat[:,space_inds], axis=0)
        space_ind_ = num_spaces.argmax()
        space_ind = space_inds[space_ind_]
        period_ind = pos_and_zero_inds[0][space_ind_]
        # print("new method...")
        # print("num_spaces: {}".format(num_spaces))
        # print("period_ind: {}".format(period_ind))
        # print("space_ind: {}".format(space_ind))

        period_inds = find(full_ciphertext, alphabet[period_ind])
        # print("period_inds: {}".format(period_inds))
        for bdry_ind in [0, len(full_ciphertext)-1]:
            if bdry_ind in period_inds:
                period_inds.remove(bdry_ind)
        breakpt_guess = None
        if ciphertext == late_ciphertext:
            period_inds.reverse()
        for i in range(len(period_inds)):
            k = period_inds[i]
            if full_ciphertext[k-1] not in [alphabet[period_ind], alphabet[space_ind]] and full_ciphertext[k+1] == alphabet[space_ind]:
                continue
            else:
                if i == 0:
                    breakpt_guess = period_inds[0]
                else:
                    breakpt_guess = period_inds[i-1]
                break
        # print("breakpt_guess: {}".format(breakpt_guess))

        if ciphertext == late_ciphertext:
            late_period_ind = period_ind; late_space_ind = space_ind
            late_breakpt_guess = breakpt_guess
        else:
            early_period_ind = period_ind; early_space_ind = space_ind
            early_breakpt_guess = breakpt_guess
        # except:
        #     period_ind = 0; space_ind = 1
    # print(mat[period_ind])
    # print("period_ind: {}, space_ind: {}".format(period_ind, space_ind))
    # print("period_letter: {}, space_letter: {}".format(alphabet[period_ind], alphabet[space_ind]))
    # assert(0)
    return early_period_ind, early_space_ind, late_period_ind, late_space_ind, early_breakpt_guess, late_breakpt_guess

def find_space_and_period(ciphertext, alphabet):
    mat = np.zeros((len(alphabet), len(alphabet)))
    for k in range(1, len(ciphertext)):
        mat[alphabet.index(ciphertext[k-1]), alphabet.index(ciphertext[k])] += 1
    # print("mat: {}".format(mat))
    top_n_pair_counts = np.sort(np.partition(mat, -2)[:,-2:])
    # print("top_n_pair_counts: {}".format(top_n_pair_counts))
    pos_and_zero_inds = np.where(np.logical_and(top_n_pair_counts[:,0] == 0, top_n_pair_counts[:,1] > 0))
    pos_and_zero_mat = top_n_pair_counts[pos_and_zero_inds,:]
    # print("pos_and_zero_inds: {}".format(pos_and_zero_inds))
    # print("pos_and_zero_mat: {}".format(pos_and_zero_mat))
    try:

        # _, period_ind_, _ = np.unravel_index(pos_and_zero_mat.argmax(), pos_and_zero_mat.shape)
        # period_ind = pos_and_zero_inds[0][period_ind_]
        # space_ind = mat[period_ind].argmax()
        # print("old method...")
        # print("period_ind_: {}".format(period_ind_))
        # print("period_ind: {}".format(period_ind))
        # print("space_ind: {}".format(space_ind))
        
        space_inds = np.argmax(mat[pos_and_zero_inds], axis=1)
        space_ind_ = np.sum(mat[:,space_inds], axis=0).argmax()
        space_ind = space_inds[space_ind_]
        period_ind = pos_and_zero_inds[0][space_ind_]
        # print("new method...")
        # print("period_ind: {}".format(period_ind))
        # print("space_ind: {}".format(space_ind))

        
    except:
        period_ind = 0; space_ind = 1
    # print(mat[period_ind])
    # print("period_ind: {}, space_ind: {}".format(period_ind, space_ind))
    # print("period_letter: {}, space_letter: {}".format(alphabet[period_ind], alphabet[space_ind]))
    # assert(0)
    return period_ind, space_ind

def finv_to_f(f_inv):
    return {v: k for k, v in f_inv.iteritems()}

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def decode_ciphertext(ciphertext, f_inv, alphabet):
    plaintext = ''
    for letter in ciphertext:
        plaintext += alphabet[f_inv[letter]]
    return plaintext

def plot_log_likelihood(log_l):
    ts = range(1, len(log_l))
    plt.figure('log likelihood')
    plt.plot(ts, log_l[1:])
    plt.xlabel('Iteration Number')
    plt.ylabel('Log-Likelihood (Of Accepted Mapping)')
    plt.savefig('plots/log_likelihood.png')

def plot_log_likelihood_per_symbol(log_l, num_symbols):
    ts = range(1, len(log_l))
    plt.figure('likelihood')
    print(log_l)
    logl_per_symb = log_l[1:]/float(num_symbols)
    plt.plot(ts, logl_per_symb)
    plt.xlabel('Iteration Number')
    plt.ylabel('Log-Likelihood Per Symbol (Of Accepted Mapping) (bits)')
    plt.savefig('plots/log_likelihood_per_symbol.png')

def plot_acceptance_rate(a, T):
    ts = range(T+1, len(a))
    rates = np.zeros_like(ts, dtype=float)
    for i, t in enumerate(ts):
        rates[i] = np.sum(a[t-T:t]) / float(T)
    print("rates", rates)
    print("a",a)
    plt.figure('acceptance rate')
    plt.plot(ts, rates)
    plt.xlabel('Iteration Number')
    plt.ylabel('Acceptance Rate')
    plt.savefig('plots/acceptance_rate.png')
    return rates

def compute_accuracy(true_plaintext, ciphertext, f_inv, alphabet):
    decoded = decode_ciphertext(ciphertext, f_inv, alphabet)
    correct_letters = np.sum([decoded[i] == true_plaintext[i] for i in range(len(ciphertext))])
    acc = correct_letters / float(len(ciphertext))
    return acc

def plot_accuracy(true_plaintext, ciphertext, f_inv, alphabet):
    ts = range(len(f_inv))
    accs = np.zeros_like(ts, dtype=float)
    for t in ts:
        accs[t] = compute_accuracy(true_plaintext, ciphertext, f_inv[t], alphabet)
    plt.figure('Accuracy')
    plt.plot(ts, accs)
    plt.xlabel('Iteration Number')
    plt.ylabel('Decoded Accuracy')
    plt.savefig('plots/accuracy.png')
    return accs

def test_likelihood():
    M = np.loadtxt(open("data/letter_transition_matrix.csv", "rb"), delimiter=",")
    P = np.loadtxt(open("data/letter_probabilities.csv", "rb"), delimiter=",")
    compute_likelihood3(P,M,"test.ametter", "test. letter")

if __name__ == '__main__':
    # test_likelihood()
    # with open('data/ciphertext_paradiselost.txt', 'r') as file:
    # with open('data/ciphertext_warandpeace.txt', 'r') as file:
    # with open('data/ciphertext_short.txt', 'r') as file:
    # with open('data/ciphertext_meghan.txt', 'r') as file:
    # with open('data/ciphertext_patrick.txt', 'r') as file:
    with open('test_ciphertext_breakpoint.txt', 'r') as file:
    # with open('data/ciphertext.txt', 'r') as file:
    # with open('data/ciphertext_feynman.txt', 'r') as file:
    # with open('data/ciphertext_feynman_breakpoint.txt', 'r') as file:
        ciphertext = file.read().rstrip('\n') # remove trailing \n
    decoded = decode(ciphertext, has_breakpoint=True, debug=True)
    # decoded = decode(ciphertext, has_breakpoint=False)

    # crop_lengths = [100, 200, 300, 500, 1000, 2000, 5000]
    # # crop_lengths = [100, 200, 500]
    # ts = None
    # for length in crop_lengths:
    #     CROP_LENGTH = length
    #     with open('data/ciphertext.txt', 'r') as file:
    #         ciphertext = file.read().rstrip('\n')[:CROP_LENGTH] # remove trailing \n
    #     decoded, acc_rate = decode(ciphertext, has_breakpoint=False)
    #     if ts is None:
    #         ts = range(len(acc_rate))
    #     plt.figure('seqlen')
    #     plt.plot(ts, acc_rate, label=str(length))
    #     plt.xlabel('Iteration Number')
    #     plt.ylabel('Decoded Accuracy')
    # plt.legend(title="Text Length", loc=4)
    # plt.savefig('plots/seqlen.png')
    # plt.show()

