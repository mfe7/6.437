import csv
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from encode import encode

def decode(ciphertext, has_breakpoint, true_plaintext=None):
    print("********")
    print("** ciphertext: {}".format(ciphertext))
    print("********")

    np.random.seed(seed=123)
    N = int(1e3) # num_iterations
    with open('data/alphabet.csv', 'rb') as f:
        reader = csv.reader(f)
        alphabet = list(reader)[0]
    m = len(alphabet) # alphabet size
    M = np.loadtxt(open("data/letter_transition_matrix.csv", "rb"), delimiter=",")
    P = np.loadtxt(open("data/letter_probabilities.csv", "rb"), delimiter=",")

    with open('data/cipher_function.csv', 'rb') as f:
        # REMOVE LATER
        reader = csv.reader(f)
        cipher_function = list(reader)[0]

    a = np.empty((N+1))
    pyf = np.empty((N+1))
    accept = np.empty((N+1), dtype=bool)
    ciphertext_len = len(ciphertext)
    num = np.empty((N+1))
    den = np.empty((N+1))
    likelihood = np.empty((N+1))

    # initialize f
    # f_inv[n]:    letter y: index in x -- {'a':6, 'b':27, ...}
    f_inv = [{} for n in range(N+1)] 
    f = [{} for n in range(N+1)]
    for i, letter in enumerate(alphabet):
        f_inv[0][letter] = i
    for i, letter in enumerate(cipher_function):
        # REMOVE LATER -- JUST TO TEST WITH THE RIGHT F
        f_inv[0][letter] = i
    f_inv_orig = deepcopy(f_inv[0])
    for kk in range(1):
        i, j = np.random.choice(alphabet, 2, replace=False)
        i,j = "a","b"
        f_inv[0][i] = f_inv_orig[j]
        f_inv[0][j] = f_inv_orig[i]
    # print("true mapping: {}".format(f_inv_orig))
    # print("init mapping: {}".format(f_inv[0]))

    # Run the Markov Chain N steps so it converges to desired distribution
    for n in range(1,N+1):
        print("-------")
        print("Step: {}".format(n))

        # Sample two indices from the previous cipher mapping
        i, j = np.random.choice(alphabet, 2, replace=False)
        # print("Random swap of: ({},{})".format(i,j))

        # propose new cipher mapping by swapping at those 2 sampled indices
        f_inv[n] = deepcopy(f_inv[n-1])
        f_inv[n][i] = f_inv[n-1][j]
        f_inv[n][j] = f_inv[n-1][i]

        # print("previous decoding: {}".format(decode_ciphertext(ciphertext, f_inv[n-1], alphabet)))
        # print('-----')
        # print("proposed decoding: {}".format(decode_ciphertext(ciphertext, f_inv[n], alphabet)))
        # assert(0)

        # # propose new cipher mapping by shuffling all indices
        # keys = f_inv[n-1].keys()
        # values = f_inv[n-1].values()
        # np.random.shuffle(values)
        # f_inv[n] = dict(zip(keys, values))
        # print("Previous mapping: {}".format(f_inv[n-1]))
        # print("Proposed mapping: {}".format(f_inv[n]))
        # print("prev mapping plaintext: {}".format(decode_ciphertext(ciphertext, f_inv[n-1],alphabet)))

        # compute likelihoods of the ciphertext under the new and old mappings
        pyf[n], num[n], den[n] = compute_likelihood2(P, M, f_inv[n], f_inv[n-1], ciphertext, alphabet)
        # num[n] = compute_likelihood(P, M, f_inv[n], ciphertext)
        # den[n] = compute_likelihood(P, M, f_inv[n-1], ciphertext)
        # pyf[n] = num[n] / den[n]
        a[n] = min(1, pyf[n])

        # print("num[n]: {}".format(num[n]))
        # print("den[n]: {}".format(den[n]))
        # print("pyf[n]: {}".format(pyf[n]))
        # print("a[n]: {}".format(a[n]))

        # print("ciphertext[0]: {}".format(ciphertext[0]))
        # print("f_inv[n-1][ciphertext[0]]: {}".format(f_inv[n-1][ciphertext[0]]))
        # print("P[f_inv[n-1][ciphertext[0]]]: {}".format(P[f_inv[n-1][ciphertext[0]]]))

        # sample from bernoulli whether to accept or reject new mapping
        accept[n] = np.random.choice([True, False], p=[a[n], 1-a[n]])
        # print("accept[n]: {}".format(accept[n]))

        # if reject, copy old mapping to new mapping (reject new sample fn)
        if accept[n] == False:
            print("reject")
            f_inv[n] = deepcopy(f_inv[n-1])
            likelihood[n] = den[n] # reject new mapping, so use old mapping to compute likelihood
        else:
            print("Accept!")
            break
            likelihood[n] = num[n] # accept the new mapping, so use it to compute likelihood

    # after MCMC has converged to proper distribution, use it to decode
    plaintext = decode_ciphertext(ciphertext, f_inv[n], alphabet)
    # f[n] = finv_to_f(f_inv[n])

    plot_likelihood(likelihood)
    plot_acceptance_rate(accept, T=10)
    if true_plaintext is None:
        with open('data/plaintext.txt', 'r') as file:
            true_plaintext = file.read().rstrip('\n') # remove trailing \n
    plot_accuracy(true_plaintext, ciphertext, f_inv, alphabet)
    plot_log_likelihood_per_symbol(np.log2(likelihood), len(ciphertext))
    plt.show()

    print(plaintext)
    return plaintext

def compute_likelihood(P, M, f_inv, ciphertext):
    # print("P[f_inv[ciphertext[0]]]: {}".format(P[f_inv[ciphertext[0]]]))
    # for k in range(1, len(ciphertext)):
    #     print("M[f_inv[ciphertext[k]], f_inv[ciphertext[k-1]]]: {}".format(M[f_inv[ciphertext[k]], f_inv[ciphertext[k-1]]]))
    l = P[f_inv[ciphertext[0]]]*np.product([np.log(M[f_inv[ciphertext[k]], f_inv[ciphertext[k-1]]]) for k in range(1, len(ciphertext))])
    return l

def compute_likelihood2(P, M, f_inv_new, f_inv, ciphertext, alphabet):
    pyf = np.log(P[f_inv_new[ciphertext[0]]] / P[f_inv[ciphertext[0]]])
    # print("first letter prob: {}".format(pyf))
    for k in range(1, len(ciphertext)):
        num = np.log(M[f_inv_new[ciphertext[k]], f_inv_new[ciphertext[k-1]]])
        den = np.log(M[f_inv[ciphertext[k]], f_inv[ciphertext[k-1]]])
        # if num == 0 or den == 0:
        #     print("new mapping: {},{}".format(alphabet[f_inv_new[ciphertext[k]]], alphabet[f_inv_new[ciphertext[k-1]]]))
        #     print("prev mapping: {},{}".format(alphabet[f_inv[ciphertext[k]]], alphabet[f_inv[ciphertext[k-1]]]))
        #     print("num: {}".format(num))
        #     print("den: {}".format(den))
        ratio = num - den
        # ratio = den - num
        pyf += ratio
        # print(ciphertext[k], num, den, ratio, pyf)
    num = den = 0
    pyf = np.exp(pyf)
    return pyf, num, den

def compute_likelihood3(P, M, previous, proposed):
    with open('data/alphabet.csv', 'rb') as f:
        reader = csv.reader(f)
        alphabet = list(reader)[0]

    pyf = np.log(P[alphabet.index(previous[0])] / P[alphabet.index(proposed[0])])
    for k in range(1, len(previous)):
        if previous[k] == proposed[k]:
            continue
        num = np.log(M[alphabet.index(previous[k]), alphabet.index(previous[k-1])])
        den = np.log(M[alphabet.index(proposed[k]), alphabet.index(proposed[k-1])])
        ratio = num - den
        pyf += ratio
        print(num, den, ratio, pyf)
    print(pyf)
    pyf = np.exp(pyf)


def finv_to_f(f_inv):
    return {v: k for k, v in f_inv.iteritems()}

def decode_ciphertext(ciphertext, f_inv, alphabet):
    plaintext = ''
    for letter in ciphertext:
        plaintext += alphabet[f_inv[letter]]
    return plaintext

def plot_likelihood(l):
    ts = range(1, len(l))
    plt.figure('likelihood')
    plt.plot(ts, l[1:])
    plt.xlabel('Iteration Number')
    plt.ylabel('Likelihood Under Accepted Mapping')

def plot_log_likelihood_per_symbol(log_l, num_symbols):
    ts = range(1, len(log_l))
    plt.figure('likelihood')
    plt.plot(ts, log_l[1:]/num_symbols)
    plt.xlabel('Iteration Number')
    plt.ylabel('Log-Likelihood Per Symbol (Under Accepted Mapping) (bits)')

def plot_acceptance_rate(a, T):
    ts = range(T+1, len(a))
    rates = np.zeros_like(ts, dtype=float)
    for i, t in enumerate(ts):
        rates[i] = np.sum(a[t-T:t]) / float(T)
    print(rates)
    print(a)
    plt.figure('acceptance rate')
    plt.plot(ts, rates)
    plt.xlabel('Iteration Number')
    plt.ylabel('Acceptance Rate')

def plot_accuracy(true_plaintext, ciphertext, f_inv, alphabet):
    ts = range(len(f_inv))
    accs = np.zeros_like(ts, dtype=float)
    for t in ts:
        decoded = decode_ciphertext(ciphertext, f_inv[t], alphabet)
        correct_letters = np.sum([decoded[i] == true_plaintext[i] for i in range(len(ciphertext))])
        accs[t] = correct_letters / float(len(ciphertext))
    plt.figure('Accuracy')
    plt.plot(ts, accs)
    plt.xlabel('Iteration Number')
    plt.ylabel('Decoded Accuracy')

def test_likelihood():
    M = np.loadtxt(open("data/letter_transition_matrix.csv", "rb"), delimiter=",")
    P = np.loadtxt(open("data/letter_probabilities.csv", "rb"), delimiter=",")
    compute_likelihood3(P,M,"test.ametter", "test. letter")

if __name__ == '__main__':
    # test_likelihood()
    with open('data/ciphertext.txt', 'r') as file:
    # with open('test_ciphertext.txt', 'r') as file:
        ciphertext = file.read().rstrip('\n') # remove trailing \n
    # true_plaintext = "the dog runs. and the cat swims."
    # ciphertext = "the dog runs. and the cat swims."
    # ciphertext = "the qog runs. and the cat swims."
    # ciphertext = "thelqoglruns.landlthelcatlswims."
    true_plaintext = None
    decode(ciphertext, has_breakpoint=False, true_plaintext=true_plaintext)

