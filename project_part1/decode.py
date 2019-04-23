import csv
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

def decode(ciphertext, has_breakpoint):
    print("********")
    print("** ciphertext: {}".format(ciphertext))
    print("********")

    np.random.seed(seed=123)
    N = int(1e2) # num_iterations
    with open('data/alphabet.csv', 'rb') as f:
        reader = csv.reader(f)
        alphabet = list(reader)[0]
    m = len(alphabet) # alphabet size
    M = np.loadtxt(open("data/letter_transition_matrix.csv", "rb"), delimiter=",")
    P = np.loadtxt(open("data/letter_probabilities.csv", "rb"), delimiter=",")

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
        # print("Previous mapping: {}".format(f_inv[n-1]))
        # print("Proposed mapping: {}".format(f_inv[n]))

        # compute likelihoods of the ciphertext under the new and old mappings
        num[n] = compute_likelihood(P, M, f_inv[n-1], ciphertext)
        den[n] = compute_likelihood(P, M, f_inv[n], ciphertext)
        pyf[n] = num[n] / den[n]
        a[n] = min(1, pyf[n])


        # print("num: {}".format(num))
        # print("den: {}".format(den))
        # print("pyf: {}".format(pyf))
        # print("a[n]: {}".format(a[n]))
        # print("ciphertext[0]: {}".format(ciphertext[0]))
        # print("f_inv[n-1][ciphertext[0]]: {}".format(f_inv[n-1][ciphertext[0]]))
        # print("P[f_inv[n-1][ciphertext[0]]]: {}".format(P[f_inv[n-1][ciphertext[0]]]))

        # for k in range(1, ciphertext_len):
        #     print("M[f_inv[n-1][ciphertext[k]], f_inv[n-1][ciphertext[k-1]]]: {}".format(M[f_inv[n-1][ciphertext[k]], f_inv[n-1][ciphertext[k-1]]]))
        #     print("M[f_inv[n][ciphertext[k]], f_inv[n][ciphertext[k-1]]]: {}".format(M[f_inv[n][ciphertext[k]], f_inv[n][ciphertext[k-1]]]))

        # sample from bernoulli whether to accept or reject new mapping
        accept[n] = np.random.choice([True, False], p=[a[n], 1-a[n]])
        # print("accept[n]: {}".format(accept[n]))

        # if reject, copy old mapping to new mapping (reject new sample fn)
        if accept[n] == False:
            f_inv[n] = deepcopy(f_inv[n-1])
            likelihood[n] = num[n] # reject new mapping, so use old mapping to compute likelihood
        else:
            likelihood[n] = den[n] # accept the new mapping, so use it to compute likelihood

    # after MCMC has converged to proper distribution, use it to decode
    plaintext = decode_ciphertext(ciphertext, f_inv[N], alphabet)
    # f[n] = finv_to_f(f_inv[n])

    plot_likelihood(likelihood)
    plot_acceptance_rate(accept, T=10)
    with open('test_plaintext.txt', 'r') as file:
        true_plaintext = file.read()[:-1] # remove trailing \n
    plot_accuracy(true_plaintext, ciphertext, f_inv, alphabet)
    plot_log_likelihood_per_symbol(np.log2(likelihood), len(ciphertext))
    plt.show()

    return plaintext

def compute_likelihood(P, M, f_inv, ciphertext):
    l = P[f_inv[ciphertext[0]]]*np.product([M[f_inv[ciphertext[k]], f_inv[ciphertext[k-1]]] for k in range(1, len(ciphertext))])
    return l

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
    rates = np.zeros_like(ts)
    for i, t in enumerate(ts):
        rates[i] = np.sum(a[t-T:t]) / T
    print(rates)
    print(a)
    plt.figure('acceptance rate')
    plt.plot(ts, rates)
    plt.xlabel('Iteration Number')
    plt.ylabel('Acceptance Rate')

def plot_accuracy(plaintext, ciphertext, f_inv, alphabet):
    ts = range(len(f_inv))
    accs = np.zeros_like(ts)
    for t in ts:
        decoded = decode_ciphertext(ciphertext, f_inv[t], alphabet)
        accs[t] = np.sum([decoded[i] == plaintext[i] for i in range(len(ciphertext))]) / len(ciphertext)
    plt.figure('Accuracy')
    plt.plot(ts, accs)
    plt.xlabel('Iteration Number')
    plt.ylabel('Decoded Accuracy')


if __name__ == '__main__':
    with open('test_ciphertext.txt', 'r') as file:
        ciphertext = file.read()[:-1] # remove trailing \n
    print decode(ciphertext, has_breakpoint=False)

