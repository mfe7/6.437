# Usage: python encode.py input.txt output.txt has_breakpoint [seed]
# For example: python encode.py test_plaintext.txt test_ciphertext.txt False 1729
#
# Reads the first command line argument as a file, applies a random cipher to it, 
# and writes it to the file specified as the second command line argument
#
# Useful for generating ciphertexts

from __future__ import print_function
import sys
import string
from random import shuffle, randint, seed
from test import first_line
import csv


def encode(input_filename, output_filename, cipher_fn_filename=None):
    alphabet = list(string.ascii_lowercase) + [' ', '.']
    if cipher_fn_filename is None:
        letter2ix = dict(map(reversed, enumerate(alphabet)))

        cipherbet = list(alphabet) # Make a new copy of alphabet
        shuffle(cipherbet)
    else:
        letter2ix = dict(map(reversed, enumerate(alphabet)))
        with open(cipher_fn_filename, 'rb') as f:
            reader = csv.reader(f)
            cipherbet = list(reader)[0]

    plaintext = first_line(input_filename)
    ciphertext = ''.join(cipherbet[letter2ix[ltr]] for ltr in plaintext)

    with open(output_filename, 'w') as f:
        f.write(ciphertext + '\n')

    return cipherbet


def encode_with_breakpoint(input_filename, output_filename):
    plaintext = first_line(input_filename)
    alphabet = list(string.ascii_lowercase) + [' ', '.']
    letter2ix = dict(map(reversed, enumerate(alphabet)))

    breakpoint = randint(0, len(plaintext))
    print(input_filename, breakpoint)

    ciphertext = ''
    # Generate ciphertext for first section
    cipherbet = list(alphabet) 
    shuffle(cipherbet)
    ciphertext += ''.join(cipherbet[letter2ix[ltr]] for ltr in plaintext[:breakpoint])
    # Generate ciphertext for first section
    shuffle(cipherbet)
    ciphertext += ''.join(cipherbet[letter2ix[ltr]] for ltr in plaintext[breakpoint:])
    with open(output_filename, 'w') as f:
        f.write(ciphertext + '\n')


def main():
    if len(sys.argv) > 5:
        seed(sys.argv[5])

    has_breakpoint = sys.argv[3].lower() == 'true'
    if has_breakpoint:
        encode_with_breakpoint(sys.argv[1], sys.argv[2])
    else:
        if len(sys.argv) <= 4:
            cipher_fn_filename = None
        else:
            cipher_fn_filename = sys.argv[4]
        encode(sys.argv[1], sys.argv[2], cipher_fn_filename=cipher_fn_filename)

if __name__ == '__main__':
    main()


'''
python encode.py data/plaintext_short.txt data/ciphertext_short.txt false data/cipher_function.csv

'''