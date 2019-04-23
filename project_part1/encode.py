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


def encode(input_filename, output_filename):
    alphabet = list(string.ascii_lowercase) + [' ', '.']
    letter2ix = dict(map(reversed, enumerate(alphabet)))

    cipherbet = list(alphabet) # Make a new copy of alphabet
    shuffle(cipherbet)

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
    if len(sys.argv) > 4:
        seed(sys.argv[4])

    has_breakpoint = sys.argv[3].lower() == 'true'
    if has_breakpoint:
        encode_with_breakpoint(sys.argv[1], sys.argv[2])
    else:
        encode(sys.argv[1], sys.argv[2])

if __name__ == '__main__':
    main()
