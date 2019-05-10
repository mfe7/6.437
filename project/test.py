##############################################################
# Note to students:                                          #
# Do NOT modify this file!                                   #
# If you get errors, you need to fix those in *your* code.   #
# Modifying this code may lead you to generate an upload.zip #
# that will not work in our grader.                          #
##############################################################

import subprocess
import datetime
from time import time
import os
import traceback

log_filename = 'log'

def log(message, logfile=log_filename):
    timestamp = '[' + str(datetime.datetime.now()) + '] ' 
    text = timestamp + str(message)
    print text
    with open(logfile, 'a') as f: # Open log file in append mode
        f.write(text + '\n')

class ComparisonError(ValueError):
    pass

def compare(a, b):
    if len(a) != len(b):
        raise ComparisonError("strings to compare must have the same length")
    # Return the number of locations where the two strings are equal
    return sum([int(i==j) for i,j in zip(a,b)])

def test(executable_path, plaintext, ciphertext, breakpoint):
    subprocess.call(['chmod', '+x', executable_path]) # Ensure executable can be executed
    executable_file = os.path.basename(executable_path) # foo/bar/decode -> decode
    executable_dir = os.path.dirname(executable_path) # foo/bar/decode -> foo/bar

    start_dir = os.getcwd()
    os.chdir(executable_dir) ###### CHANGE TO CODE DIRECTORY (the student's code may use relative paths)
    
    try:
        start_time = time()
        output = subprocess.check_output(['./'+executable_file, ciphertext, str(breakpoint)], stderr=subprocess.STDOUT).rstrip('\r\n')
        end_time = time()
    except subprocess.CalledProcessError as e:
        os.chdir(start_dir) ###### CHANGE BACK TO ORIGINAL DIRECTORY
        output = "Error! See log for details."
        elapsed_time = 0
        score = 0 
        max_score = 1
        log("!!! ERROR !!!")
        log(e.output, logfile = executable_dir + '/' + log_filename)
    else:
        os.chdir(start_dir) ###### CHANGE BACK TO ORIGINAL DIRECTORY
        elapsed_time = end_time - start_time
        score = compare(plaintext, output)
        max_score = len(plaintext)

    return (elapsed_time, score, max_score, output)

def first_line(filename):
    # Return first line of file as string, without trailing newline
    with open(filename) as f:
        string = f.readline().rstrip('\r\n')

    return string

def main():
    executable = './decode'
    plaintext = first_line('./test_plaintext.txt')
    ciphertext = first_line('./test_ciphertext.txt')
    ciphertext_with_breakpoint = first_line('./test_ciphertext_breakpoint.txt')
    print "Running your code..."
    try:
        (elapsed_time1, score1, max_score1, output) = test(executable, plaintext, ciphertext, False)
        (elapsed_time2, score2, max_score2, output) = test(executable, plaintext, ciphertext_with_breakpoint, True)
    except ComparisonError as e:
        log("!!! ERROR !!!")
        log(traceback.format_exc())
        print "Please verify that your plaintext output is the same length as the ciphertext."
    else:
        print "Elapsed time: " + str(elapsed_time1 + elapsed_time2) + " s"
        print "Score (no breakpoint): " + str(score1) + " out of " + str(max_score1)
        print "Score (with breakpoint): " + str(score2) + " out of " + str(max_score2)
        if max_score1 != 1 and max_score2 != 1: # Don't make upload.zip if there were errors
            print "Creating an upload.zip that you can submit..."
            subprocess.call(['zip','-r','upload.zip'] + sorted(os.listdir('.')))
        else:
            print "Your code seems to have errors. Please fix them and then rerun this test."

if __name__ == '__main__':
    main()
