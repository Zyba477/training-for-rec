#!/usr/bin/python3
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from mode import Mode
from output import Output
 

def greeting():
    print("The program consists of 3 modes.")
    print("The mode 1 is a training without augmentation.")
    print("The mode 2 is a training with augmentation.")
    print("The final mode is a training without augmentation.")


def choose_mode():
    # mode = int(input("Choose the program mode [1, 2, 3 (final)]: "))
    mode = 3
    if mode <= 0 or mode > 3:
        raise Exception("The passed mode " + str(mode) + " does not exist!")

    return mode


def main():
    greeting()
    mode = choose_mode()
    print()

    program = Mode(mode, sys.argv[-1], False)
    print()
    program.data_info()
    print()

    program.run_mode(256, 50, 0)
    NN = program.get_NN()

    NN.plot("loss")

    output = Output(NN.get_report(), program)
    output.create_report_directory()


if __name__ == '__main__':
    main()