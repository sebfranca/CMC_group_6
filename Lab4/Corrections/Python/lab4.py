""" Lab 4 """

import farms_pylog as pylog
from exercise1 import exercise1


def main():
    """Main function that runs all the exercises."""
    pylog.info('Implementing Lab 4 : Exercise 1')
    exercise1()


if __name__ == '__main__':
    from cmcpack import parse_args
    parse_args()
    main()