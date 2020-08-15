import sys
sys.path.append('../')

from src import *
from src.utils import prog

def run_project(args):
    prog.demo()

if __name__ == '__main__':
    run_project(sys.argv)