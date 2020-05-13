import sys

from utils import prog

def run_project(args):
    #options = Options()
    #options.parse(args[1:])
    #project = Project(options)
    #print ('Printing date: {}'.format( project.date()))
    #print ('Printing example arg: {}'.format(project.print_example_arg()))
    prog.demo()

if __name__ == '__main__':
    run_project(sys.argv)