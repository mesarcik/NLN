import os
from utils import cmd_input 
from reporting import heatmap, barplot, barplot_shared

def report():
    '''
        Checks if the results files have been correctly generated 
            and produces desired plots
    '''

    if os.path.exists('outputs/results_{}.csv'.format(cmd_input.args.data)):
        #heatmap(cmd_input.args.data)
        #barplot(cmd_input.args.data)
        #generate_tables(cmd_input.args)
    else:
        raise Exception('Need {} results to call \
                         heatmat() and barplot()'.format(cmd_input.args.data))

    if (os.path.exists('outputs/results_MNIST.csv') and 
        os.path.exists('outputs/results_CIFAR10.csv')):
    
        barplot_shared()
    else:
        raise Exception(' Need both CIFAR and MNIST results to call barplot')


if __name__ == '__main__':
    report()
