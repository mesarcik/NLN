import os
from utils import cmd_input 
from reporting import (generate_tables, 
                        generate_residual_maps, 
                        generate_segmenatation_maps, 
                        mvtec_eval)

def report():
    '''
        Checks if the results files have been correctly generated 
            and produces desired plots
    '''
    if ((args.data == 'CIFAR10') or 
        (args.data == 'MNIST') or
        (args.data == 'FASHION_MNIST')):

        if os.path.exists('outputs/results_{}_{}.csv'.format(cmd_input.args.data,
                                                             cmd_input.args.seed)):
            generate_tables(cmd_input.args)
        else:
            raise Exception('Need {} results to call \
                             generate_tables()'.format(cmd_input.args.data))

    elif args.data == 'MVTEC':
        if os.path.exists('outputs/results_{}_{}.csv'.format(cmd_input.args.data,
                                                             cmd_input.args.seed)):
            mvtec_eval(cmd_input.args)
            generate_residual_maps(cmd_input.args)
            generate_segmenatation_maps(cmd_input.args)
        else:
            raise Exception('{} data does not exist'.format(cmd_input.args.data))

    else:
        raise Exception('Can only generate results on\
                            MVTEC/MNIST/CIFAR10 or FMNIST')


if __name__ == '__main__':
    report()
