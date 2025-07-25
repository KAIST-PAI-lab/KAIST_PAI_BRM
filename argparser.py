import argparse

def argparser():
    parser=argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Run the Number Line Estimation Experiment')
    parser.add_argument('--n_trials', type=int, default=45, help='Number of trials in the NLE block')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    ## Those related to gpr_instance.py
    parser.add_argument('--n_kernels', type=int, default=1, help='Number of individual kernels to be combined')
    parser.add_argument('--return_std', type=bool, default=True, help='A mask indicating whether to return standard deviation of posterior distribution at each query value.')
    parser.add_argument('--return_cov', type=bool, default=False, help='A mask indicating whether to return covaraince matrix of posterior distribution at each query value.')
    parser.add_argument('--alpha', default=1e-10, help='')
    args = parser.parse_args()