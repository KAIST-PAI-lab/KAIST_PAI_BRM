import argparse

def argparser():
    parser = argparse.ArgumentParser(description='Run the Number Line Estimation Experiment.')
    parser.add_argument('--n_trials', type=int, default=45, help='Number of trials in the NLE block.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    
    ## Arguments related to gpr_instance.py
    parser.add_argument('--n_kernels',default=1, help='The number of individual kernels to be combined.')
    parser.add_argument('--alpha', default=1e-10, help='A value added to the diagonal of the kernel matrix during fitting.')
    parser.add_argument('--normalize_y', default=True, help='A binary mask indicating whether to normalize the target values while fitting.')
    parser.add_argument('--n_restarts_optimizer', default=0, help='The number of restarts of the optimizer to find the optimal kernel parameters.')
    parser.add_argument('--type_kernels_index', default=[0,6], help='A list of indices of kernels to be combined.')
    parser.add_argument('--parameters_list', default=[[1.0, 'fixed'],[1.0, 'fixed']], help='A list of list of arguments to be fed to each kernel.')
    parser.add_argument('--multiplied_indices', default=[[0,1]], help='A list of lists indicating the kernels to be multiplied.')
    parser.add_argument('--summed_indices', default=[[]], help='A list of lists indicating the kernels to be summed.')
    parser.add_argument('--gpr_random_state', default=None, help='A parameter determining random number generation in initializing the centers.')
    
    ## Arguments related to gpr_optimize.py
    parser.add_argument('--return_std', default=True, help='A binary mask indicating whether to return standard deviation of posterior distribution at each query value.')
    parser.add_argument('--return_cov', default=False, help='A binary mask indicating whether to return covaraince matrix of posterior distribution at each query value.')
    
    ## Arguments related to experiment.py
    parser.add_argument('--optim_mode', default='GPAL', help='Optimization algorithm to use.')
    parser.add_argument('--base_model', default=None, help='A base model for ADO optimization.')
    parser.add_argument('--save_results_dir', default='../saved_results', help='A directory to store the task results.')
    parser.add_argument('--save_models_dir', default='../saved_models', help='A directory to store the trained Gaussian process regressor model.')
    parser.add_argument('--enable_gpu', type=bool, default=False, help='A binary mask indicating whether to use GPU.')
    parser.add_argument('--subject_prefix', type=str, default='Subject', help='A prefix attached to the unique subject ID to construct a full indicator of each subject.')
    
    ## Arguments related to plotting
    parser.add_argument('--figure_size_1', default=(8,6), help='A size of the figure #1, visualizing the estimates and associated standard deviation.')
    parser.add_argument('--label_x_1', default='Given Number', help='A label for the x-axis of the figure #1.')
    parser.add_argument('--label_y_1', default='Number Estimate', help='A label for the y-axis of the figure #1.')
    parser.add_argument('--title_1', default="Given Number Estimates (2D)", help='A title of the figure #1.')
    parser.add_argument('--sigma_coef_1', default=1.0, help='A coefficient multiplied to the standard deviation, determining the range of uncertainty to be plotted.')
    parser.add_argument('--figure_size_2', default=(8,6), help='The size of the figure #2, visualizing the 3D estimation plot for 2D number line task.')
    parser.add_argument('--label_x_2', default='Given Number', help='A label for the x-axis of the figure #2.')
    parser.add_argument('--label_y_2', default='Upper Bound', help='A label for the y-axis of the figure #2.')
    parser.add_argument('--label_z_2', default='Number Estimate', help='A label for the z-axis of the figure #2.')
    parser.add_argument('--title_2', default='Given Number Estimates (3D)', help='The title of the figure #2.')
    parser.add_argument('--label_x_3', default='Given Number', help='A label for the x-axis of the figure #3, a histogram illustrating the frequencies each design was selected by the optimization algorithm.')
    parser.add_argument('--label_y_3', default='Upper Bound', help='A label for the y-axis of the figure #3.')
    parser.add_argument('--label_z_3', default='Frequency', help='A label for the z-axis of the figure #3.')
    parser.add_argument('--title_3', default='Design Seelction Frequencies', help='The title of the figure #3.')


    ## Arguments related to ADO
    
    
    ## return args
    return parser.parse_args()