import argparse

def get_config(notebook=False):
    # notebook variable allows the configuration to be imported in a jupyter notebook, ignoring command line parameters

    # get the parameters
    parser = argparse.ArgumentParser(description='PEV Battery Charging multi-agent environment.')

    # prepare
    parser.add_argument("--algorithm_name", type=str, default='mappo_gru')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", action='store_false', default=True)
    parser.add_argument("--cuda_deterministic", action='store_false', default=True)
    parser.add_argument("--n_training_threads", type=int, default=12)
    parser.add_argument("--n_rollout_threads", type=int, default=1)
    parser.add_argument("--num_env_steps", type=int, default=10e6, help='number of environment steps to train (default: 10e6)') 
    
    # env
    parser.add_argument("--env_name", type=str, default='BatteryCharge')
    parser.add_argument("--scenario_name", type=str, default='PEVChargeSchedule')
    parser.add_argument("--num_agents", type=int, default=6, help='Number of charging stations to control.')
    parser.add_argument("--share_reward", action='store_false', default=False)
    
    # PEV Charge Environment
    
    parser.add_argument("--n_pevs", type=int, default=10, help='Number of PEVs to schedule during training')
    parser.add_argument("--soc_max", type=float, default=24, help='Maximum SOC capacity by PEV.')
    parser.add_argument("--soc_ref", type=float, default=24, help='Reference SOC goal per PEV.')
    parser.add_argument("--soc_initial", type=float, default=0, help='Initial State of Charge of PEV')
    parser.add_argument("--p_min", type=float, default=0, help='Minimum power supply capacity by charging station.')
    parser.add_argument("--p_max", type=float, default=22, help='Maximum power supply capacity by charging station.')
    parser.add_argument("-ctd", "--charge_time_desired", type=int, default=180, help='Charge Time Desired by PEV (in minutes)')
    parser.add_argument("--xi", type=float, default=0.1, help='PEV conversion losses.')
    parser.add_argument("--P_min", type=float, default=0, help='Minimum power supply capacity by load area.')
    parser.add_argument("--P_max", type=float, default=200, help='Maximum power supply capacity by load area.')
    parser.add_argument("--P_ref", type=float, default=31.5, help="Referece power supply bound by load area. Sum of stations' powers should not exceed this value.")
    parser.add_argument("--sampling_time", type=int, default=5, help='Sampling time (Delta_t).')
    parser.add_argument("--total_time", type=int, default=480, help='Total time (minutes) of the simulation.')
    ### Parameters of random load distribution
    parser.add_argument("--initial_charge_max", type=float, default=0.5, help='Maximum percentage of value to start charge wrt. soc_max.')
    parser.add_argument("--charge_duration_tolerance", type=float, default=0.2, help='Tolerance on the maximum duration of the charge value.')
    parser.add_argument("--random_start_coeff", type=float, default=1, help='To randomize the start time, from a point between it and the next random_start_coeff elements.')
    # reward weights. Pass as (example): -rw 1 1 0.5 1 3
    parser.add_argument("-rw", "--reward_weights", nargs=3, default=[1,2,2], \
    help='Weights for reward components. 0: Penalize on remaining SOC. 1: Surpassing local limit. 2: Surpassing global limit.')
    
    
    # network
    parser.add_argument("--share_policy", action='store_false', default=True, help='agent share the same policy')
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--layer_N", type=int, default=1)
    parser.add_argument("--use_ReLU", action='store_false', default=True)
    parser.add_argument("--use_common_layer", action='store_true', default=False)
    parser.add_argument("--use_popart", action='store_false', default=True)
    parser.add_argument("--use_feature_popart", action='store_true', default=False)
    parser.add_argument("--use_feature_normlization", action='store_false', default=True)   
    parser.add_argument("--use_orthogonal", action='store_false', default=True) 
    parser.add_argument("--use_same_dim", action='store_true', default=False)  
    
    # lstm
    parser.add_argument("--naive_recurrent_policy", action='store_true', default=False, help='use a naive recurrent policy')
    parser.add_argument("--recurrent_policy", action='store_false', default=True, help='use a recurrent policy')
    parser.add_argument("--recurrent_N", type=int, default=1) #TODO now only 1 is support
    parser.add_argument("--data_chunk_length", type=int, default=10)
    
    # attn
    parser.add_argument("--attn", action='store_true', default=False)  
    parser.add_argument("--attn_only_critic", action='store_true', default=False)    
    parser.add_argument("--attn_N", type=int, default=1)
    parser.add_argument("--attn_size", type=int, default=64)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_average_pool", action='store_false', default=True)      
    
    # ppo
    parser.add_argument("--ppo_epoch", type=int, default=15, help='number of ppo epochs (default: 4)')    
    parser.add_argument("--use_clipped_value_loss", action='store_false', default=True)
    parser.add_argument("--clip_param", type=float, default=0.2, help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1, help='number of batches for ppo (default: 32)')   
    parser.add_argument("--entropy_coef", type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float, default=1, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--lr", type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument("--eps", type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--gain", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--use-max-grad-norm", action='store_false', default=True)
    parser.add_argument("--max-grad-norm", type=float, default=20.0, help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use-gae", action='store_false', default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae-lambda", type=float, default=0.95, help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use-proper-time-limits", action='store_true', default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', default=True)
    parser.add_argument("--use_value_high_masks", action='store_false', default=True)
    parser.add_argument("--huber_delta", type=float, default=10.0)   
    
    # replay buffer
    parser.add_argument("--episode_length", type=int, default=95, help='number of forward steps in A2C (default: 5)')

    # run
    parser.add_argument("--use-linear-lr-decay", action='store_true', default=False, help='use a linear schedule on the learning rate')
    
    # save
    parser.add_argument("--save_interval", type=int, default=150)
    
    # log in tb
    parser.add_argument("--log_interval", type=int, default=5)    
    # log in console (should be multiple of log_interval)
    parser.add_argument("--log_console", type=int, default=20)
    
    # eval
    parser.add_argument("--eval", action='store_true', default=False)
    parser.add_argument("--eval_interval", type=int, default=25)
    parser.add_argument("--eval_episodes", type=int, default=32)
    
    # render
    parser.add_argument("--save_gifs", action='store_true', default=True)
    parser.add_argument("--save_gifs_interval", type=int, default=200)

    parser.add_argument("--ifi", type=float, default=0.333333)
    parser.add_argument("--model_dir", type=str, default=None)
    
    # colab
    parser.add_argument("--colab", action='store_true', default=False)

    # restore
    parser.add_argument("--restore_model", action='store_true', default=False)
    parser.add_argument("--restore_model_path", type=str, default=None, help="It must lead to the curr_run folder, like : 'results/MPE/simple_spread/mappo_gru/run6'")

    if notebook:
      args = parser.parse_args("")
    else:
      args = parser.parse_args()

    return args