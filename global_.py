
lr_actor = 0.0003
lr_critic = 0.0003
Iter = 15000
MAX_STEP = 10000
gamma =0.9
lambd = 0.9
batch_size = 32 #64
epsilon = 0.2
l2_rate = 0.001
beta = 3
update_timestep = 256 # update policy every n timesteps
inner_loop_max_steps = 20000


def set_task(task):
    global environment, network, repeat_time, in_count
    if task == '1':
        network = 'NN_ab' # the 2-network
        environment = 'AntBullet-v0'
    elif task == '2':
        network = 'NN_forward'
        environment = 'AntBullet-v0'
    elif task == '3':
        network = 'NN_a_random_b' # only evolve critic network, random action network at each lifecycle. 
        environment = 'AntBullet-v0'
    elif task == '4':
        network = 'NN_forward_hyperNEAT_b'
        environment = 'AntBullet-v0'
    elif task == '5':
        network = 'NN_a_hyperNEAT_b' # evovle critic network and a cppn that generate action network.
        environment = 'AntBullet-v0'
    elif task == '6':
        network = 'NN_forward_part_random'
        environment = 'AntBullet-v0'
    elif task == '7':
        network = 'NN_forward_hebbian' # hebbian network.
        environment = 'AntBullet-v0'
    elif task == '10':
        network = 'NN_rnn'
        environment = 'AntBullet-v0'

    else:
        raise NameError('Didn\'t have this task.')

    if task in ['2','4','6','10']:
        repeat_time = 10
        in_count = 0
    elif task == '7':
        repeat_time = 11
        in_count = 1
    else:
        in_count = 2
        repeat_time = in_count*2
