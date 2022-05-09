try:
    import Box2D
    from my_gym.envs.box2d.lunar_lander import LunarLander
    from my_gym.envs.box2d.lunar_lander import LunarLanderContinuous
    from my_gym.envs.box2d.bipedal_walker import BipedalWalker, BipedalWalkerHardcore
    from my_gym.envs.box2d.car_racing import CarRacing
except ImportError:
    Box2D = None
