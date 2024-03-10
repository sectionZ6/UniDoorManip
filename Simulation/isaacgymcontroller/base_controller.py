from abc import abstractclassmethod
from env.base_env import BaseEnv
from manipulation.base_manipulation import BaseManipulation
from logging import Logger

class BaseController :

    def __init__(
            self,
            env : BaseEnv,
            manipulation : BaseManipulation,
            cfg : dict,
            logger : Logger
        ) :

        self.env = env
        self.manipulation = manipulation
        self.controller = None
        self.cfg = cfg
        self.logger = logger
    
    @abstractclassmethod
    def run(self) :

        pass