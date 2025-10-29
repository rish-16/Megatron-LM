from torch.optim import Adam
from megatron.core.hypercore.optimizers.radam import RiemannianAdam
from megatron.core.hypercore.optimizers.rsgd import RiemannianSGD
from megatron.core.hypercore.optimizers.initialize import Optimizer
from megatron.core.hypercore.optimizers.initialize import LR_Scheduler