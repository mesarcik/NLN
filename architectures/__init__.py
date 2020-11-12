# INIT FILE
import sys
sys.path.insert(1,'../')

from .ae import main as train_ae 
from .aae import main as train_aae
from .dae import main as train_dae
from .vae import main as train_vae
from .ganomaly import main as train_ganomaly
from .gpnd import main as train_gpnd
from .bigan import main as train_bigan
from .vaegan import main as train_vaegan
