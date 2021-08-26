# INIT FILE
import sys
sys.path.insert(1,'../')

from .ae import main as train_ae 
from .resnet import main as train_resnet
from .ae_ssim import main as train_ae_ssim 
from .ae_resnet import main as train_ae_resnet
from .nnae import main as train_nnae
from .aae import main as train_aae
from .dae import main as train_dae
from .vae import main as train_vae
from .ganomaly import main as train_ganomaly
