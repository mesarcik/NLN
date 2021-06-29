#INIT FILE
import sys 
sys.path.insert(1,'../')

from .plot_neighbours import plot_neighs
from .tables import generate_tables
from .residuals import main as generate_residual_maps 
from .segmentation import main as generate_segmenatation_maps 
from .mvtec_eval import main as mvtec_eval 

