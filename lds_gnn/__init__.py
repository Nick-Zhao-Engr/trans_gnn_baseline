import sys

try:
    import far_ho as far
except ImportError as e:
    print('Please install FAR-HO, available at https://github.com/lucfra/FAR-HO', file=sys.stderr)
    sys.exit()
try:
    import gcn
except ImportError as e:
    print('Please install GCN, available at https://github.com/tkipf/gcn', file=sys.stderr)
    sys.exit()

from lds_gnn.lds import *
from lds_gnn.hyperparams import *
from lds_gnn import utils, models, data
