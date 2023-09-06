import flwr as fl
import torch as T
import numpy as np
import torchbnn as bnn

from collections import OrderedDict
from typing import Dict, List, Tuple

import flcore.datasets as ds

def accuracy_quick( model, dataset ):
    n = len( dataset )
    X = dataset[ 0:n ][ 'predictors' ]
    Y = T.flatten( dataset[ 0:n ][ 'labels' ] )
    with T.no_grad():
        oupt = model( X )
    arg_maxs = T.argmax( oupt, dim = 1 )
    num_correct = T.sum( Y == arg_maxs )
    acc = ( num_correct * 1.0 / len( dataset ) )
    return acc.item()

class BayesianNet( T.nn.Module ):
    def __init__( self, inlayer, outlayer ):
        super( BayesianNet, self ).__init__()
        self.hid1 = bnn.BayesLinear( 
            prior_mu     = inlayer[ 'prior_mu' ],
            prior_sigma  = inlayer[ 'prior_sigma' ],
            in_features  = inlayer[ 'in_features' ], 
            out_features = inlayer[ 'out_features' ]
        )
        self.oupt = bnn.BayesLinear(
            prior_mu     = outlayer[ 'prior_mu' ],
            prior_sigma  = outlayer[ 'prior_sigma' ],
            in_features  = outlayer[ 'in_features' ],
            out_features = outlayer[ 'out_features' ]
        )

    def forward(self, x):
        z = T.relu( self.hid1( x ) )
        z = self.oupt( z )
        return z


class FLClient( fl.client.NumPyClient ):
    def __init__( self, data, model, optimizer, train_loader, accuracy_fun = None ):
        self.model        = model
        self.train_loader = train_loader
        self.optimizer    = optimizer
        self.max_epochs   = 100
        if accuracy_fun is not None:
            self.accuracy_fun = accuracy_fun
        else:
            self.accuracy_fun = accuracy_quick
        self.train_ds     = data[ 'train' ]
        self.test_ds      = data[ 'test' ]

    def get_parameters( self, config ) -> List[ np.ndarray ]:
        x = self.model.state_dict().items()
        return [ val.cpu().numpy() for _, val in self.model.state_dict().items() ]

    def set_parameters( self, parameters: List[ np.ndarray ] ) -> None:
        params_dict = zip( self.model.state_dict().keys(), parameters )
        state_dict = OrderedDict( { k: T.tensor(v) for k, v in params_dict } )
        self.model.load_state_dict( state_dict, strict = True )

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict ]:
        ce_loss = T.nn.CrossEntropyLoss()
        kl_loss = bnn.BKLLoss( reduction = 'mean', last_layer_only = False )
        self.set_parameters(parameters)
        self.model.train()
        for _ in range( 0, self.max_epochs ):
            epoch_loss = 0
            for ( _, batch ) in enumerate( self.train_loader ):
                X = batch[ 'predictors' ]
                Y = batch[ 'species' ]
                self.optimizer.zero_grad()
                oupt = self.model( X )
                cel = ce_loss( oupt, Y )
                kll = kl_loss( self.model )
                tot_loss = cel + ( 0.10 * kll )
                epoch_loss += tot_loss.item()
                tot_loss.backward()
                self.optimizer.step()
        return self.get_parameters( config = {} ), -1, {}
    def evaluate(
        self, parameters: List[ np.ndarray ], config: Dict[ str, str ]
    ) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        #loss, metrics = test_routine(self.model, self.criterion, self.val_loader, device=self.device)
        loss = 0.0
        acc = accuracy_quick( self.model, self.train_ds )
        return loss, len( self.train_ds ), { 'accuracy': float( acc ) }


def get_client( data, device = T.device( 'cpu' ) ) -> fl.client.Client:
    net = BayesianNet( data[ 'inlayer' ], data[ 'outlayer' ] ).to( device )
    optimizer = T.optim.Adam( net.parameters(), lr = 0.01 )
    train_ldr = T.utils.data.DataLoader(
       data[ 'train_ds' ],
       batch_size = data[ 'batch_size' ], 
       shuffle    = True
    )
    if 'accuracy_fun' in data.keys():
        return FLClient( net, optimizer, train_ldr, device, accuracy_fun = data[ 'accuracy_fun' ] )
    else:
        return FLClient( net, optimizer, train_ldr, device )


if __name__ == "__main__":
    np.random.seed( 1 )
    T.manual_seed( 1 )
    np.set_printoptions( precision = 4, suppress = True, sign = " " )
    np.set_printoptions( formatter = { 'float': '{: 0.4f}'.format } )

    device = T.device( 'cpu' )
    train_ds = ds.load_iris()

    bat_size = 4
    train_ldr = T.utils.data.DataLoader(
       train_ds,
       batch_size = bat_size, 
       shuffle=True
    )

    net = BayesianNet().to( device )
    optimizer = T.optim.Adam( net.parameters(), lr = 0.01 )

    client = FLClient( net, optimizer, train_ldr, device )
    fl.client.start_numpy_client( server_address = f"0.0.0.0:8080", client = client )
