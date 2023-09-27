######################################################################
#DropOut methods implemented by Esmeralda Ruiz                      ##
#Following the Akis paper of dropout center                         ##
#https://link.springer.com/chapter/10.1007/978-3-031-09002-8_42     ##
#C total centers, P percentage of dropout we choose P*C centers     ##
#for training each round where P is a fixed percentage              ##
#Methods implemented:                                               ##
# Aki´s paper:Fast_at_odd_rounds, Fast_every_three,random_dropout   ##
# New method: Less_participants_at_odd_rounds                       ##
######################################################################

import math
from collections import OrderedDict
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from typing import Dict, List, Optional
from logging import INFO
from flwr.common.logger import log
import random

def select_clients(dropout_method, percentage_drop,clients,clients_first_round_time,server_round,clients_num_examples):
    match dropout_method:
        case "Fast_at_odd_rounds":
            clients = Fast_at_odd_rounds(server_round,clients,clients_first_round_time, percentage_drop)

        case "Fast_every_three":
            clients = Fast_every_three(server_round,clients,clients_first_round_time, percentage_drop)

        case "random_dropout":
            clients = random_dropout(server_round,clients,clients_first_round_time, percentage_drop)
            
        case _:
            clients = Less_participants_at_odd_rounds(server_round,clients, clients_num_examples,percentage_drop)

    return clients

#Fast_at_odd_rounds: On the first variant we use only the fastest
#ones every odd round and all the slow ones on the rest of the rounds 
def Fast_at_odd_rounds(server_round,clients_proxys, clients_time,percentageDrop):
    client_ordered_time = OrderedDict(sorted(clients_time.items(), key=lambda x:x[1]))
    client_ordered_time_keys = list(client_ordered_time.keys())
    number_clients = len(clients_time)
    clientToDropOut = math.ceil((percentageDrop/100)*number_clients)
    clientsTokeep = number_clients-clientToDropOut
    client_list_selected = []
    
    #If odd round send fastest  
    if((server_round%2)==0):
        Fast_round = True
        client_list_selected = client_ordered_time_keys[0:clientsTokeep]
    #If even round send slowest  
    else:
        Fast_round = False
        client_list_selected = client_ordered_time_keys[number_clients-clientsTokeep:number_clients]  

    clients_nextround = []
    for client in clients_proxys:
        name_client = client.cid
        if(name_client in client_list_selected):
            clients_nextround.append(client)
            print("Client "+ name_client+ " selected in round " + str(server_round) + \
                  " with time " +str(client_ordered_time[name_client]))
   
    return  clients_nextround #(clients_nextround, Fast_round)  





#Fast_every_three second variant train on the fastest centers every 
#third round, randomly dropping centers on the rest
def Fast_every_three(server_round,clients_proxys, clients_time,percentageDrop):
    client_ordered_time = OrderedDict(sorted(clients_time.items(), key=lambda x:x[1]))
    client_ordered_time_keys = list(client_ordered_time.keys())
    number_clients = len(clients_time)
    clientToDropOut = math.ceil((percentageDrop/100)*number_clients)
    clientsTokeep = number_clients-clientToDropOut
    client_list_selected = []
    
    #Every three send fastest only
    if((server_round%3)==0):
        client_list_selected = client_ordered_time_keys[0:clientsTokeep]
    #the rest  randomly dropping centers on the rest
    else:
        client_list_selected = random.sample(client_ordered_time_keys, clientsTokeep)

    clients_nextround = []
    for client in clients_proxys:
        name_client = client.cid
        if(name_client in client_list_selected):
            clients_nextround.append(client)
            print("Client "+ name_client+ " selected in round " + str(server_round) + \
                  " with time " +str(client_ordered_time[name_client]))
   
    return  clients_nextround  


#Select random clients
def random_dropout(server_round,clients_proxys, clients_time,percentageDrop):
    number_clients = len(clients_proxys)
    clientToDropOut = math.ceil((percentageDrop/100)*number_clients)
    clientsTokeep = number_clients-clientToDropOut
    
    #Select clients randomly
    clients_nextround = random.sample(clients_proxys, clientsTokeep)
   
    return  clients_nextround  


#Less_partipants_at_odd_rounds: we use only the ones for more participants
#in every odd round and the ones with less participants on the rest of the rounds 
def Less_participants_at_odd_rounds(server_round,clients_proxys, clients_num_examples,percentageDrop):
    clients_nextround  = Fast_at_odd_rounds(server_round,clients_proxys, clients_num_examples,percentageDrop)
    return  clients_nextround  


#If we select random sample in dropout, there is no need to modify the
#aggregator just customize the sample of fl.server.client_manager
#and add client_manager= CenterDropoutClientManager()
#to history = fl.server.start_server in main.py
# class CenterDropoutClientManager(fl.server.client_manager.SimpleClientManager):
#     def sample(
#         self,
#         num_clients: int,
#         min_num_clients: Optional[int] = None,
#         criterion: Optional[Criterion] = None,
#     ) -> List[ClientProxy]:
#         """Apply Center Dropout."""

#         # Block until at least num_clients are connected.
#         if min_num_clients is None:
#             min_num_clients = num_clients
#         self.wait_for(min_num_clients)
#         # Sample clients which meet the criterion
#         available_cids = list(self.clients)
#         if criterion is not None:
#             available_cids = [
#                 cid for cid in available_cids if criterion.select(self.clients[cid])
#             ]
#         sampled_cids = random.sample(available_cids, num_clients)
#         return [self.clients[cid] for cid in sampled_cids]

### ====== Center Dropout (Under Development) ====== ###
# class CDCriterion(fl.server.criterion.Criterion):
#     def __init__(self, criterion, dropout_prob):
#         super().__init__()
#         self.criterion = criterion
#         self.dropout_prob = dropout_prob

#     def select():