########################################################################
#Serialization code implemented by Esmeralda Ruiz Pujadas TEMPORALLY  ##
#IMPORTANT REPLACE FOR SOMETHING SAFER                                ##
#that returns BytesIO() forced by the library                         ##
#OR ADD A TOP LAYER OF SECURITY: (e.g., encription)                   ##
#WARNING!!!!!!!!: README                                              ##
#USE CLIENT NOT NUMPY CLIENT TO CUSTOMIZE SERIALIZATION               ##
########################################################################

import numpy as np
from typing import Tuple, Union, List
from io import BytesIO
import numpy.typing as npt
from typing import Any
NDArray = npt.NDArray[Any]
NDArrays = List[NDArray]
from typing import cast

from flwr.common import   Parameters

############
#SERIALIZE #
############
def ndarray_to_bytes_RF(ndarray: NDArray) -> bytes:
    """Serialize NumPy ndarray to bytes."""
    bytes_io = BytesIO()
    np.save(bytes_io, ndarray, allow_pickle=True)  # type: ignore
    return bytes_io.getvalue()

def ndarrays_to_parameters_RF(ndarrays: NDArrays) -> Parameters:
    """Convert NumPy ndarrays to parameters object."""
    tensors = [ndarray_to_bytes_RF(ndarray) for ndarray in ndarrays]
    return Parameters(tensors=tensors, tensor_type="numpy.ndarray")

def serialize_RF(params) -> Parameters:
    parameters_to_ndarrays_final = ndarrays_to_parameters_RF(params)
    return parameters_to_ndarrays_final

##############
#DESERIALIZE #
##############
def bytes_to_ndarray_RF(tensor: bytes) -> NDArray:
    """Deserialize NumPy ndarray from bytes."""
    bytes_io = BytesIO(tensor)
    ndarray_deserialized = np.load(bytes_io, allow_pickle=True)  # type: ignore
    return cast(NDArray, ndarray_deserialized)

def parameters_to_ndarrays_RF(parameters: Parameters) -> NDArrays:
    """Convert parameters object to NumPy ndarrays."""
    return [bytes_to_ndarray_RF(tensor) for tensor in parameters.tensors]

def deserialize_RF(params) -> Parameters:
    parameters_to_ndarrays_final = parameters_to_ndarrays_RF(params)
    return parameters_to_ndarrays_final


##############################   

#pickled = codecs.encode(pickle.dumps(params), "base64").decode()
#params = codecs.encode(pickle.dumps(params), "base64").decode()
#p = Parameters(tensors=pickled, tensor_type="numpy.ndarray")
#p = Parameters(tensors=bytes, tensor_type="numpy.ndarray")
#a = parameters_to_proto(parameters_to_ndarrays_final) 
#pickled =  pickle.loads(codecs.decode(pickled.encode(), "base64"))

##############################  