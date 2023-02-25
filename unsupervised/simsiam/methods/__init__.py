from .SimSiam.simsiam import SimSiamModel

def set_model(method, arch, dataset):
    if method == 'simclr':
        raise('Please do not use simclr in this method')
    elif method == 'moco':
        raise('Please do not use moco in this method')
    elif method == 'simsiam':
        return SimSiamModel(method, arch, dataset) 

