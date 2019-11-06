from importlib import import_module
from pathlib import Path

# Imports all loader modules. New modules are registered automatically,
# as long as they adhere to the naming convention *_loader.py
loaders = {}
DIR_NAME = __name__ if __name__ != "__main__" else 'data'
module_paths = Path(DIR_NAME).glob('*_loader.py')
for path in module_paths:
    name = path.stem.split('_')[0]
    import_module(f'{DIR_NAME}.{path.stem}')
    loaders[name] = locals()[f'{path.stem}']


def load(dataset):
    """Loads the preprocessed dataset for the given platform"""
    return loaders[dataset].load_dataset()
