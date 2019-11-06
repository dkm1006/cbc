from data import (
    twitter_loader, formspring_loader, wikipedia_loader, jigsaw_loader
)

datasets = {
    'twitter': twitter_loader.load_dataset(),
    'formspring': formspring_loader.load_dataset(),
    'wikipedia': wikipedia_loader.load_dataset(),
    'jigsaw': jigsaw_loader.load_dataset()
}
