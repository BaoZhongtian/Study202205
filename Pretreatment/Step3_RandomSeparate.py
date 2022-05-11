import json
import numpy
from Loader_WikiLingual import Loader_WikiLingual, Loader_WikiLingual_Both

if __name__ == '__main__':
    target_lingual = 'portuguese+spanish'

    # treat_data = Loader_WikiLingual(target_lingual)
    treat_data = Loader_WikiLingual_Both('portuguese', 'spanish')
    sample_ids = numpy.arange(len(treat_data))
    numpy.random.shuffle(sample_ids)

    train_ids = sample_ids[0:int(0.95 * len(sample_ids))].tolist()
    test_ids = sample_ids[int(0.95 * len(sample_ids)):].tolist()
    json.dump({'train_ids': train_ids, 'test_ids': test_ids}, open('%sRandomSeparateIds.json' % target_lingual, 'w'))
