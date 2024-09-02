from datasets import load_dataset
import random
import numpy as np

np.random.seed(101)

dataset = "cola"

seq_keys = {
    'cola': 'sentence',
    'sst2': 'sentence',
    'rotten_tomatoes': 'text'
}
seq_key = seq_keys[dataset]

if dataset in ['cola', 'sst2']:
    full = load_dataset('glue', dataset)['train']
else:
    full = load_dataset(dataset)['train']

idxs = list(range(len(full)))
np.random.shuffle(idxs)
if dataset == 'cola':
    assert idxs[0] == 2310  # with seed 101

n_samples = 128

sentences = []
labels = []
for i in range(n_samples):
    sentences.append(full[idxs[i]][seq_key])
    labels.append(full[idxs[i]]['label'])
#
# if split == 'test':
#     assert n_samples <= 1000
#     idxs = idxs[:n_samples]
# elif split == 'val':
#     idxs = idxs[1000:]  # first 1000 saved for testing
#     assert len(idxs) >= n_samples
#
#     zipped = [(idx, len(full[idx][seq_key])) for idx in idxs]
#     zipped = sorted(zipped, key=lambda x: x[1])
#     chunk_sz = len(zipped) // n_samples
#     idxs = []
#     for i in range(n_samples):
#         tmp = chunk_sz * i + np.random.randint(0, chunk_sz)
#         idxs.append(zipped[tmp][0])
#     np.random.shuffle(idxs)
with open(f"data/{dataset}_data_{n_samples}.txt", "w") as f:
    for i in range(len(sentences)):
        f.write(f"{sentences[i]}\n")
        f.write(f"{labels[i]}\n")