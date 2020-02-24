import h5py
import pickle

f = pickle.load(open('./data/CK_data.h5', 'rb'))
train = f['train']
print(f)
print(f.keys)