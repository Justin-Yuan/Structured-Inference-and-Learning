import pickle 

with open('atis.fold0.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')

print(data)

print(type(data))
