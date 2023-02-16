from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# define example
data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
values = array(data)
print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
# invert first example
the_index_in_the_one_hot_array = 8
the_nubmber_again = label_encoder.inverse_transform([argmax(onehot_encoded[the_index_in_the_one_hot_array, :])])[0]
print(the_nubmber_again)