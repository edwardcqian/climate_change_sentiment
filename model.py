import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding,SpatialDropout1D, Bidirectional, GRU, CuDNNGRU
from keras.layers import Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, Dense, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import clean_text, get_onehot

data = pd.read_csv('data/sample_data.csv')
data = data.dropna(subset=['message']) # remove any empty tweets

# cleaning the raw text data
data['message'] = data['message'].apply(clean_text)

# remove duplicates from data
data = data.drop_duplicates(subset=['message'])

# isolate label and text
y = data['sentiment']
y += 1
X_data = data['message']

# splitting into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size= 0.1, random_state = 1)

# Setup model parameters
EMBEDDING_FILE = "/home/edward/Documents/work/climate/glove.twitter.27B.200d.txt" # path to embedding file 
max_features = 20000                           # we are only interested in top 50k most frequently used words
maxlen = 150                                   # tweets longer than 150 words will be truncated otherwise padded
embed_size = 200                               # size of each vector (must match the size of the glove embedding)

# tokenize training and testing data
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train))

X_tr = tokenizer.texts_to_sequences(X_train)
X_tr = sequence.pad_sequences(X_tr, maxlen=maxlen)
X_ts = tokenizer.texts_to_sequences(X_test)
X_ts = sequence.pad_sequences(X_ts, maxlen=maxlen)

# loading the embedding file (may take some time)
embeddings_index = {}
with open(EMBEDDING_FILE,encoding='utf8') as f:
  for line in f:
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

word_index = tokenizer.word_index

# prepare embedding matrix
num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_index.items():
  if i >= max_features:
    continue
  embedding_vector = embeddings_index.get(word)
  if embedding_vector is not None:
    # words not found in embedding index will be all-zeros.
    embedding_matrix[i] = embedding_vector
  
y_tr_one = get_onehot(y_train, 4)
y_ts_one = get_onehot(y_test, 4) 

# creating the model
gru_input = Input(shape=(maxlen, ))
x = Embedding(max_features, embed_size, weights=[embedding_matrix],trainable = False)(gru_input)
x = SpatialDropout1D(0.2)(x)
x = Bidirectional(GRU(128, return_sequences=True, reset_after=True, recurrent_activation='sigmoid'))(x)
x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
x = concatenate([avg_pool, max_pool]) 
x = Dense(128, activation='relu')(x)
x = Dropout(0.1)(x)
gru_output = Dense(4, activation="sigmoid")(x)
gru_model = Model(gru_input, gru_output)
gru_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# training parameters
epochs = 20 
batch_size = 128
patience = 4    # for early stopping
file_path="weights/gru_final.hdf5" # where to save the best model weights

# creating checkpoint to save the model
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=4)
callbacks_list = [checkpoint, early] #early

# training the model
gru_model.fit(X_tr, y_tr_one, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)

# loading the best weights
gru_model.load_weights(file_path)

# predicting on testing set
class_prob = gru_model.predict(X_ts)
class_pred = np.argmax(class_prob,axis=1)

# some basic metrics for accuracy
print('accuracy score: ' + str(accuracy_score(y_test,class_pred)))
classes = ['Anti','Neutral','Pro','News']
print(classification_report(y_test,class_pred, target_names= classes))

# ploting the confusion matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test,class_pred)
cm = cm/cm.astype(np.float).sum(axis=1)

df_cm = pd.DataFrame(cm, index = classes, columns = classes)
sn.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')