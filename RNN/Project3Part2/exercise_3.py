import os
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Embedding, BatchNormalization, Dropout , SimpleRNN
from keras.layers import Reshape
from keras.optimizers import  Adam


def create_word_embeddings(embeddings_path, vocab, dim=300, save=True):
    embeddings = np.zeros([len(vocab), dim])

    with open(
      os.path.join(embeddings_path, "glove.6B.{}d.txt".format(dim))) as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in vocab:
                embeddings[vocab[word]] = np.asarray(values[1:],
                                                     dtype="float32")

    if save:
        np.savez_compressed("glove.6B.{}d.trimmed".format(dim),
                            embeddings=embeddings)
    return embeddings


def build_dense_model(embedding_matrix=None, max_len=2494, dim=300):
    model = Sequential()

    # Word embeddings
    if embedding_matrix is not None:
        model.add(Embedding(input_dim=embedding_matrix.shape[0],
                            output_dim=embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            trainable=False))
    else:
        model.add(Embedding(input_dim=88584,
                            output_dim=300))

    ### YOUR CODE HERE ###
    # Step 1: Change your embedding so that it can fed into a fully connected
    #  layer. (Hint: Think about the dimensionality going in and how many
    # dimensions you need in a fully-connected layer)
    # 1-2 lines
    # Step 2: Build your fully connected NN. Remember that you need 2 outputs.
    # 2-3 lines

    model.add(Reshape((max_len*dim, )))
    
    model.add(Dense(300, input_dim= max_len*dim , activation  = 'relu'))
    model.add(BatchNormalization())
 
    model.add(Dense(30, activation  = 'relu'))
    model.add(BatchNormalization())
     
    model.add(Dense(2, activation='softmax'))

    ######################
    opt = Adam(lr=0.01)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt, metrics=["acc"])
    return model
    
    
    

def build_simpleRNN_model(embedding_matrix=None):
    model = Sequential()

    # Word embeddings
    if embedding_matrix is not None:
        model.add(Embedding(input_dim=embedding_matrix.shape[0],
                            output_dim=embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            trainable=False))
    else:
        model.add(Embedding(input_dim=88584,
                            output_dim=300))

    ### YOUR CODE HERE ###
    # Step 1: Add an LSTM layer with however many units you want
    # 1-2 lines
    # Step 2: Add a densely connected network with 2 output units.
    # 2-3 lines

    model.add(SimpleRNN(100,  dropout = 0,  recurrent_dropout = 0))
    model.add(Dense(2, activation = "softmax"))
 
    ######################
    opt = Adam(lr=0.01)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt, metrics=["acc"])
    return model


def build_lstm_model(embedding_matrix=None):
    model = Sequential()

    # Word embeddings
    if embedding_matrix is not None:
        model.add(Embedding(input_dim=embedding_matrix.shape[0],
                            output_dim=embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            trainable=False))
    else:
        model.add(Embedding(input_dim=88584,
                            output_dim=300))

    ### YOUR CODE HERE ###
    # Step 1: Add an LSTM layer with however many units you want
    # 1-2 lines
    # Step 2: Add a densely connected network with 2 output units.
    # 2-3 lines
    model.add(LSTM(100,  dropout = 0,  recurrent_dropout = 0))
    model.add(Dense(2, activation = "softmax"))

    ######################
    opt = Adam(lr=0.01)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt, metrics=["acc"])
    return model



def build_bidirectional_lstm_model(embedding_matrix=None):

    model = Sequential()

    # Word embeddings
    if embedding_matrix is not None:
        model.add(Embedding(input_dim=embedding_matrix.shape[0],
                            output_dim=embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            trainable=False))
    else:
        model.add(Embedding(input_dim=88584,
                            output_dim=300))

    ### YOUR CODE HERE ###
    # Step 1: Add bidirectional LSTM layer with however many units you want
    # 1+ lines
    # Step 2: Add a densely connected network with 2 output units.
    # 2-3 lines
    model.add(Bidirectional(LSTM(100,  dropout = 0,  recurrent_dropout = 0)))
    model.add(Dense(2, activation = "softmax"))    

    ######################
    opt = Adam(lr=0.01)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt, metrics=["acc"])
    return model


def build_final_model(embedding_matrix=None):

    model = Sequential()

    # Word embeddings
    if embedding_matrix is not None:
        model.add(Embedding(input_dim=embedding_matrix.shape[0],
                            output_dim=embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            trainable=False))
    else:
        model.add(Embedding(input_dim=88584,
                            output_dim=300))

    ### YOUR CODE HERE ###
    # replace "pass" with your model (you can copy from above)
    # pass
    #model.add(Bidirectional(LSTM(50,  dropout = 0.2,  recurrent_dropout = 0.1)))   # loss: 0.6531 - acc: 0.5976 - val_loss: 0.5956 - val_acc: 0.6811    total       0.69      0.68      0.68
    #model.add(Bidirectional(LSTM(100,  dropout = 0.2,  recurrent_dropout = 0.2)))
    #model.add(Bidirectional(LSTM(60,  dropout = 0.1,  recurrent_dropout = 0.1))) #loss: 0.6725 - acc: 0.5775 - val_loss: 0.6269 - val_acc: 0.6378      total       0.65      0.63      0.62 
    #model.add(Bidirectional(LSTM(30,   return_sequences=True, dropout = 0.2,  recurrent_dropout = 0.1)))   
    #model.add(Bidirectional(LSTM(10,  dropout = 0.2,  recurrent_dropout = 0.1)))   # loss: 0.6592 - acc: 0.5941 - val_loss: 0.6087 - val_acc: 0.6720     total       0.71      0.67      0.65 
    model.add(Bidirectional(LSTM(50,   return_sequences=True, dropout = 0.2,  recurrent_dropout = 0.1)))   
    model.add(Bidirectional(LSTM(6,  dropout = 0,  recurrent_dropout = 0)))   # loss: 0.6592 - acc: 0.5941 - val_loss: 0.6087 - val_acc: 0.6720     total       0.71      0.67      0.65 
    model.add(Dense(2, activation = "softmax"))    

    ######################
    opt = Adam(lr=0.01)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt, metrics=["acc"])   
    

    return model
