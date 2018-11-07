from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.constraints import maxnorm






def build_convolutional_model(input_shape, class_num):
    

    model = Sequential()
    ### YOUR CODE HERE ###
    # The required model layers are listed.     # You are welcome to add other layers as you see fit     # e.g. pooling, dropout, batch normalization
    # Required step 1: Add at least one convolutional layer     # 1+ lines
    model.add(Conv2D(32, (3, 3), input_shape= input_shape[1:], padding='same', activation='relu', kernel_constraint= maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint= maxnorm(3)))
    model.add(BatchNormalization())   
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))  
    
    model.add(Conv2D(32, (3, 3),  padding='same', activation='relu', kernel_constraint= maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint= maxnorm(3)))
    model.add(BatchNormalization())   
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))  
    
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_constraint= maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint= maxnorm(3)))
    model.add(BatchNormalization())   
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1)) 

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_constraint= maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint= maxnorm(3)))
    model.add(BatchNormalization())   
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0))       
    
    # Required step 2: flatten the activations    # so that we can use fully connected layers     # 1 line
    model.add(Flatten())

    # Required step 3: Add a few fully connected hidden layers that will     # allow us to make predictions.     # 1-2 lines
    model.add(Dense(512, activation='relu', kernel_constraint= maxnorm(3)))

    # Required step 4: Add a final fully connected "readout" layer
    model.add(Dense(class_num, activation='softmax'))  

    ######################
    # The loss function and optimizer are provided for you.
    opt = Adam(lr=0.005)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt, metrics=["acc"])
    return model


def image_augmentation():
    pass
