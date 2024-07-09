import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout, Dense, Flatten, Reshape, LayerNormalization

kernel_initializer = 'he_uniform'

def kan_layer(x, units):
    x = Flatten()(x)
    x = Dense(units, activation='relu')(x)
    x = Reshape((1, 1, 1, units))(x)
    return x

def tok_kan_block(x, filters):
    x = Conv3D(filters, (1, 1, 1), padding='same')(x)
    x = LayerNormalization()(x)
    x = kan_layer(x, filters)
    x = Conv3D(filters, (3, 3, 3), padding='same', groups=filters)(x)
    x = LayerNormalization()(x)
    return x

def U_KAN(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
    s = inputs

    #Contraction path
    #convultional layer with 16 filters, kernel size of 3x3x3, relu activation function and he_uniform kernel initializer , padding same to keep the same size of the image
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(s)
    #dropout layer to prevent overfitting, which randomly sets 10% of the input units to 0 at each update during training time
    c1 = Dropout(0.1)(c1)
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
    #maxpooling layer with pool size of 2x2x2 to reduce the size of the image by factor of 2,which downsamples the feature map, also helps in reducing the computational cost and prevent overfitting
    p1 = MaxPooling3D((2, 2, 2))(c1)
    
    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)

    c3 = tok_kan_block(p2, 64)
    p3 = MaxPooling3D((2, 2, 2))(c3)
     
    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
    p4 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(c4)
    print("p4 shape:", p4.shape)

    #deepest layer / bottleneck layer
    c5 = tok_kan_block(p4, 256)
    
    #Expansive path 
    #Tranpose convolutional layer / deconvolutional layer 
    #it upsamples the feature map by a factor of 2, which helps in increasing the size of the image
    #Stride determines how much the window is moved in each step. A stride of 2 in each dimension doubles the spatial dimensions.
    u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    print("u6 shape:", u6.shape)
    print("c4 shape:", c4.shape)
    #helps to preserve spatial information lost during downsamplin by combining u6 and c4
    u6 = concatenate([u6, c4])
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)
     
    u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    print("u7 shape:", u7.shape)
    print("c3 shape:", c3.shape)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)

    c7 = tok_kan_block(c7, 64)
     
    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)
     
    u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)
     
    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    
    return model