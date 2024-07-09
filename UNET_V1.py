from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, Dropout
################################################################
kernel_initializer =  'he_uniform'
act='relu'
pad='same'
def UNET(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
    s = inputs
    #Contraction path
    c1 = Conv3D(16, (3, 3, 3), activation=act, kernel_initializer=kernel_initializer, padding=pad)(s)
    # 16 = filter/kernel --> produces 16 feature maps 
    # each filter of size 3x3x3
    c1 = Dropout(0.1)(c1)
    c1 = Conv3D(16, (3, 3, 3), activation=act, kernel_initializer=kernel_initializer, padding=pad)(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)
    
    c2 = Conv3D(32, (3, 3, 3), activation=act, kernel_initializer=kernel_initializer, padding=pad)(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv3D(32, (3, 3, 3), activation=act, kernel_initializer=kernel_initializer, padding=pad)(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)
     
    c3 = Conv3D(64, (3, 3, 3), activation=act, kernel_initializer=kernel_initializer, padding=pad)(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv3D(64, (3, 3, 3), activation=act, kernel_initializer=kernel_initializer, padding=pad)(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)
     
    c4 = Conv3D(128, (3, 3, 3), activation=act, kernel_initializer=kernel_initializer, padding=pad)(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv3D(128, (3, 3, 3), activation=act, kernel_initializer=kernel_initializer, padding=pad)(c4)
    p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)
    #Bottleneck
    c5 = Conv3D(256, (3, 3, 3), activation=act, kernel_initializer=kernel_initializer, padding=pad)(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv3D(256, (3, 3, 3), activation=act, kernel_initializer=kernel_initializer, padding=pad)(c5)
    
    #Expansive path 
    u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding=pad)(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv3D(128, (3, 3, 3), activation=act, kernel_initializer=kernel_initializer, padding=pad)(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv3D(128, (3, 3, 3), activation=act, kernel_initializer=kernel_initializer, padding=pad)(c6)
     
    u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding=pad)(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(64, (3, 3, 3), activation=act, kernel_initializer=kernel_initializer, padding=pad)(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv3D(64, (3, 3, 3), activation=act, kernel_initializer=kernel_initializer, padding=pad)(c7)
     
    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding=pad)(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(32, (3, 3, 3), activation=act, kernel_initializer=kernel_initializer, padding=pad)(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv3D(32, (3, 3, 3), activation=act, kernel_initializer=kernel_initializer, padding=pad)(c8)
     
    u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding=pad)(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv3D(16, (3, 3, 3), activation=act, kernel_initializer=kernel_initializer, padding=pad)(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv3D(16, (3, 3, 3), activation=act, kernel_initializer=kernel_initializer, padding=pad)(c9)    
    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(c9)     
    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    
    return model

#Test if everything is working ok. 
model = UNET(128, 128, 128, 4, 4)

# (length, width , height, channels , label classes)
# Total params: 5,645,828
# Trainable params: 5,645,828
