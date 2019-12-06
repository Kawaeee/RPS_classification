import os 
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

import tensorflow.keras

from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from efficientnet.tfkeras import EfficientNetB3
from efficientnet.tfkeras import preprocess_input

# Get test set report
def f1_score(test_y, pred_y):
    return classification_report(test_y, pred_y,digits=4)

# Model instantiation
def create_model():
    base_model = EfficientNetB3(weights='imagenet', include_top=False)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    predictions = layers.Dense(3, activation='softmax')(x)
      
    model = Model(inputs=base_model.input, outputs=predictions)

    return model

batch_size = 16
workers = 2
image_size = 300

dataset_path = '/data'
print(os.listdir(dataset_path))

# Image augmentation
train_datagen = ImageDataGenerator(
                             width_shift_range=0.05, 
                             height_shift_range=0.05, 
                             rotation_range=15, 
                             horizontal_flip = True,
                             shear_range=0.05, 
                             zoom_range=0.3, 
                             brightness_range=(0.9, 1.1),
                             rescale=1.0 / 255,
                             preprocessing_function=preprocess_input)

test_datagen = ImageDataGenerator(rescale=1.0 / 255, preprocessing_function=preprocess_input)

# Image generator
train_generator = train_datagen.flow_from_directory(
    dataset_path+"/train",
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True
)

valid_generator = test_datagen.flow_from_directory(
    dataset_path+"/valid",
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    dataset_path+"/test",
    target_size=(image_size, image_size),
    batch_size=1,
    class_mode="categorical",
    shuffle=False
)

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.6, patience=2, min_lr=1e-8, verbose=1)
checkpoint = ModelCheckpoint(monitor='val_acc', filepath='/models/RPS_efficientnet_10.h5', verbose=1)
earlystopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

epochs = 10

steps_per_epoch = train_generator.n // batch_size
valid_steps = valid_generator.n // batch_size


model = create_model()

for layer in model.layers:
    layer.trainable = True

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())

# Training
history = model.fit_generator(train_generator, 
                              steps_per_epoch = steps_per_epoch,
                              epochs=epochs, 
                              workers=workers,
                              validation_data=valid_generator, 
                              validation_steps=valid_steps,
                              callbacks =[reduce_lr,earlystopping,checkpoint])

#View results in progress
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,max(plt.ylim())])
plt.title('Training and Validation Loss')

plt.savefig('training_graph.jpg')

# Check test set
test_preds = model.predict_generator(test_generator,steps=len(test_generator),verbose=1)
test_res = [t.argmax() for t in test_preds]

print(f1_score(test_generator.labels,test_res))