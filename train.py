import tensorflow as tf
from sklearn.datasets import load_files
import numpy as np
from keras.utils import np_utils
from sklearn.metrics import accuracy_score, f1_score


def load_dataset(dataset_path):
    '''Returns image paths and labels from the given path'''
    data = load_files(dataset_path)
    files = np.array(data['filenames'])
    targets = np_utils.to_categorical(np.array(data['target']), 133)
    return files, targets


def load_bottleneck_features(features_path):
    '''Returns train, valid and test features from the given path'''
    bottleneck_features = np.load(features_path)
    train_Resnet50 = bottleneck_features['train']
    valid_Resnet50 = bottleneck_features['valid']
    test_Resnet50 = bottleneck_features['test']
    return train_Resnet50, valid_Resnet50, test_Resnet50


def define_model(input_shape, output_neurons):
    '''Returns a model defined on the basis of the input shape and output neurons'''
    Resnet50_model = tf.keras.models.Sequential()
    Resnet50_model.add(tf.keras.layers.GlobalAveragePooling2D(input_shape=input_shape))
    Resnet50_model.add(tf.keras.layers.Dense(1024, activation='relu'))
    Resnet50_model.add(tf.keras.layers.Dense(output_neurons, activation='softmax'))

    return Resnet50_model

def train():
    '''train the model on the training data and validate on the validation data'''
    Resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='saved_models/weights_best_Resnet50.hdf5',
                                                      verbose=1, save_best_only=True)

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss')

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=2, monitor='val_loss')
    Resnet50_model.fit(train_Resnet50, train_targets,
                       validation_data=(valid_Resnet50, valid_targets),
                       epochs=50, batch_size=20, callbacks=[checkpointer, early_stopping, reduce_lr],
                       verbose=1)

def validate():
    test_preds = Resnet50_model.predict(test_Resnet50)

    accuracy = accuracy_score(np.argmax(test_targets, axis=1), np.argmax(test_preds, axis=1)) * 100
    f1 = f1_score(np.argmax(test_targets, axis=1), np.argmax(test_preds, axis=1), average='weighted') * 100

    print('Accuracy using Resnet50: {}'.format(round(accuracy, 3)))
    print('F1 Score using Resnet50: {}'.format(round(f1, 3)))

if __name__ == "__main__":
    train_files, train_targets = load_dataset('Data/dogImages/train')
    valid_files, valid_targets = load_dataset('Data/dogImages/valid')
    test_files, test_targets = load_dataset('Data/dogImages/test')

    train_Resnet50, valid_Resnet50, test_Resnet50 = load_bottleneck_features('Data/bottleneck_features/DogResnet50Data.npz')
    print(train_Resnet50.shape)
    Resnet50_model = define_model(train_Resnet50.shape[1:], 133)

    #train()
    #validate()




