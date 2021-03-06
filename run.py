import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras.applications import InceptionV3, ResNet50V2, Xception
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.initializers import Zeros, glorot_normal
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # new!

np.random.seed(5242)


'''
GROUP 11
TEAM MEMBERS: 
 - A0225551L Thu Ya Kyaw
 - A0212253W Tran Khanh Hung
 - A0155491B Ian Sim
 - A0206934A Pham Minh Khang
'''


def get_cmd_arguments():
    parser = argparse.ArgumentParser(description='This script produces an image classifier using provided train images'
                                                 'and classify test images using the trained classifier')
    parser.add_argument('train_data', type=str, help="folder path to locate the train images and labels")
    parser.add_argument('test_data', type=str, help="folder path to locate the test images")
    return parser.parse_args()


def setup_logger(logger_name, logger_level=logging.INFO, msg_format=None, datetime_format=None):
    if not msg_format:
        msg_format = '%(asctime)s - %(levelname)s - %(message)s'

    if not datetime_format:
        datetime_format = '%d/%m/%Y %I:%M:%S %p'

    # Initialize logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logger_level)

    if not len(logger.handlers):
        # Add a steam handler to logger
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logger_level)
        stream_handler.setFormatter(logging.Formatter(msg_format, datefmt=datetime_format))
        logger.addHandler(stream_handler)

    return logger


def upsampling(data_frame, logger):
    l0 = data_frame[data_frame.Label == '0']
    l1 = data_frame[data_frame.Label == '1']
    l2 = data_frame[data_frame.Label == '2']

    n_l0 = l0.shape[0]
    n_l1 = l1.shape[0]
    n_l2 = l2.shape[0]
    max_v = np.max([n_l0, n_l1, n_l2])
    logger.info("Before upsampling: ")
    logger.info("Class 0 count: {}".format(n_l0))
    logger.info("Class 1 count: {}".format(n_l1))
    logger.info("Class 2 count: {}".format(n_l2))
    logger.info("Max count : {}".format(max_v))

    if n_l0 < max_v:
        logger.info("Need to upsample class 0")
        n_rand = max_v - n_l0
        choices = np.random.choice(n_l0, n_rand)

        l0_sample = l0.iloc[choices]
        l0 = l0.append(l0_sample,ignore_index=True)
        l0 = l0.sample(frac=1)

    if n_l1 < max_v:
        logger.info("Need to upsample class 1")
        n_rand = max_v - n_l1
        choices = np.random.choice(n_l1, n_rand)

        l1_sample = l1.iloc[choices]
        l1 = l1.append(l1_sample, ignore_index=True)
        l1 = l1.sample(frac=1)

    if n_l2 < max_v:
        logger.info("Need to upsample class 2")
        n_rand = max_v - n_l2
        choices = np.random.choice(n_l2, n_rand)

        l2_sample = l2.iloc[choices]
        l2 = l2.append(l2_sample, ignore_index=True)
        l2 = l2.sample(frac=1)

    df_upsampled = pd.concat([l0, l1, l2])
    df_upsampled = df_upsampled.sample(frac=1).reset_index(drop=True)

    logger.info("After upsampling:")
    logger.info("Class 0 count : {}".format(df_upsampled[df_upsampled.Label == '0'].shape[0]))
    logger.info("Class 1 count : {}".format(df_upsampled[df_upsampled.Label == '1'].shape[0]))
    logger.info("Class 2 count : {}".format(df_upsampled[df_upsampled.Label == '2'].shape[0]))

    return df_upsampled


def inceptionV3_build_model():
    w_init = glorot_normal()
    b_init = Zeros()
    incep_net3 = InceptionV3(input_shape=(512, 512, 3), include_top=False, weights="imagenet", pooling="avg")

    for layer in incep_net3.layers:
        layer.trainable = False

    model = Sequential()
    model.add(incep_net3)
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(
        Dense(276, activation='relu', kernel_initializer=w_init, bias_initializer=b_init, kernel_regularizer='l2'))
    model.add(BatchNormalization())
    model.add(
        Dense(3, activation='softmax', kernel_initializer=w_init, bias_initializer=b_init, kernel_regularizer='l2'))
    model.summary()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.00001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        amsgrad=False,
        name="Adam",
    )

    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False,
        label_smoothing=0.05,
        reduction="auto",
        name="categorical_crossentropy",
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model


def inceptionV3_train_model(img_size, train_dir, train_label, output_dir, model, logger):
    batch_size = 32

    kf = KFold(n_splits=6, shuffle=True, random_state=42)
    train_gen = ImageDataGenerator(rescale=1. / 255)
    val_gen = ImageDataGenerator(rescale=1. / 255)

    output_logs = os.path.join(output_dir, 'logs')
    tensorboard = TensorBoard(output_logs)

    model_checkpoint = ModelCheckpoint(
                                       filepath=output_dir + "/inceptionV3_model.h5",
                                       monitor='val_loss',
                                       mode='min',
                                       save_best_only=True)
    history = []

    for train_index, val_index in kf.split(train_label):
        train_set = train_label.loc[train_index]
        val_set = train_label.loc[val_index]
        train_set_up = upsampling(train_set, logger)

        train_data = train_gen.flow_from_dataframe(dataframe=train_set_up, directory=train_dir,
                                                   x_col="ID", y_col="Label", shuffle=True, class_mode="categorical",
                                                   seed=123,
                                                   target_size=img_size, batch_size=batch_size)
        val_data = val_gen.flow_from_dataframe(dataframe=val_set, directory=train_dir,
                                               x_col="ID", y_col="Label", shuffle=True, class_mode="categorical",
                                               seed=123,
                                               target_size=img_size, batch_size=batch_size)

        hist = model.fit(train_data, epochs=70, validation_data=val_data, verbose=1, callbacks=[tensorboard, model_checkpoint])
        history.append(hist)
        model.evaluate(val_data, verbose=2)

    return history

def resnet50V2_build_model():
    w_init = glorot_normal()
    b_init = Zeros()
    res_net50v2 = ResNet50V2(input_shape=(512, 512, 3), include_top=False, weights="imagenet", pooling="avg")

    for layer in res_net50v2.layers:
        layer.trainable = False

    model = Sequential()
    model.add(res_net50v2)
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(
        Dense(276, activation='relu', kernel_initializer=w_init, bias_initializer=b_init, kernel_regularizer='l2'))
    model.add(BatchNormalization())
    model.add(
        Dense(3, activation='softmax', kernel_initializer=w_init, bias_initializer=b_init, kernel_regularizer='l2'))
    model.summary()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.00001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        amsgrad=False,
        name="Adam",
    )

    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False,
        label_smoothing=0.05,
        reduction="auto",
        name="categorical_crossentropy",
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model


def resnet50V2_train_model(img_size, train_dir, train_label, output_dir, model, logger):
    batch_size = 32

    kf = KFold(n_splits=6, shuffle=True, random_state=42)
    train_gen = ImageDataGenerator(rescale=1. / 255)
    val_gen = ImageDataGenerator(rescale=1. / 255)

    output_logs = os.path.join(output_dir, 'logs')
    tensorboard = TensorBoard(output_logs)

    model_checkpoint = ModelCheckpoint(
                                       filepath=output_dir + "/resnet50V2_model.h5",
                                       monitor='val_loss',
                                       mode='min',
                                       save_best_only=True)
    history = []

    for train_index, val_index in kf.split(train_label):
        train_set = train_label.loc[train_index]
        val_set = train_label.loc[val_index]
        train_set_up = upsampling(train_set, logger)

        train_data = train_gen.flow_from_dataframe(dataframe=train_set_up, directory=train_dir,
                                                   x_col="ID", y_col="Label", shuffle=True, class_mode="categorical",
                                                   seed=123,
                                                   target_size=img_size, batch_size=batch_size)
        val_data = val_gen.flow_from_dataframe(dataframe=val_set, directory=train_dir,
                                               x_col="ID", y_col="Label", shuffle=True, class_mode="categorical",
                                               seed=123,
                                               target_size=img_size, batch_size=batch_size)

        hist = model.fit(train_data, epochs=50, validation_data=val_data, verbose=1, callbacks=[tensorboard, model_checkpoint])
        history.append(hist)
        model.evaluate(val_data, verbose=2)

    return history


def xception_build_model():
    w_init = glorot_normal()
    b_init = Zeros()
    xception = Xception(input_shape=(512, 512, 3), include_top=False, weights="imagenet", pooling="avg")

    for layer in xception.layers:
        layer.trainable = False

    model = Sequential()
    model.add(xception)
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(
        Dense(512, activation='relu', kernel_initializer=w_init, bias_initializer=b_init, kernel_regularizer='l2'))
    model.add(BatchNormalization())
    model.add(
        Dense(3, activation='softmax', kernel_initializer=w_init, bias_initializer=b_init, kernel_regularizer='l2'))
    model.summary()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.00001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        amsgrad=False,
        name="Adam",
    )

    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False,
        label_smoothing=0.05,
        reduction="auto",
        name="categorical_crossentropy",
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model


def xception_train_model(img_size, train_dir, train_label, output_dir, model, logger):
    batch_size = 32

    kf = KFold(n_splits=6, shuffle=True, random_state=42)
    train_gen = ImageDataGenerator(rescale=1. / 255)
    val_gen = ImageDataGenerator(rescale=1. / 255)

    output_logs = os.path.join(output_dir, 'logs')
    tensorboard = TensorBoard(output_logs)

    model_checkpoint = ModelCheckpoint(
                                       filepath=output_dir + "/xception_model.h5",
                                       monitor='val_loss',
                                       mode='min',
                                       save_best_only=True)
    history = []

    for train_index, val_index in kf.split(train_label):
        train_set = train_label.loc[train_index]
        val_set = train_label.loc[val_index]
        train_set_up = upsampling(train_set, logger)

        train_data = train_gen.flow_from_dataframe(dataframe=train_set_up, directory=train_dir,
                                                   x_col="ID", y_col="Label", shuffle=True, class_mode="categorical",
                                                   seed=123,
                                                   target_size=img_size, batch_size=batch_size)
        val_data = val_gen.flow_from_dataframe(dataframe=val_set, directory=train_dir,
                                               x_col="ID", y_col="Label", shuffle=True, class_mode="categorical",
                                               seed=123,
                                               target_size=img_size, batch_size=batch_size)

        hist = model.fit(train_data, epochs=70, validation_data=val_data, verbose=1, callbacks=[tensorboard, model_checkpoint])
        history.append(hist)
        model.evaluate(val_data, verbose=2)

    return history


def test_model(img_size, test_dir, output_dir, output_filename):
    testgen = ImageDataGenerator(rescale=1. / 255)
    data_test = testgen.flow_from_directory(directory=test_dir,
                                            shuffle=False, target_size=img_size, class_mode='categorical', batch_size=1)

    inceptionV3_model = tf.keras.models.load_model(output_dir + "/inceptionV3_model.h5")
    resnet50V2_model = tf.keras.models.load_model(output_dir + "/resnet50V2_model.h5")
    xception_model = tf.keras.models.load_model(output_dir + "/xception_model.h5")

    models = [inceptionV3_model, resnet50V2_model, xception_model]
    yhats = [model.predict(data_test) for model in models]
    yhats = np.array(yhats)
    predicted_sum = np.sum(yhats, axis = 0)
    predicted_classes = np.argmax(predicted_sum, axis=1)
    
    test_id = data_test.filenames
    new_test_id = [Path(x).stem for x in test_id]
    evaluation = pd.DataFrame({'ID': new_test_id, 'Label': predicted_classes})
    evaluation.to_csv(output_filename, index=False)


def run(args, logger):
    logger.info("{} has started".format(__file__))
    train_data = args.train_data
    test_data = args.test_data

    # get respective directories
    train_dir = os.path.join(train_data, 'train_images')
    test_dir = test_data

    output_dir = 'ckp'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_size = (512, 512)

    train_label_file = os.path.join(train_data, 'train_label.csv')
    train_label = pd.read_csv(train_label_file)
    filename = train_label.ID.map(lambda x: str(x) + ".png").to_numpy()
    label = train_label.Label.map(lambda x: str(x)).to_numpy()
    train_label = train_label.assign(ID=filename, Label=label)


    logger.info("Building model")
    pretrained_inceptionV3 = inceptionV3_build_model()
    pretrained_resnet50V2 = resnet50V2_build_model()
    pretrained_xception = xception_build_model()

    logger.info("Training model")
    inceptionV3_train_model(img_size, train_dir, train_label, output_dir, pretrained_inceptionV3, logger)
    resnet50V2_train_model(img_size, train_dir, train_label, output_dir, pretrained_resnet50V2, logger)
    xception_train_model(img_size, train_dir, train_label, output_dir, pretrained_xception, logger)

    logger.info("Testing model")
    output_filename = 'test_result.csv'
    test_model(img_size, test_dir, output_dir, output_filename)

    logger.info("The output file is stored at {}.".format(output_filename))
    logger.info("{} has finished".format(__file__))


def main():
    # Get command line arguments
    args = get_cmd_arguments()

    # Initialize logger
    logger = setup_logger(logger_name='app')

    # Record start time
    start_time = datetime.now()

    try:
        run(args, logger)

    except Exception as e:
        logger.exception("!!! {} has encountered an error !!!".format(__file__))
        raise e

    finally:
        # Record end time
        end_time = datetime.now()
        logger.info("Total runtime duration: {}".format(end_time - start_time))


if __name__ == '__main__':
    main()
