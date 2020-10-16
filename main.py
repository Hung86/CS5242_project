import argparse
import logging
import sys
import os
from glob import glob
from datetime import datetime
import pandas as pd


'''
GROUP 11
TEAM MEMBERS: 
 - A0225551L THU YA KYAW
 - 
'''


def get_cmd_arguments():
    parser = argparse.ArgumentParser(description='This script produces an image classifier using provided train images'
                                                 'and classify test images using the trained classifier')
    parser.add_argument('--data_dir', type=str, required=True,
                        help="folder path to locate the train images, their respective labels and the test images")
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


def get_dir(data_dir, dir_name):
    dir_1 = os.path.join(data_dir, dir_name)

    if os.path.exists(dir_1):  # check target directory existence in the system
        img_count = len(glob("{}/*.png".format(dir_1)))

        if img_count == 0:  # if no image is found, increase 1 more layer
            dir_2 = os.path.join(dir_1, dir_name)
            img_count = len(glob("{}/*.png".format(dir_2)))

            if img_count == 0:  # if no image is found here also, raise error
                raise FileNotFoundError("!!! No images found at both {} and {} !!!".format(dir_1, dir_2))
            else:
                return dir_2  # if images are found here, return this directory path instead

        else:
            return dir_1  # if images are found here, return this directory path

    else:  # raise error if data directory doesn't exist
        raise OSError("!!! {} folder doesn't exist !!!".format(dir_1))


def run(args, logger):
    logger.info("{} has started".format(__file__))
    data_dir = args.data_dir

    # get respective directories
    train_dir = get_dir(data_dir, 'train_image')
    test_dir = get_dir(data_dir, 'test_image')
    label_file = os.path.join(data_dir, 'train_label.csv')

    df = pd.read_csv(label_file)
    df_0 = df[df['Label'] == 0]
    df_1 = df[df['Label'] == 1]
    df_2 = df[df['Label'] == 2]

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
