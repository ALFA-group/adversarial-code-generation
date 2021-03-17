from argparse import ArgumentParser
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging
logging.getLogger('tensorflow').disabled = True
from config import Config
from interactive_predict import InteractivePredictor
from model import Model

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d", "--data", dest="data_path",
                        help="path to preprocessed dataset", required=False)
    parser.add_argument("-te", "--test", dest="test_path",
                        help="path to test file", metavar="FILE", required=False)

    parser.add_argument("-s", "--save_prefix", dest="save_path_prefix",
                        help="path to save file", metavar="FILE", required=False)
    parser.add_argument("-l", "--load", dest="load_path",
                        help="path to saved file", metavar="FILE", required=False)
    parser.add_argument('--release', action='store_true',
                        help='if specified and loading a trained model, release the loaded model for a smaller model '
                             'size.')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=239)
    parser.add_argument("--batch_size", dest="batch_size", type=int, help="size of batch in training", required=False, default=256)
    parser.add_argument('--num_replace_tokens', default=20, type=int)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    replace_tokens = ["@R_%d@"%i for i in range(1, args.num_replace_tokens+1)]

    if args.debug:
        config = Config.get_debug_config(args)
    else:
        config = Config.get_default_config(args)

    model = Model(config, replace_tokens)
    print('Created model')
    if config.TRAIN_PATH:
        model.train()
    if config.TEST_PATH and not args.data_path:
        results, precision, recall, f1, rouge = model.evaluate()
        print('Accuracy: ' + str(results))
        print('Precision: ' + str(precision) + ', recall: ' + str(recall) + ', F1: ' + str(f1))
        print('Rouge: ', rouge)
    if args.predict:
        predictor = InteractivePredictor(config, model)
        predictor.predict()
    if args.release and args.load_path:
        model.evaluate(release=True)
    model.close_session()
