from argparse import ArgumentParser

from config import Config
from interactive_predict import InteractivePredictor
from model import Model
import os
import warnings
warnings.filterwarnings("ignore",category=FutureWarning)
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging
logging.getLogger('tensorflow').disabled = True


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
    parser.add_argument('--adv_eval', action='store_true')
    parser.add_argument('--debug', action='store_true')
    ###Additional Args for adv-train####
    parser.add_argument("-bs", dest="batch_size", type=int, help="size of batch in training", required=False)
    parser.add_argument("-t", dest="transformations", type=int, help="number of transformations in the dataset", required=False)
    parser.add_argument("-td", dest="train_dir", help="directory for adv-training", required=False)
    # for gradient attack
    parser.add_argument('--num_replace_tokens', default=20, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lamb', default=0.5, type=float)
    args = parser.parse_args()

    replace_tokens = ["@R_%d@"%i for i in range(1, args.num_replace_tokens+1)]

    if args.debug:
        config = Config.get_debug_config(args)
    else:
        config = Config.get_default_config(args)

    # Composite training loss
    lamb = args.lamb
    print('Lamb :=' + str(lamb))

    model = Model(config, replace_tokens)
    print('Created model')

    if config.TRAIN_PATH:
        model.train(lamb=lamb)
    if config.TEST_PATH and not args.data_path:
        results, precision, recall, f1 = model.evaluate()
        print('Accuracy: ' + str(results))
        print('Precision: ' + str(precision) + ', recall: ' + str(recall) + ', F1: ' + str(f1))
    if args.predict:
        predictor = InteractivePredictor(config, model)
        predictor.predict()
    if args.adv_eval:
        model.adv_eval_batched()    
    if args.release and args.load_path:
        model.evaluate(release=True)
    model.close_session()
