import argparse

import numpy as np
import torch

from Configure import model_configs, training_configs
from DataLoader import load_data, train_valid_split, load_testing_images
from Model import MyModel

parser = argparse.ArgumentParser()
parser.add_argument("mode", help="[train, test, predict]")
parser.add_argument("--data_dir", help="data path", default='../data/')
parser.add_argument("--checkpoint", help=".pth checkpoint file")
parser.add_argument("--save_dir", help="path to save the results")
args = parser.parse_args()

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MyModel(model_configs)
    if args.mode == 'train':
        train, test, original_train = load_data(args.data_dir)

        train, valid = train_valid_split(train, original_train, train_ratio=1)
        model.train(train, training_configs)
        test_accuracy, correct, total = model.evaluate(test)
        print("[Test Results] Model Accuracy %f, Total Test Samples %d" % (test_accuracy, total))

    elif args.mode == 'test':
        _, test, _ = load_data(args.data_dir)
        checkpoint = torch.load(model_configs["saved_models"] + args.checkpoint, map_location=torch.device(device))

        model.network.load_state_dict(checkpoint['net'])
        test_accuracy, correct, total = model.evaluate(test)
        print("[Test Results] Model Accuracy %f, Total Test Samples %d" % (test_accuracy, total))

    elif args.mode == 'predict':
        x_test = load_testing_images(args.data_dir)
        checkpoint = torch.load(model_configs["saved_models"] + args.checkpoint, map_location=torch.device(device))

        model.network.load_state_dict(checkpoint['net'])
        predictions = model.predict_prob(x_test)
        np.save("predictions.npy", predictions)

    else:
        parser.print_help()
