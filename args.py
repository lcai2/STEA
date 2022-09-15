import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--gpu',
    default = '1',
    type = str,
    help='choose gpu device')
parser.add_argument(
    '--dataset',
    default = 'ICEWS05-15',
    type = str,
    help='choose dataset')
parser.add_argument(
    '--seed',
    default = 1000,
    type = int,
    help='choose the number of align seeds')
parser.add_argument(
    '--dropout',
    default = 0.3,
    type = float,
    help='choose dropout rate')
parser.add_argument(
    '--depth',
    default = 2,
    type = int,
    help='choose number of GNN layers')
parser.add_argument(
    '--gamma',
    default = 3.0,
    type = float,
    help='choose margin')
parser.add_argument(
    '--lr',
    default = 0.005,
    type = float,
    help='choose learning rate')
parser.add_argument(
    '--dim',
    default = 100,
    type = int,
    help='choose embedding dimension')
parser.add_argument(
    '--unsupervised',
    default = False,
    type = bool,
    help='choose unsupervised seeds or golden seeds')
parser.add_argument(
    '--alpha',
    default = 0.3,
    type = float,
    help='choose balance factor for entity similarity and time similarity')
parser.add_argument(
    '--nthread',
    default = 30,
    type = int,
    help='choose number of threads for computing time similarity ')

args = parser.parse_args()