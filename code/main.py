import argparse
from train import trainIters
from evaluate import runTest


def parse():
    parser = argparse.ArgumentParser(description='Attention Seq2Seq Chatbot')
    parser.add_argument('-tr', '--train', help='Train the model with corpus')
    parser.add_argument('-te', '--test', help='Test the saved model')
    parser.add_argument('-l', '--load', help='Load the model and train')
    parser.add_argument('-c', '--corpus', help='Test the saved model with vocabulary of the corpus')
    parser.add_argument('-r', '--reverse', action='store_true', help='Reverse the input sequence')
    parser.add_argument('-f', '--filter', action='store_true', help='Filter to small training data set')
    parser.add_argument('-i', '--input', action='store_true', help='Test the model by input the sentence')
    parser.add_argument('-it', '--iteration', type=int, default=10000, help='Train the model with it iterations')
    parser.add_argument('-p', '--print', type=int, default=100, help='Print every p iterations')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('-la', '--layer', type=int, default=1, help='Number of layers in encoder and decoder')
    parser.add_argument('-hi', '--hidden', type=int, default=256, help='Hidden size in encoder and decoder')
    parser.add_argument('-be', '--beam', type=int, default=1, help='Hidden size in encoder and decoder')
    parser.add_argument('-s', '--save', type=int, default=500, help='Save every s iterations')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('-d', '--dropout', type=float, default=0.1, help='Dropout probability for rnn and dropout layers')

    args = parser.parse_args()
    return args

def parseFilename(filename, test=False):  # 定义一个函数 parseFilename，这个函数接受两个参数，一个是filename（文件名），另一个是test，test的默认值为False。
    filename = filename.split('/')  # 将输入的文件名按照"/"进行分割，返回一个包含分割后元素的列表。
    dataType = filename[-1][:-4]  # 从列表filename中取最后一个元素的前四个字符（即从开始到倒数第四个字符）并赋值给变量dataType。
    parse = dataType.split('_')  # 将dataType字符串按照 "_" 进行分割，返回一个包含分割后元素的列表。
    reverse = 'reverse' in parse  # 判断'reverse'是否在parse列表中，如果在，则返回True，否则返回False。
    layers, hidden = filename[-2].split('_')  # 从列表filename中取倒数第二个元素，并按 "_" 进行分割，然后将分割后的两个元素分别赋值给变量layers和hidden。
    n_layers = int(layers.split('-')[0])  # 将layers按 "-" 分割，取出第一个元素，并转换为整数，然后赋值给变量n_layers。
    hidden_size = int(hidden)  # 将hidden转换为整数，然后赋值给变量hidden_size。
    return n_layers, hidden_size, reverse  # 返回三个变量：n_layers，hidden_size和reverse。

def run(args):  # 定义一个函数run，这个函数接受一个参数args。
    reverse, fil, n_iteration, print_every, save_every, learning_rate, \
        n_layers, hidden_size, batch_size, beam_size, inp, dropout = \
        args.reverse, args.filter, args.iteration, args.print, args.save, args.learning_rate, \
        args.layer, args.hidden, args.batch_size, args.beam, args.input, args.dropout  # 这一行通过一次性解包args对象的各属性值，将其赋值给对应的变量。
    if args.train and not args.load:  # 如果args对象的train属性为True，并且load属性为False，则执行下一行代码。
        trainIters(args.train, reverse, n_iteration, learning_rate, batch_size,
                    n_layers, hidden_size, print_every, save_every, dropout)  # 调用trainIters函数，传入args.train, reverse, n_iteration, learning_rate, batch_size, n_layers, hidden_size, print_every, save_every, dropout作为参数。
    elif args.load:  # 如果args对象的load属性为True，则执行下一行代码。
        n_layers, hidden_size, reverse = parseFilename(args.load)  # 调用parseFilename函数，传入args.load作为参数，并将返回的三个值分别赋值给n_layers, hidden_size, reverse。
        trainIters(args.train, reverse, n_iteration, learning_rate, batch_size,
                    n_layers, hidden_size, print_every, save_every, dropout, loadFilename=args.load)  # 调用trainIters函数，传入args.train, reverse, n_iteration, learning_rate, batch_size, n_layers, hidden_size, print_every, save_every, dropout, args.load作为参数。
    elif args.test:  # 如果args对象的test属性为True，则执行下一行代码。
        n_layers, hidden_size, reverse = parseFilename(args.test, True)  # 调用parseFilename函数，传入args.test和True作为参数，并将返回的三个值分别赋值给n_layers, hidden_size, reverse。
        runTest(n_layers, hidden_size, reverse, args.test, beam_size, inp, args.corpus)  # 调用runTest函数，传入n_layers, hidden_size, reverse, args.test, beam_size, inp, args.corpus作为参数。



if __name__ == '__main__':
    args = parse()
    run(args)
