import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.backends.cudnn as cudnn

import itertools
import random
import math
import os
from tqdm import tqdm
from load import loadPrepareData
from load import SOS_token, EOS_token, PAD_token
from model import EncoderRNN, LuongAttnDecoderRNN
from config import MAX_LENGTH, teacher_forcing_ratio, save_dir

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

cudnn.benchmark = True
#############################################
# generate file name for saving parameters
#############################################
def filename(reverse, obj):
	filename = ''
	if reverse:
		filename += 'reverse_'
	filename += obj
	return filename


#############################################
# Prepare Training Data
#############################################
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

# batch_first: true -> false, i.e. shape: seq_len * batch
def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# convert to index, add EOS
# return input pack_padded_sequence
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = [len(indexes) for indexes in indexes_batch]
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# convert to index, add EOS, zero padding
# return output variable, mask, max length of the sentences in batch
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# pair_batch is a list of (input, output) with length batch_size
# sort list of (input, output) pairs by input length, reverse input
# return input, lengths for pack_padded_sequence, output_variable, mask
def batch2TrainData(voc, pair_batch, reverse):
    if reverse:
        pair_batch = [pair[::-1] for pair in pair_batch]
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len

#############################################
# Training
#############################################

def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, max_length=MAX_LENGTH):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    loss = 0
    print_losses = []
    n_totals = 0

    encoder_outputs, encoder_hidden = encoder(input_variable, lengths, None)

    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    decoder_hidden = encoder_hidden[:decoder.n_layers]

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Run through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_input = target_variable[t].view(1, -1) # Next input is current target
            loss += F.cross_entropy(decoder_output, target_variable[t], ignore_index=EOS_token)
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            _, topi = decoder_output.topk(1) # [64, 1]

            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            loss += F.cross_entropy(decoder_output, target_variable[t], ignore_index=EOS_token)

    loss.backward()

    clip = 50.0
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / max_target_len 


def trainIters(corpus, reverse, n_iteration, learning_rate, batch_size, n_layers, hidden_size,
                print_every, save_every, dropout, loadFilename=None, attn_model='dot', decoder_learning_ratio=5.0):
    # 定义一个名为 `trainIters` 的函数，它接受很多参数，主要用于训练一个神经网络模型。

    voc, pairs = loadPrepareData(corpus)
    # 调用 `loadPrepareData` 函数，用于加载并预处理给定语料库中的数据，并返回词汇表（vocabulary）对象和句子对。

    # training data
    corpus_name = os.path.split(corpus)[-1].split('.')[0]
    # 通过操作系统路径的处理方法获取语料库的名称。

    training_batches = None
    # 初始化 `training_batches` 为 None。`training_batches` 将用于保存训练数据。

    try:
        # 在 `try` 语句块中，试图加载已经保存的训练数据。
        training_batches = torch.load(os.path.join(save_dir, 'training_data', corpus_name,
                                                   '{}_{}_{}.tar'.format(n_iteration, \
                                                                         filename(reverse, 'training_batches'), \
                                                                         batch_size)))
    except FileNotFoundError:
        # 如果在上述路径未找到训练数据文件，则执行 `except` 块中的代码。

        print('Training pairs not found, generating ...')
        # 输出一个消息，表明正在生成训练对。

        training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)], reverse)
                          for _ in range(n_iteration)]
        # 使用 `batch2TrainData` 函数生成新的训练数据。这个过程中，会从所有句子对中随机选择出指定数量的句子对，并按照指定的批量大小生成训练数据。

        torch.save(training_batches, os.path.join(save_dir, 'training_data', corpus_name,
                                                  '{}_{}_{}.tar'.format(n_iteration, \
                                                                        filename(reverse, 'training_batches'), \
                                                                        batch_size)))
        # 将生成的训练数据保存在指定的路径。

    # model
    # model
    checkpoint = None  # 初始化一个变量 `checkpoint`，它可能会用来存储模型的状态。
    print('Building encoder and decoder ...')  # 输出一个消息，表明正在构建编码器和解码器。
    embedding = nn.Embedding(voc.n_words, hidden_size)  # 创建一个嵌入层，其输入维度为词汇表的大小，输出维度为隐藏层的大小。
    encoder = EncoderRNN(voc.n_words, hidden_size, embedding, n_layers, dropout)  # 使用`EncoderRNN`类构建一个编码器。
    attn_model = 'dot'  # 设置注意力模型的类型为 'dot'。
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.n_words, n_layers, dropout)  # 使用`LuongAttnDecoderRNN`类构建一个解码器。
	
    if loadFilename:  # 如果提供了 `loadFilename`，则从该路径加载模型的状态，并将状态加载到编码器和解码器中。
	checkpoint = torch.load(loadFilename)
	encoder.load_state_dict(checkpoint['en'])
	decoder.load_state_dict(checkpoint['de'])
	
    # use cuda
    encoder = encoder.to(device)  # 将编码器放置到指定的设备上（CPU或者GPU）。
    decoder = decoder.to(device)  # 将解码器放置到指定的设备上（CPU或者GPU）。
	
    # optimizer
    print('Building optimizers ...')  # 输出一个消息，表明正在构建优化器。
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)  # 对编码器创建一个Adam优化器。
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)  # 对解码器创建一个Adam优化器。
	
    if loadFilename:  # 如果提供了 `loadFilename`，则从checkpoint中加载优化器的状态，并将状态加载到编码器和解码器的优化器中。
	encoder_optimizer.load_state_dict(checkpoint['en_opt'])
	decoder_optimizer.load_state_dict(checkpoint['de_opt'])
	
    # initialize
    print('Initializing ...')  # 输出一个消息，表明正在进行初始化操作。
    start_iteration = 1  # 初始化开始的迭代次数为1。
    perplexity = []  # 初始化困惑度为一个空列表。
    print_loss = 0  # 初始化打印的损失为0。
	
    if loadFilename:  # 如果提供了 `loadFilename`，则从checkpoint中加载开始的迭代次数和困惑度。
	start_iteration = checkpoint['iteration'] + 1
	perplexity = checkpoint['plt']

    for iteration in tqdm(range(start_iteration, n_iteration + 1)):  # 使用tqdm库创建一个进度条，并按照设定的迭代次数进行迭代。
        training_batch = training_batches[iteration - 1]  # 从训练批次列表中获取当前批次的数据。
        input_variable, lengths, target_variable, mask, max_target_len = training_batch  # 从当前批次的数据中获取输入变量、长度、目标变量、掩码和最大目标长度。

        # 调用train函数进行训练，并得到训练的损失。
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                    decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size)
        print_loss += loss  # 将损失加入到打印的损失中。
        perplexity.append(loss)  # 将损失加入到困惑度列表中。

        # 如果当前迭代次数可以被设置的打印频率整除，那么就计算平均损失并打印出来。
        if iteration % print_every == 0:
            print_loss_avg = math.exp(print_loss / print_every)
            print('%d %d%% %.4f' % (iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0  # 打印完平均损失后，重置打印的损失。

        # 如果当前迭代次数可以被设置的保存频率整除，那么就保存模型和优化器的状态，以及损失和困惑度。
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, 'model', corpus_name, '{}-{}_{}'.format(n_layers, n_layers, hidden_size))  # 定义保存目录的路径。
            if not os.path.exists(directory):  # 如果保存目录不存在，就创建它。
                os.makedirs(directory)
            torch.save({  # 使用torch.save函数保存模型和优化器的状态，以及损失和困惑度。
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'plt': perplexity
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, filename(reverse, 'backup_bidir_model'))))
