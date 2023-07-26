import os
import re
from tqdm import tqdm
import thulac  # 导入thulac库，thulac是清华大学开发的一个用于中文文本分词和词性标注的工具。

fout = open('crosstalk.txt', 'wb')  # 创建一个以二进制写模式('wb')打开的文件对象，用于写入处理后的内容。
thu1 = thulac.thulac()  # 创建一个 thulac 对象，用于后续的文本分词操作。

for fname in os.listdir('./data'):  # 遍历当前目录下 data 子目录内的所有文件。
    fin = open('./data/' + fname, 'rb')  # 以二进制读模式('rb')打开一个文件，该文件的名字由 data 子目录和当前遍历到的文件名构成。
    lst_line = None  # 初始化 lst_line 变量，用于存储前一行的数据。
    for line in fin:  # 遍历当前打开的文件中的每一行。
        line = line.decode('gbk')  # 将二进制格式的行内容解码为 'gbk' 编码的字符串。
        line = line[line.find('：') + 1:].strip()  # 去掉每一行中 '：' 符号前的所有字符，然后使用 `strip()` 方法去除字符串两侧的空格。
        line = ' '.join([x[0] for x in thu1.cut(line, text=False)]) + '\n'  # 使用 thulac 对每一行进行分词，然后将得到的分词结果用空格连接起来。
        # pattern = re.compile('.')
        # line = ' '.join(pattern.findall(line)) + '\n'
        if lst_line is not None:  # 检查 lst_line 是否为 None，如果不是，说明已经处理过至少一行数据。
            fout.write(lst_line.encode('utf-8'))  # 将 lst_line 和当前行的内容以 'utf-8' 编码格式写入到输出文件。
            fout.write(line.encode('utf-8'))  # 将 lst_line 和当前行的内容以 'utf-8' 编码格式写入到输出文件。
            lst_line = None  # 重置 lst_line 变量为 None，准备处理下一对行。
        else:
            lst_line = line  # 如果 lst_line 是 None，说明这是正在处理的第一行，或者上一对行已经写入输出文件。将当前行的内容保存到 lst_line，准备和下一行一起写入输出文件。
'''
这段代码主要用于处理存储在 './data' 文件夹下的文本文件，它执行以下步骤：

首先，该代码使用 thulac，一个中文文本分词和词性标注工具，初始化了一个 thulac 对象。

它遍历 './data' 文件夹中的所有文件。对于每个文件，它会逐行读取内容。

对于读取到的每一行，它先使用 'gbk' 编码进行解码，然后移除了字符串中 "：" 之前的所有字符，同时去除了字符串两端的空白字符。

接着，它使用 thulac 对字符串进行分词，并将分词后的结果用空格连接起来。

这段代码接着检查前一行（如果存在的话）是否已经写入输出文件，如果是，则重置 lst_line 变量为 None，否则，将当前行存储到 lst_line。

当 lst_line 不为 None 时，它会将 lst_line 和当前行的内容编码为 'utf-8'，并写入输出文件 'crosstalk.txt'。

综上所述，这段代码主要用于处理 './data' 文件夹下的文件，使用 thulac 对文本进行分词，并将处理后的结果写入输出文件 'crosstalk.txt'。注意，它每次都是处理一对行，然后将这一对行写入到输出文件中。'''
