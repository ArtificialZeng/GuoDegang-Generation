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
