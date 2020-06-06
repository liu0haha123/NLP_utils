import re
import os

#1 过滤 标点

def filter_punc(sentence):
    # 将文本中的标点符号过滤掉
    sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\'“”《》?“]+|[+——！，。？、~@#￥%……&*（）：]+", "", sentence)
    
    return(sentence)

def filt_noneENG(sentence):
    # 过滤非英文字符
    processed_text = re.sub(r'[^\x00-\x7f]',r'', sentence)

    return processed_text

#2 词表方法1 适合单独txt文件
with open(os.path.join(root_path,''),"r") as fp:
    max_length = 0
    for line in fp.read().split("\n"):
        tokens = line.split()
        #分词 根据实际文件需求修改
        if len(line)>=2:
            # 根据标注分离文件名和描述
            id = tokens[0].split(".")[0]
            desc = tokens[1:]

            desc = [word.lower() for word in desc]
            # 去除未知符号
            desc = [w.translate(null_punct) for w in desc]
            # 去除过短词和非英语词
            desc = [word for word in desc if len(word)>1]
            desc = [word for word in desc if word.isalpha()]
            max_length= max(max_length,len(desc))

            if id not in lookup:
                lookup[id] = list()
            lookup[id].append(" ".join(desc))

lex = set()
for key in lookup:
    [lex.update(d.split()) for d in lookup[key]]


# 3 词表方法2 适合可能有复杂标注的办法

def load_corpus():
    data_set = []
    # 具体方法根据涉及情况写
    # 根据训练方式读取语料和标签，这一步可以不用分词
    data_set.append((A,B))# data,label的二元组
    return data_set

corpus = load_corpus()


def corpus_preprocess(corpus):
    data_set = []
    for XX,YY in corpus:
        #这里有一个小trick是把所有的句子转换为小写，从而减小词表的大小
        #一般来说这样的做法有助于效果提升
        # 分词
        XX = XX.strip().lower()
        XX = XX.split(" ")
        
        data_set.append((XX,YY))

    return data_set
# 输入分好词的语料对
def build_dict(corpus):
    word_freq_dict = dict()
    for sentence, _ in corpus:
        for word in sentence:
            if word not in word_freq_dict:
                word_freq_dict[word] = 0
            word_freq_dict[word] += 1

    word_freq_dict = sorted(word_freq_dict.items(), key = lambda x:x[1], reverse = True)
    
    word2id_dict = dict()
    word2id_freq = dict()

    #一般来说，我们把oov和pad放在词典前面，给他们一个比较小的id，这样比较方便记忆，并且易于后续扩展词表
    word2id_dict['[oov]'] = 0
    word2id_freq[0] = 1e10

    word2id_dict['[pad]'] = 1
    word2id_freq[1] = 1e10

    for word, freq in word_freq_dict:
        word2id_dict[word] = len(word2id_dict)
        word2id_freq[word2id_dict[word]] = freq

    return word2id_freq, word2id_dict


#把语料转换为id序列
def convert_corpus_to_id(corpus, word2id_dict):
    data_set = []
    for sentence, sentence_label in corpus:
        #将句子中的词逐个替换成id，如果句子中的词不在词表内，则替换成oov
        #这里需要注意，一般来说我们可能需要查看一下test-set中，句子oov的比例，
        #如果存在过多oov的情况，那就说明我们的训练数据不足或者切分存在巨大偏差，需要调整
        sentence = [word2id_dict[word] if word in word2id_dict \
                    else word2id_dict['[oov]'] for word in sentence]    
        data_set.append((sentence, sentence_label))
    return data_set
