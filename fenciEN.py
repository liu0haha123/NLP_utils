import re

def filter_punc(sentence):
    # 将文本中的标点符号过滤掉
    sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\'“”《》?“]+|[+——！，。？、~@#￥%……&*（）：]+", "", sentence)
    
    return(sentence)

def filt_noneENG(sentence):
    # 过滤非英文字符
    processed_text = re.sub(r'[^\x00-\x7f]',r'', sentence)

    return processed_text

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