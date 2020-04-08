import numpy as np

word_counts_threold=10
word_counts = {}
nsents = 0
# 遍历语料的列表获取词频
for sent in all_train_captions:
    nsents+=1
    for w in sent.split(" "):
        word_counts[w] = word_counts.get(w,0)+1

vocab = [w for w in word_counts if word_counts[w]>=word_counts_threold]

print('preprocessed words %d ==> %d' % (len(word_counts), len(vocab)))

idxtoword={}
wordtoidx={}
ix =1

for w in vocab:
    idxtoword[ix] = w
    wordtoidx[w] = ix

    ix+=1
vocab_size = len(idxtoword)+1
print(vocab_size)
embeddings_index = {}

with open(os.path.join(root_path,"glove.6B.200d.txt"),encoding="UTF-8") as f:
    for line in tqdm(f):
        values = line.split()
        word = values[0]

        coefs = np.asarray(values[1:],dtype="float32")

        embeddings_index[word] = coefs

print(f'Found {len(embeddings_index)} word vectors.')

emb_dim =200

embeddings_matrx = np.zeros((vocab_size,emb_dim))

for i,word in wordtoidx.items():
    emb_vecctor = embeddings_index.get(word)
    if emb_vecctor is not None:
        embeddings_matrx[i] = emb_vecctor
