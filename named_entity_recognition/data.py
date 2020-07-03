def batch_generate(data, batch_size, epoch_num=1000):
    batch_num = len(data)//batch_size
    for epoch in range(epoch_num):
        for batch in range(batch_num):
            yield data[batch*batch_size:(batch+1)*batch_size]
        #yield data[(batch+1)*batch_size:]
        #notice we did not use the last batch!
def sents2id(sents, vocab):
    sent_ids = []
    for sent in sents:
        sent_id = []
        for char in sent:
            if char in vocab.keys():
                sent_id.append(vocab[char])
            else:
                sent_id.append(0)
        sent_ids.append(sent_id)
    return sent_ids
def labels2id(sent_labels):
    label2id = {"O": 0,
                "B-PER": 1, "I-PER": 2,
                "B-LOC": 3, "I-LOC": 4,
                "B-ORG": 5, "I-ORG": 6
                }
    sent_ids = []
    for sent_label in sent_labels:
        sent_id = []
        for label in sent_label:
            sent_id.append(label2id[label])
        sent_ids.append(sent_id)
    return sent_ids
def id2label(sent_ids):
    id2label = {0: "O",
            1: "B-PER", 2: "I-PER",
            3: "B-LOC", 4: "I-LOC",
            5: "B-ORG", 6: "I-ORG"
            }
    sent_labels = []
    for sent_id in sent_ids:
        sent_label = []
        for iid in sent_id:
            sent_label.append(id2label[iid])
        sent_labels.append(sent_label)
    return sent_labels
def pad_sequence(seqs, padding_mark = 0):
    seq_len = []
    for i in range(len(seqs)):
        seq_len.append(len(seqs[i]))
    maxlen = max(seq_len)
    for i in range(len(seqs)):
        if maxlen>len(seqs[i]):
            seqs[i].extend([0 for _ in range(maxlen-len(seqs[i]))])
    return (seq_len, seqs)