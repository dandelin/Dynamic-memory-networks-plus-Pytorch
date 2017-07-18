from glob import glob
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torch.nn.utils.rnn as rnn_utils
import os
import re
import numpy as np
import pickle

class adict(dict):
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self

def pad_collate(batch):
    max_context_sen_len = float('-inf')
    max_context_len = float('-inf')
    max_question_len = float('-inf')
    for elem in batch:
        context, question, _ = elem
        max_context_len = max_context_len if max_context_len > len(context) else len(context)
        max_question_len = max_question_len if max_question_len > len(question) else len(question)
        for sen in context:
            max_context_sen_len = max_context_sen_len if max_context_sen_len > len(sen) else len(sen)
    for i, elem in enumerate(batch):
        _context, question, answer = elem
        context = np.zeros((max_context_len, max_context_sen_len))
        for j, sen in enumerate(_context):
            context[j] = np.pad(sen, (0, max_context_sen_len - len(sen)), 'constant', constant_values=0)
        question = np.pad(question, (0, max_question_len - len(question)), 'constant', constant_values=0)
        batch[i] = (context, question, answer)
    return default_collate(batch)

class BabiDataset(Dataset):
    def __init__(self, task_id, is_train=True):
        self.vocab_path = 'dataset/babi{}_vocab.pkl'.format(task_id)
        self.is_train = is_train
        raw_train, raw_test = get_raw_babi(task_id)
        if os.path.isfile(self.vocab_path):
            self.load()
        else:
            self.QA = adict()
            self.QA.VOCAB = {'<EOS>': 0}
            self.QA.IVOCAB = {0: '<EOS>'}
            self.build_vocab(raw_train + raw_test)
            self.save()
        self.train = self.get_indexed_qa(raw_train)
        self.test = self.get_indexed_qa(raw_test)

    def __len__(self):
        if self.is_train:
            return len(self.train[0])
        else:
            return len(self.test[0])
    
    def __getitem__(self, index):
        contexts, questions, answers = self.train if self.is_train else self.test
        return contexts[index], questions[index], answers[index]

    def save(self):
        os.makedirs(os.path.dirname(self.vocab_path), exist_ok=True)
        with open(self.vocab_path, 'wb') as fp:
            pickle.dump(self.QA, fp, pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open(self.vocab_path, 'rb') as fp:
            self.QA = pickle.load(fp)
    
    def get_indexed_qa(self, raw_babi):
        unindexed = get_unindexed_qa(raw_babi)
        questions = []
        contexts = []
        answers = []
        for qa in unindexed:
            context = [c.lower().split() for c in qa['C']]
            context = [[self.QA.VOCAB[token] for token in sentence] for sentence in context]
            question = qa['Q'].lower().split()
            question = [self.QA.VOCAB[token] for token in question]
            answer = self.QA.VOCAB[qa['A']]
            
            contexts.append(context)
            questions.append(question)
            answers.append(answer)
        return (contexts, questions, answers)
    
    def build_vocab(self, raw_babi):
        lowered = raw_babi.lower()
        tokens = re.findall('[a-z]+|\.', lowered)
        types = set(tokens)
        for t in types:
            if not t in self.QA.VOCAB:
                next_index = len(self.QA.VOCAB)
                self.QA.VOCAB[t] = next_index
                self.QA.IVOCAB[next_index] = t


def get_raw_babi(taskid):
    paths = glob('babi_data/en-10k/qa{}*'.format(taskid))
    for path in paths:
        if 'train' in path:
            with open(path, 'r') as fp:
                train = fp.read()
        elif 'test' in path:
            with open(path, 'r') as fp:
                test = fp.read()
    return train, test

def build_vocab(raw_babi):
    lowered = raw_babi.lower()
    tokens = re.findall('[a-zA-Z]+', lowered)
    types = set(tokens)
    return types

# adapted from https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/
def get_unindexed_qa(raw_babi):
    tasks = []
    task = None
    babi = raw_babi.strip().split('\n')
    for i, line in enumerate(babi):
        id = int(line[0:line.find(' ')])
        if id == 1:
            task = {"C": [], "Q": "", "A": "", "S": ""} 
            counter = 0
            id_map = {}
            
        line = line.strip()
        line = line.replace('.', ' . ')
        line = line[line.find(' ')+1:]
        # if not a question
        if line.find('?') == -1:
            task["C"].append(line)
            id_map[id] = counter
            counter += 1     
        else:
            idx = line.find('?')
            tmp = line[idx+1:].split('\t')
            task["Q"] = line[:idx]
            task["A"] = tmp[1].strip()
            task["S"] = [] # Supporting facts
            for num in tmp[2].split():
                task["S"].append(id_map[int(num.strip())])
            tasks.append(task.copy())
    return tasks

if __name__ == '__main__':
    dset_train = BabiDataset(20, is_train=True)
    train_loader = DataLoader(dset_train, batch_size=2, shuffle=True, collate_fn=pad_collate)
    for batch_idx, data in enumerate(train_loader):
        contexts, questions, answers = data
        break