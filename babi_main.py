from babi_loader import BabiDataset, adict, pad_collate
import itertools

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image

import numpy as np

# from https://github.com/domluna/memn2n
def position_encoding(sentences):
    batch_size, sentence_size, embedding_size = sentences.size()
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    encoding = np.expand_dims(np.transpose(encoding), 0)
    encoding = Variable(torch.Tensor(encoding).cuda()).expand_as(sentences)
    return torch.sum(sentences * encoding, dim=1)

class AttentionGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionGRUCell, self).__init__()
        self.hidden_size = hidden_size
        
        self.Wr = nn.Parameter(init.xavier_normal(torch.Tensor(input_size, hidden_size)))
        self.Ur = nn.Parameter(init.xavier_normal(torch.Tensor(hidden_size, hidden_size)))
        self.br = nn.Parameter(torch.ones(hidden_size,))
        self.W = nn.Parameter(init.xavier_normal(torch.Tensor(input_size, hidden_size)))
        self.U = nn.Parameter(init.xavier_normal(torch.Tensor(hidden_size, hidden_size)))
        self.b = nn.Parameter(torch.ones(hidden_size,))

    def forward(self, inp, hidden, g):
        batch_num, embedding_size = inp.size()

        hidden = hidden.expand(batch_num, self.hidden_size)
        br = self.br.expand(batch_num, self.hidden_size)
        b = self.b.expand(batch_num, self.hidden_size)

        # print(inp @ self.Wr, hidden @ self.Ur, self.br)
        r = F.sigmoid(inp @ self.Wr + hidden @ self.Ur + br)
        h_tilda = F.tanh(inp @ self.W + r * (hidden @ self.U) + b)
        g = g.contiguous().view(batch_num, 1).expand_as(h_tilda)
        h = g * h_tilda + (1 - g) * hidden
        return h

class AttentionGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionGRU, self).__init__()
        self.hidden_size = hidden_size
        self.AGRU = AttentionGRUCell(input_size, hidden_size)

    def forward(self, facts, G):
        batch_num, sen_num, embedding_size = facts.size()
        c = Variable(torch.zeros(self.hidden_size).cuda())
        for sid in range(sen_num):
            fact = facts[:, sid, :]
            c = self.AGRU(fact, c, G[:, sid])
        return c

class EpisodicMemory(nn.Module):
    def __init__(self, hidden_size):
        super(EpisodicMemory, self).__init__()
        self.AGRU = AttentionGRU(hidden_size, hidden_size)
        self.z1 = nn.Linear(4 * hidden_size, hidden_size)
        self.z2 = nn.Linear(hidden_size, 1)
        self.next_mem = nn.Linear(3 * hidden_size, hidden_size)

    def make_interaction(self, facts, questions, prevM):
        # facts = n x #sen x E
        # questions = n x 1 x E
        # M = n x 1 x E
        batch_num, sen_num, embedding_size = facts.size()
        questions = questions.expand_as(facts)
        prevM = prevM.expand_as(facts)
        z = torch.cat([
            facts * questions,
            facts * prevM,
            torch.abs(facts - questions),
            torch.abs(facts - prevM)
        ], dim=2)
        z = z.view(batch_num * sen_num, 4 * embedding_size)
        z = F.tanh(self.z1(z))
        z = self.z2(z)
        z = z.view(batch_num, sen_num)
        return F.softmax(z)

    def forward(self, facts, questions, prevM):
        # G -> n x #sen
        # facts -> n x #sen x E
        G = self.make_interaction(facts, questions, prevM)
        c = self.AGRU(facts, G)
        prevM = torch.squeeze(prevM)
        questions = torch.squeeze(questions)
        z = torch.cat([prevM, c, questions], dim=1)
        next_mem = F.relu(self.next_mem(z))
        next_mem = torch.unsqueeze(next_mem, dim=1)
        return next_mem


class QuestionModule(nn.Module):
    def __init__(self, vocab_size, hidden_size, word_embedding):
        super(QuestionModule, self).__init__()
        self.word_embedding = word_embedding

    def forward(self, questions):
        questions = Variable(questions.long().cuda())
        questions = self.word_embedding(questions)
        questions = position_encoding(questions)
        return questions

class InputModule(nn.Module):
    def __init__(self, vocab_size, hidden_size, word_embedding):
        super(InputModule, self).__init__()
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.word_embedding = word_embedding
        self.dropout = nn.Dropout(0.1)

    def forward(self, contexts):
        # contexts -> n x #sen x #token (LongTensor)
        contexts = Variable(contexts.long().cuda())
        # contexts -> (n x #sen) x #token (for embedding)
        batch_num, sen_num, token_num = contexts.size()
        contexts = contexts.view(batch_num * sen_num, -1)
        # WORD_EMBEDDING -> (n x #sen) x #token x E (FloatTensor)
        contexts = self.word_embedding(contexts)
        # position encoding -> (n x #sen) x E
        contexts = position_encoding(contexts)
        contexts = contexts.view(batch_num, sen_num, -1)
        facts, hdn = self.gru(contexts)
        # forward facts and backward facts
        # element-wise sum : n x #sen x E
        facts = facts[:, :, :hidden_size] + facts[:, :, hidden_size:]
        facts = self.dropout(facts)
        return facts

class AnswerModule(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(AnswerModule, self).__init__()
        self.z = nn.Linear(2 * hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, prevM, questions):
        prevM = self.dropout(prevM)
        concat = torch.cat([prevM, questions], dim=2).squeeze()
        z = self.z(concat)
        return F.log_softmax(z)

class DMNPlus(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_hop=3):
        super(DMNPlus, self).__init__()
        self.num_hop = num_hop
        self.word_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0).cuda()
        init.uniform(self.word_embedding.state_dict()['weight'], a=-(3**0.5), b=3**0.5)
        self.input_module = InputModule(vocab_size, hidden_size, self.word_embedding)
        self.question_module = QuestionModule(vocab_size, hidden_size, self.word_embedding)
        for hop in range(num_hop):
            setattr(self, 'memory{}'.format(hop), EpisodicMemory(hidden_size))
        self.answer_module = AnswerModule(vocab_size, hidden_size)

    def forward(self, contexts, questions):
        facts = self.input_module(contexts)
        questions = self.question_module(questions)
        M = questions
        for hop in range(self.num_hop):
            episode = getattr(self, 'memory{}'.format(hop))
            M = episode(facts, questions, M)
        preds = self.answer_module(M, questions)
        return preds

def get_loss(preds, targets, model):
    criterion = nn.NLLLoss()
    targets = Variable(targets.cuda())
    loss = criterion(preds, targets)

    return loss

if __name__ == '__main__':
    for task_id in range(1, 21):
        dset_train = BabiDataset(task_id, is_train=True)
        dset_test = BabiDataset(task_id, is_train=False)
        vocab_size = len(dset_train.QA.VOCAB)
        print(vocab_size)
        hidden_size = 80
        
        model = DMNPlus(hidden_size, vocab_size, num_hop=3)
        model.cuda()

        for epoch in range(256):
            train_loader = DataLoader(
                dset_train, batch_size=128, shuffle=True, collate_fn=pad_collate
            )
            test_loader = DataLoader(
                dset_test, batch_size=len(dset_test), shuffle=False, collate_fn=pad_collate
            )

            early_stopping_cnt = 0
            early_stopping_flag = False
            best_acc = 0
            optim = torch.optim.Adam(model.parameters())
            model.train()

            if not early_stopping_flag:
                for batch_idx, data in enumerate(train_loader):
                    optim.zero_grad()
                    contexts, questions, answers = data

                    preds = model(contexts, questions)
                    loss = get_loss(preds, answers, model)
                    loss.backward()

                    print('[Task {}] Training... loss : {}, batch_idx : {}, epoch : {}'.format(task_id, loss.data[0], batch_idx, epoch))
                    optim.step()
                model.eval()
                for batch_idx, data in enumerate(test_loader):
                    contexts, questions, answers = data

                    preds = model(contexts, questions)
                    _, pred_ids = torch.max(preds, dim=1)
                    corrects = (pred_ids.data.cpu() == answers)
                    acc = torch.mean(corrects.float())

                    if acc > best_acc:
                        best_acc = acc
                        early_stopping_cnt = 0
                    else:
                        early_stopping_cnt += 1
                        if early_stopping_cnt > 20:
                            early_stopping_flag = True

                    print('[Task {}] Validation Accuracy : {}, epoch : {}'.format(task_id, acc, epoch))
            else:
                print('[Task {}] Early Stopping at Epoch {}, Valid Accuracy : {}').format(task_id, best_acc, epoch)