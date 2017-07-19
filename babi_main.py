from babi_loader import BabiDataset, adict, pad_collate
from visualize import make_dot
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

def position_encoding(embedded_sentence):
    '''
    embedded_sentence.size() -> (#batch, #sentence, #token, #embedding)
    l.size() -> (#sentence, #embedding)
    output.size() -> (#batch, #sentence, #embedding)
    '''
    _, slen, _, elen = embedded_sentence.size()

    l = [[(1 - s/slen) - (e/elen) * (1 - 2*s/slen) for e in range(elen)] for s in range(slen)]
    l = torch.FloatTensor(l)
    l = l.unsqueeze(0) # for #batch
    l = l.unsqueeze(2) # for #token
    l = l.expand_as(embedded_sentence)
    weighted = embedded_sentence * Variable(l.cuda())
    return torch.sum(weighted, dim=2).squeeze(2) # sum with tokens

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
        self.init_hidden()
    
    def init_hidden(self):
        '''
        c.size() -> (#hidden, )
        '''
        self.c = Variable(torch.zeros(self.hidden_size)).cuda()

    def forward(self, fact, g):
        '''
        fact.size() -> (#batch, #hidden = #embedding)
        c.size() -> (#hidden, ) -> (#batch, #hidden = #embedding)
        r.size() -> (#batch, #hidden = #embedding)
        h_tilda.size() -> (#batch, #hidden = #embedding)
        g.size() -> (#batch, )
        '''
        c = self.c.unsqueeze(0).expand_as(fact)
        br = self.br.unsqueeze(0).expand_as(fact)
        b = self.br.unsqueeze(0).expand_as(fact)

        r = F.sigmoid(fact @ self.Wr + c @ self.Ur + br)
        h_tilda = F.tanh(fact @ self.W + r * (c @ self.U) + b)
        g = g.unsqueeze(1).expand_as(h_tilda)
        h = g * h_tilda + (1 - g) * c
        return h

class AttentionGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionGRU, self).__init__()
        self.hidden_size = hidden_size
        self.AGRUCell = AttentionGRUCell(input_size, hidden_size)

    def forward(self, facts, G):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        fact.size() -> (#batch, #hidden = #embedding)
        G.size() -> (#batch, #sentence)
        g.size() -> (#batch, )
        C.size() -> (#batch, #hidden)
        '''
        batch_num, sen_num, embedding_size = facts.size()
        for sid in range(sen_num):
            fact = facts[:, sid, :]
            g = G[:, sid]
            C = self.AGRUCell(fact, g)
        return C

class EpisodicMemory(nn.Module):
    def __init__(self, hidden_size):
        super(EpisodicMemory, self).__init__()
        self.AGRU = AttentionGRU(hidden_size, hidden_size)
        self.z1 = nn.Linear(4 * hidden_size, hidden_size)
        self.z2 = nn.Linear(hidden_size, 1)
        self.next_mem = nn.Linear(3 * hidden_size, hidden_size)
        init.xavier_normal(self.z1.state_dict()['weight'])
        init.xavier_normal(self.z2.state_dict()['weight'])
        init.xavier_normal(self.next_mem.state_dict()['weight'])

    def make_interaction(self, facts, questions, prevM):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, #sentence = 1, #embedding)
        prevM.size() -> (#batch, #sentence = 1, #hidden = #embedding)
        z.size() -> (#batch, #sentence, 4 x #embedding)
        G.size() -> (#batch, #sentence)
        '''
        batch_num, sen_num, embedding_size = facts.size()
        questions = questions.expand_as(facts)
        prevM = prevM.expand_as(facts)

        z = torch.cat([
            facts * questions,
            facts * prevM,
            torch.abs(facts - questions),
            torch.abs(facts - prevM)
        ], dim=2)

        z = z.view(-1, 4 * embedding_size)

        G = F.tanh(self.z1(z))
        G = self.z2(G)
        G = G.view(batch_num, -1)
        G = F.softmax(G)
        
        return G

    def forward(self, facts, questions, prevM):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, #sentence = 1, #embedding)
        prevM.size() -> (#batch, #sentence = 1, #hidden = #embedding)
        G.size() -> (#batch, #sentence)
        C.size() -> (#batch, #hidden)
        concat.size() -> (#batch, 3 x #embedding)
        '''
        G = self.make_interaction(facts, questions, prevM)
        C = self.AGRU(facts, G)
        concat = torch.cat([prevM.squeeze(), C, questions.squeeze()], dim=1)
        next_mem = F.relu(self.next_mem(concat))
        next_mem = next_mem.unsqueeze(1)
        return next_mem


class QuestionModule(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(QuestionModule, self).__init__()

    def forward(self, questions, word_embedding):
        '''
        questions.size() -> (#batch, #token)
        word_embedding() -> (#batch, #token, #embedding)
        position_encoding() -> (#batch, #sentence = 1, #embedding)
        '''
        questions = Variable(questions.long().cuda())
        questions = word_embedding(questions)
        questions = questions.unsqueeze(1)
        questions = position_encoding(questions)
        return questions

class InputModule(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(InputModule, self).__init__()
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=True)
        for name, param in self.gru.state_dict().items():
            if 'weight' in name: init.xavier_normal(param)
        self.dropout = nn.Dropout(0.1)

    def forward(self, contexts, word_embedding):
        '''
        contexts.size() -> (#batch, #sentence, #token)
        word_embedding() -> (#batch, #sentence x #token, #embedding)
        position_encoding() -> (#batch, #sentence, #embedding)
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        '''
        contexts = Variable(contexts.long().cuda())
        batch_num, sen_num, token_num = contexts.size()

        contexts = contexts.view(batch_num, -1)
        contexts = word_embedding(contexts)

        contexts = contexts.view(batch_num, sen_num, token_num, -1)
        contexts = position_encoding(contexts)

        facts, hdn = self.gru(contexts)
        facts = facts[:, :, :hidden_size] + facts[:, :, hidden_size:]
        facts = self.dropout(facts)
        return facts

class AnswerModule(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(AnswerModule, self).__init__()
        self.z = nn.Linear(2 * hidden_size, vocab_size)
        init.xavier_normal(self.z.state_dict()['weight'])
        self.dropout = nn.Dropout(0.1)

    def forward(self, prevM, questions):
        prevM = self.dropout(prevM)
        concat = torch.cat([prevM, questions], dim=2).squeeze()
        z = self.z(concat)
        return z

class DMNPlus(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_hop=3, qa=None):
        super(DMNPlus, self).__init__()
        self.num_hop = num_hop
        self.qa = qa
        self.word_embedding = nn.Embedding(vocab_size, hidden_size).cuda()
        init.uniform(self.word_embedding.state_dict()['weight'], a=-(3**0.5), b=3**0.5)

        self.input_module = InputModule(vocab_size, hidden_size)
        self.question_module = QuestionModule(vocab_size, hidden_size)
        # for hop in range(num_hop):
        #     setattr(self, f'memory{hop}', EpisodicMemory(hidden_size))
        self.memory = EpisodicMemory(hidden_size)
        self.answer_module = AnswerModule(vocab_size, hidden_size)

    def forward(self, contexts, questions):
        '''
        contexts.size() -> (#batch, #sentence, #token) -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, #token) -> (#batch, #sentence = 1, #embedding)
        '''
        facts = self.input_module(contexts, self.word_embedding)
        questions = self.question_module(questions, self.word_embedding)
        M = questions
        for hop in range(self.num_hop):
            # episode = getattr(self, f'memory{hop}')
            M = self.memory(facts, questions, M)
        preds = self.answer_module(M, questions)
        return preds

    def interpret_indexed_tensor(self, tensor):
        if len(tensor.size()) == 3:
            # tensor -> n x #sen x #token
            for n, sentences in enumerate(tensor):
                for i, sentence in enumerate(sentences):
                    s = ' '.join([self.qa.IVOCAB[elem] for elem in sentence])
                    print(f'{n}th of batch, {i}th sentence, {s}')
        elif len(tensor.size()) == 2:
            # tensor -> n s #token
            for n, sentence in enumerate(tensor):
                s = ' '.join([self.qa.IVOCAB[elem] for elem in sentence])
                print(f'{n}th of batch, {s}')

def get_loss(preds, targets, model):
    criterion = nn.CrossEntropyLoss()
    targets = Variable(targets.cuda())
    loss = criterion(preds, targets)
    return loss

if __name__ == '__main__':
    for task_id in range(1, 21):
        dset_train = BabiDataset(task_id, is_train=True)
        dset_test = BabiDataset(task_id, is_train=False)
        vocab_size = len(dset_train.QA.VOCAB)
        hidden_size = 80
        
        model = DMNPlus(hidden_size, vocab_size, num_hop=3, qa=dset_train.QA)
        model.cuda()
        optim = torch.optim.Adam(model.parameters(), weight_decay=0.001)

        for epoch in range(256):
            train_loader = DataLoader(
                dset_train, batch_size=100, shuffle=False, collate_fn=pad_collate
            )
            test_loader = DataLoader(
                dset_test, batch_size=len(dset_test), shuffle=False, collate_fn=pad_collate
            )

            early_stopping_cnt = 0
            early_stopping_flag = False
            best_acc = 0
            model.train()

            if not early_stopping_flag:
                for batch_idx, data in enumerate(train_loader):
                    optim.zero_grad()
                    contexts, questions, answers = data

                    preds = model(contexts, questions)
                    loss = get_loss(preds, answers, model)
                    loss.backward()

                    if batch_idx == 0 and epoch == 0:
                        with open('init.txt', 'w', encoding='utf-8') as fp:
                            for name, param in model.state_dict().items():
                                if 'bias' not in name:
                                    fp.write(name + '\n')
                                    fp.write(repr(param))

                    if batch_idx == 50:
                        with open('after.txt', 'w',  encoding='utf-8') as fp:
                            for name, param in model.state_dict().items():
                                if 'bias' not in name:
                                    fp.write(name + '\n')
                                    fp.write(repr(param))

                    print(f'[Task {task_id}] Training... loss : {loss.data[0]}, batch_idx : {batch_idx}, epoch : {epoch}')
                    optim.step()
                    # ww = model.memory.AGRU.AGRUCell.Ur.grad
                    # wr = model.memory.next_mem.weight.grad
                    # print(ww, wr)
                
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

                    print(f'[Task {task_id}] Validation Accuracy : {acc}, epoch : {epoch}')
            else:
                print(f'[Task {task_id}] Early Stopping at Epoch {best_acc}, Valid Accuracy : {epoch}')