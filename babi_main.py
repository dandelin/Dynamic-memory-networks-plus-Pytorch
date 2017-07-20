from babi_loader import BabiDataset, pad_collate
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.data import DataLoader

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
        self.br = nn.Parameter(init.constant(torch.Tensor(hidden_size), 1))
        self.W = nn.Parameter(init.xavier_normal(torch.Tensor(input_size, hidden_size)))
        self.U = nn.Parameter(init.xavier_normal(torch.Tensor(hidden_size, hidden_size)))
        self.b = nn.Parameter(init.constant(torch.Tensor(hidden_size), 1))

    def forward(self, fact, hidden, g):
        '''
        fact.size() -> (#batch, #hidden = #embedding)
        c.size() -> (#hidden, ) -> (#batch, #hidden = #embedding)
        r.size() -> (#batch, #hidden = #embedding)
        h_tilda.size() -> (#batch, #hidden = #embedding)
        g.size() -> (#batch, )
        '''
        c = hidden
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
        C = Variable(torch.zeros(self.hidden_size)).cuda()
        for sid in range(sen_num):
            fact = facts[:, sid, :]
            g = G[:, sid]
            if sid == 0:
                C = C.unsqueeze(0).expand_as(fact)
            C = self.AGRUCell(fact, C, g)
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
        questions.size() -> (#batch, 1, #hidden)
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
        questions.size() -> (#batch, #sentence = 1, #hidden)
        prevM.size() -> (#batch, #sentence = 1, #hidden = #embedding)
        G.size() -> (#batch, #sentence)
        C.size() -> (#batch, #hidden)
        concat.size() -> (#batch, 3 x #embedding)
        '''
        G = self.make_interaction(facts, questions, prevM)
        C = self.AGRU(facts, G)
        concat = torch.cat([prevM.squeeze(1), C, questions.squeeze(1)], dim=1)
        next_mem = F.relu(self.next_mem(concat))
        next_mem = next_mem.unsqueeze(1)
        return next_mem


class QuestionModule(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(QuestionModule, self).__init__()
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, questions, word_embedding):
        '''
        questions.size() -> (#batch, #token)
        word_embedding() -> (#batch, #token, #embedding)
        gru() -> (1, #batch, #hidden)
        '''
        questions = word_embedding(questions)
        _, questions = self.gru(questions)
        questions = questions.transpose(0, 1)
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
        concat = torch.cat([prevM, questions], dim=2).squeeze(1)
        z = self.z(concat)
        return z

class DMNPlus(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_hop=3, qa=None):
        super(DMNPlus, self).__init__()
        self.num_hop = num_hop
        self.qa = qa
        self.word_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0).cuda()
        init.uniform(self.word_embedding.state_dict()['weight'], a=-(3**0.5), b=3**0.5)
        self.criterion = nn.CrossEntropyLoss()

        self.input_module = InputModule(vocab_size, hidden_size)
        self.question_module = QuestionModule(vocab_size, hidden_size)
        self.memory = EpisodicMemory(hidden_size)
        self.answer_module = AnswerModule(vocab_size, hidden_size)

    def forward(self, contexts, questions):
        '''
        contexts.size() -> (#batch, #sentence, #token) -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, #token) -> (#batch, 1, #hidden)
        '''
        facts = self.input_module(contexts, self.word_embedding)
        questions = self.question_module(questions, self.word_embedding)
        M = questions
        for hop in range(self.num_hop):
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

    def get_loss(self, contexts, questions, targets):
        preds = self.forward(contexts, questions)
        loss = self.criterion(preds, targets)
        reg_loss = 0
        for param in self.parameters():
            reg_loss += 0.001 * torch.sum(param * param)
        acc = self.get_accuracy(preds, targets)
        return loss + reg_loss, acc

    def get_accuracy(self, preds, targets):
        _, pred_ids = torch.max(preds, dim=1)
        corrects = (pred_ids.data == answers.data)
        acc = torch.mean(corrects.float())
        return acc

if __name__ == '__main__':
    for task_id in range(1, 21):
        dset = BabiDataset(task_id)
        vocab_size = len(dset.QA.VOCAB)
        hidden_size = 80

        model = DMNPlus(hidden_size, vocab_size, num_hop=3, qa=dset.QA)
        model.cuda()
        early_stopping_cnt = 0
        early_stopping_flag = False
        best_acc = 0
        optim = torch.optim.Adam(model.parameters())


        for epoch in range(100):
            dset.set_train(True)
            train_loader = DataLoader(
                dset, batch_size=100, shuffle=True, collate_fn=pad_collate
            )

            model.train()

            if not early_stopping_flag:
                total_acc = 0
                cnt = 0
                for batch_idx, data in enumerate(train_loader):
                    optim.zero_grad()
                    contexts, questions, answers = data
                    contexts = Variable(contexts.long().cuda())
                    questions = Variable(questions.long().cuda())
                    answers = Variable(answers.cuda())

                    loss, acc = model.get_loss(contexts, questions, answers)
                    loss.backward()
                    total_acc += acc
                    cnt += 1

                    if batch_idx % 20 == 0:
                        print(f'[Task {task_id}, Epoch {epoch}] [Training] loss : {loss.data[0]: {10}.{8}}, acc : {total_acc / cnt: {5}.{2}}, batch_idx : {batch_idx}')
                    optim.step()

                dset.set_train(False)
                test_loader = DataLoader(
                    dset, batch_size=100, shuffle=False, collate_fn=pad_collate
                )

                model.eval()
                total_acc = 0
                cnt = 0
                for batch_idx, data in enumerate(test_loader):
                    contexts, questions, answers = data
                    contexts = Variable(contexts.long().cuda())
                    questions = Variable(questions.long().cuda())
                    answers = Variable(answers.cuda())

                    _, acc = model.get_loss(contexts, questions, answers)
                    total_acc += acc
                    cnt += 1

                total_acc = total_acc / cnt
                if total_acc > best_acc:
                    best_acc = total_acc
                    early_stopping_cnt = 0
                else:
                    early_stopping_cnt += 1
                    if early_stopping_cnt > 20:
                        early_stopping_flag = True

                print(f'[Task {task_id}, Epoch {epoch}] [Training] loss : {loss.data[0]: {10}.{8}}, acc : {total_acc / cnt: {5}.{2}}, batch_idx : {batch_idx}')
                print(f'[Task {task_id}, Epoch {epoch}] [Validate] Accuracy : {total_acc: {5}.{2}}')
                with open('log.txt', 'a') as fp:
                    fp.write(f'[Task {task_id}, Epoch {epoch}] [Validate] Accuracy : {total_acc: {5}.{2}}' + '\n')
                if total_acc == 1.0:
                    break
            else:
                print(f'[Task {task_id}] Early Stopping at Epoch {epoch}, Valid Accuracy : {best_acc: {5}.{2}}')
