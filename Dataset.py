import torch
from transformers import AutoTokenizer
#加载分词器
tokenizer = AutoTokenizer.from_pretrained('hfl/rbt3')
# print(tokenizer)
from datasets import load_dataset, load_from_disk

# 分词测试
# out=tokenizer.batch_encode_plus(
#     [[
#         '习', '近', '平', '说', '中', '国', '马', '上', '会', '繁', '容', '富', '强', '。'
#     ],
#      [
#          '这', '座', '大', '山', '非', '常', '美', '丽', '他', '是', '由', '我', '国', '著', '名','设', '计', '师', '打', '造', '的','。'
#      ]],
#     truncation=True,
#     padding=True,
#     return_tensors='pt',               #返回tensor
#     is_split_into_words=True)
# print(tokenizer.decode(out['input_ids'][0]))
# print(tokenizer.decode(out['input_ids'][1]))
# for k,v in out.items():
#     print(k,v)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        dataset = load_from_disk(dataset_path='./data/peoples_daily_ner')[split]
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        tokens = self.dataset[i]['tokens']
        labels = self.dataset[i]['ner_tags']
        return tokens, labels

dataset = Dataset('train')
# tokens, labels = dataset[9]
# print(tokens)
# print(labels)
# print(len(dataset))
#数据整理函数
def collate_fn(data):
    tokens = [i[0] for i in data]
    labels = [i[1] for i in data]
    inputs = tokenizer.batch_encode_plus(tokens, truncation=True, padding=True, return_tensors='pt', max_length=512, is_split_into_words=True)
    #最长句子的长度
    lens = inputs['input_ids'].shape[1]
    for i in range(len(labels)):
        labels[i] = [7] + labels[i]
        labels[i] += [7] * lens
        labels[i] = labels[i][:lens]
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    labels = torch.LongTensor(labels).to(device)
    return inputs, labels
#数据加载器
loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, collate_fn=collate_fn, shuffle=True, drop_last=True)
# print(len(loader))

from transformers import AutoModel
pretrained = AutoModel.from_pretrained('hfl/rbt3')
pretrained.to(device)
# print(sum(i.numel() for i in pretrained.parameters()) / 10000)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tuning = False
        self.pretrained = None
        #网络层
        self.rnn = torch.nn.GRU(input_size=768, hidden_size=768, batch_first=True)
        self.fc = torch.nn.Linear(in_features=768, out_features=8)

    def forward(self, inputs):
        if self.tuning:
            out = self.pretrained(**inputs).last_hidden_state
        else:
            with torch.no_grad():
                out = pretrained(**inputs).last_hidden_state
        out, _ = self.rnn(out)
        out = self.fc(out).softmax(dim=2)
        return out

    def fine_tuning(self, tuning):
        self.tuning = tuning
        if tuning:
            for i in pretrained.parameters():
                i.requires_grad = True
            pretrained.train()
            self.pretrained = pretrained
        else:
            for i in pretrained.parameters():
                i.requires_grad_(False)
            pretrained.eval()
            self.pretrained = None

model = Model()
model.to(device)

def reshape_and_remove_pad(outs, labels, attention_mask):
    # 变形,便于计算loss
    # [b, lens, 8] -> [b*lens, 8]
    outs = outs.reshape(-1, 8)
    # [b, lens] -> [b*lens]
    labels = labels.reshape(-1)

    # 忽略对pad的计算结果
    # [b, lens] -> [b*lens - pad]
    select = attention_mask.reshape(-1) == 1
    outs = outs[select]
    labels = labels[select]
    return outs, labels

def get_correct_and_total_count(labels, outs):
    # [b*lens, 8] -> [b*lens]
    outs = outs.argmax(dim=1)
    correct = (outs == labels).sum().item()
    total = len(labels)
    select = labels != 0
    outs = outs[select]
    labels = labels[select]
    correct_content = (outs == labels).sum().item()
    total_content = len(labels)
    return correct, total, correct_content, total_content

from transformers import AdamW
from transformers.optimization import get_scheduler


def train(epochs):
    lr = 2e-5 if model.tuning else 5e-4
    optimizer = AdamW(model.parameters(), lr=lr,no_deprecation_warning=True)
    criterion = torch.nn.CrossEntropyLoss()
    #定义学习率调度器
    scheduler = get_scheduler(name='linear', num_warmup_steps=0, num_training_steps=len(loader) * epochs, optimizer=optimizer)
    model.train()
    for epoch in range(epochs):
        for step, (inputs, labels) in enumerate(loader):
            outs = model(inputs)
            outs, labels = reshape_and_remove_pad(outs, labels, inputs['attention_mask'])
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if step % (len(loader) * epochs // 30) == 0:
                counts = get_correct_and_total_count(labels, outs)
                accuracy = counts[0] / counts[1]
                accuracy_content = counts[2] / counts[3]
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                print(epoch, step, loss.item(), lr, accuracy, accuracy_content)
    torch.save(model, 'model/Chinese.model')

# model.fine_tuning(False)
# print(sum(p.numel() for p in model.parameters()) / 10000)
# train(1)

# model.fine_tuning(True)
# print(sum(p.numel() for p in model.parameters())/10000)
# train(5)

def test():
    model_load = torch.load('model/Chinese.model')
    model_load.eval()
    model_load.to(device)
    loader_test = torch.utils.data.DataLoader(dataset=Dataset('validation'), batch_size=128, collate_fn=collate_fn, shuffle=True, drop_last=True)
    correct = 0
    total = 0
    correct_content = 0
    total_content = 0
    for step, (inputs, labels) in enumerate(loader_test):
        if step == 5:
            break
        print(step)
        with torch.no_grad():
            outs = model_load(inputs)
        outs, labels = reshape_and_remove_pad(outs, labels, inputs['attention_mask'])
        counts = get_correct_and_total_count(labels, outs)
        correct += counts[0]
        total += counts[1]
        correct_content += counts[2]
        total_content += counts[3]
    print(correct/total, correct_content/ total_content)
# test()


def predict():
    model_load=torch.load('model/Chinese.model')
    model_load.eval()
    model_load.to(device)
    loader_test=torch.utils.data.DataLoader(dataset=Dataset('validation'),
                                            batch_size=32,
                                            collate_fn=collate_fn,
                                            shuffle=True,
                                            drop_last=True)
    for i,(inputs,labels) in enumerate(loader_test):
        break
    with torch.no_grad():
        outs=model_load(inputs).argmax(dim=2)
    for i in range(32):
        select=inputs['attention_mask'][i]==1
        input_id=inputs['input_ids'][i,select]
        out=outs[i,select]
        label=labels[i,select]
        print(tokenizer.decode(input_id).replace(' ',''))
        for tag in [label,out]:
            s=''
            for j in range(len(tag)):
                if tag[j]==0:
                    s+='.'
                    continue
                s+=tokenizer.decode(input_id[j])
                s+=str(tag[j].item())
            print(s)
        print('=======================================')
# predict()


def predict_sentence(sentence):
    model_load = torch.load('model/Chinese.model')
    model_load.eval()
    model_load.to(device)

    # 预处理
    encoded_input = tokenizer.encode_plus(sentence, truncation=True, padding=True, return_tensors='pt', max_length=512, is_split_into_words=False)

    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model_load({'input_ids': input_ids, 'attention_mask': attention_mask}).argmax(dim=2)

    decoded_sentence = tokenizer.decode(input_ids.squeeze())
    select = attention_mask.squeeze() == 1
    input_id = input_ids.squeeze()[select]
    output = outputs.squeeze()[select]
    s = ''
    for j in range(len(output)):
        if output[j] == 0:
            s += '.'
            continue
        s += tokenizer.decode(input_id[j])
        s += str(output[j].item())
    # print(decoded_sentence.replace(' ', ''))
    print(s)


# predict_sentence('学习邓小平理论，最根本的是要认真学习邓小平同志运用马克思主义的立场、观点和方法，研究新情况、解决新问题的科学态度和创造精神。')







