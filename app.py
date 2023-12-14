from flask import Flask, request
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from Dataset import tokenizer, pretrained
import re

app = Flask(__name__)
device ='cpu'

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tuning = False
        self.pretrained = None
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


# 加载模型


@app.route('/process_text', methods=['POST'])
def process_text():
    try:
        sentence = request.data.decode('utf-8')  # 获取 POST 请求的正文作为文本数据
        # print(sentence)
    except Exception as e:
        return 'Failed to decode text data. Details: ' + str(e), 400
    model_load=torch.load('model/Chinese.model',map_location=torch.device('cpu'))
    model_load.eval()
    model_load.to(device)

    inputs = tokenizer.encode_plus(sentence, truncation=True, padding=True, return_tensors='pt', max_length=512,
                                   is_split_into_words=False)
    # print(inputs)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    # print(model_load)
    with torch.no_grad():
        outputs = model_load({'input_ids': input_ids, 'attention_mask': attention_mask}).argmax(dim=2)
    # print(outputs)
    select = attention_mask.squeeze() == 1
    input_id = input_ids.squeeze()[select]
    output = outputs.squeeze()[select]
    processed_text = ''
    for j in range(len(output)):
        if output[j] == 0:
            processed_text += '.'
            continue
        processed_text += tokenizer.decode(input_id[j])
        processed_text += str(output[j].item())
    pattern = r'([^0-9]*)([1256])'
    result = re.findall(pattern, processed_text)

    # 处理匹配结果
    output = ""
    for group in result:
        output += group[0]
    output = re.sub(r'\.+', ' ', output)
    return output


if __name__ == '__main__':
    app.run()
