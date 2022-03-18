from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration
import json

app = Flask(__name__)

CORS(app, resources={r'/*': {'origins': '*'}})

tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization')
model.load_state_dict(torch.load('.models/pytorch_model.bin'))

@app.route('/', methods=['POST', 'GET'])
def hello_world():
    if request.method == 'POST':
        text = json.loads(request.get_data())['contents']
        return jsonify(predict(text))
    else:
        return jsonify('Invalid Method')

def predict(text):
    summaryList = []

    for i in range(len(text)):
        summaryList.append([])
        for j in range(len(text[i])):
            raw_input_ids = tokenizer.encode(text[i][j])
            input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]

            summary_ids = model.generate(torch.tensor([input_ids]))
            summaryList[i].append(tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True))

    return summaryList


if __name__ == '__main__':
    app.run(debug=True)
