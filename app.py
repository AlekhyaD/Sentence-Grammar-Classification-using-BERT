from flask import Flask, render_template, request
import numpy as np
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from transformers import AutoModelForSequenceClassification
import os


app = Flask(__name__)
model_path = os.path.join(os.getcwd(), 'model_save/')


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

tokenizer = BertTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)


@app.route('/')
def index_view():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        sentences = request.form['freeform']
        sentences = [sentences]
        input_ids = []
        for sent in sentences:
            encoded_sent = tokenizer.encode(sent, add_special_tokens=True)

            input_ids.append(encoded_sent)
        input_ids = pad_sequences(input_ids, maxlen=64,
                                  dtype="long", truncating="post", padding="post")

        attention_masks = []

        # Create a mask of 1s for each token followed by 0s for padding
        for seq in input_ids:
            seq_mask = [float(i > 0) for i in seq]
            attention_masks.append(seq_mask)
        prediction_inputs = torch.tensor(input_ids).to(device)
        prediction_masks = torch.tensor(attention_masks).to(device)
        model.eval()

        predictions, true_labels = [], []
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(prediction_inputs, token_type_ids=None,
                            attention_mask=prediction_masks)

        logits = outputs[0]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()

        # Store predictions and true labels
        predictions.append(logits)
        class_labels = ["correct", "incorrect"]
        predictions = np.argmax(logits, axis=-1)

        return render_template('predict.html', result=class_labels[predictions[0]])
    else:
        return "Unable to read the file. Please check file extension"


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8000)
