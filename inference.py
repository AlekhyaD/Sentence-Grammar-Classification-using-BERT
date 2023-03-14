

import numpy as np
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
import transformers
from transformers import AutoModelForSequenceClassification
from sys import argv



if torch.cuda.is_available():    
      device = torch.device("cuda")
else:
    device = torch.device("cpu")

tokenizer = BertTokenizer.from_pretrained('./model_save/')

model = AutoModelForSequenceClassification.from_pretrained("./model_save/")
model.to(device)

#sentences = '"Agreed, nothing has, but big a opportunity this for any player is."'
if argv[1]=='string':
  sentences = [argv[2]]
elif argv[1]=='file':
  my_file = open(argv[2], "r")
    
  # reading the file
  data = my_file.read()
    
  # replacing end splitting the text 
  # when newline ('\n') is seen.
  sentences = data.split("\n")
  # print(data_into_list)
  my_file.close()
  print(sentences)

input_ids = []
for sent in sentences:
  encoded_sent = tokenizer.encode(sent, add_special_tokens = True)
      
  input_ids.append(encoded_sent)

# Pad our input tokens
input_ids = pad_sequences(input_ids, maxlen=64, 
                          dtype="long", truncating="post", padding="post")

attention_masks = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask) 
print(input_ids)
# Convert to tensors.
prediction_inputs = torch.tensor(input_ids).to(device)
prediction_masks = torch.tensor(attention_masks).to(device)

model.eval()

# # Tracking variables 
predictions , true_labels = [], []


with torch.no_grad():
    # Forward pass, calculate logit predictions
    outputs = model(prediction_inputs, token_type_ids=None, 
                    attention_mask=prediction_masks)

logits = outputs[0]
print(logits)
# Move logits and labels to CPU
logits = logits.detach().cpu().numpy()
#label_ids = b_labels.to('cpu').numpy()

# Store predictions and true labels
predictions.append(logits)
#print(predictions)
class_labels = ["correct","incorrect"]
predictions = np.argmax(logits, axis=-1)
#print(predictions)
# print the label of the class with maximum score
if len(sentences)==1:
  print(class_labels[predictions[0]])
else:
  prediction_json=[]
  for sent,pred in zip(sentences,predictions):
    temp={}
    temp["sentence"]=sent
    temp["tag"]=class_labels[pred]
    prediction_json.append(temp)
    #print(class_labels[pred])
  print(prediction_json)
