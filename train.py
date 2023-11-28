import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
from functools import partial
import json

from data_store import *

class Dataset_train(Dataset):
    def __init__(self, input_ids, input_masks, output_ids, output_masks):
        self.input_input_ids = input_ids
        self.input_attention_masks = input_masks
        self.output_input_ids = output_ids
        self.output_attention_masks = output_masks

    def __len__(self):
        return len(self.input_input_ids)

    def __getitem__(self, idx):
        x = self.input_input_ids[idx]
        y = self.input_attention_masks[idx]
        z = self.output_input_ids[idx]
        w = self.output_attention_masks[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long), torch.tensor(z, dtype=torch.long), torch.tensor(w, dtype=torch.long)

def read_lines(path):
    # Read the dataset file line by line
    with open(path, 'r') as file:
        lines = file.readlines()
    return lines

def data_parsing(lines):
    lines_sep = []
    for line in lines:
      lines_sep.append(line.split('\t'))
    return lines_sep

def get_train_dataloader(task, subtask, tokenizer):
    lines = read_lines(f'modified_data/{task}/{subtask}/train.txt')
    lines_sep = data_parsing(lines)

    inputs = [x[0].strip() for x in lines_sep]
    outputs = [x[1].strip() for x in lines_sep]

    input_encoded = tokenizer.__call__(inputs, return_tensors = "pt", padding=True)
    output_encoded = tokenizer.__call__(outputs, return_tensors = "pt", padding=True)
    
    train_dataset = Dataset_train(input_encoded["input_ids"], input_encoded["attention_mask"], output_encoded["input_ids"], output_encoded["attention_mask"])
    return DataLoader(train_dataset, batch_size = 32, shuffle = True)

def get_val_dataloader(task, subtask, tokenizer):
    lines = read_lines(f'modified_data/{task}/{subtask}/dev.txt')
    lines_sep = data_parsing(lines)

    inputs = [x[0].strip() for x in lines_sep]
    outputs = [x[1].strip() for x in lines_sep]

    input_encoded = tokenizer.__call__(inputs, return_tensors = "pt", padding=True)
    output_encoded = tokenizer.__call__(outputs, return_tensors = "pt", padding=True)
    
    train_dataset = Dataset_train(input_encoded["input_ids"], input_encoded["attention_mask"], output_encoded["input_ids"], output_encoded["attention_mask"])
    return DataLoader(train_dataset, batch_size = 32, shuffle = True)

def get_test_dataloader(task, subtask, tokenizer):
    lines = read_lines(f'modified_data/{task}/{subtask}/test.txt')
    lines_sep = data_parsing(lines)

    inputs = [x[0].strip() for x in lines_sep]
    outputs = [x[1].strip() for x in lines_sep]

    input_encoded = tokenizer.__call__(inputs, return_tensors = "pt", padding=True)
    output_encoded = tokenizer.__call__(outputs, return_tensors = "pt", padding=True)
    
    train_dataset = Dataset_train(input_encoded["input_ids"], input_encoded["attention_mask"], output_encoded["input_ids"], output_encoded["attention_mask"])
    return DataLoader(train_dataset, batch_size = 16, shuffle = True)

def extract_spans_para(seq, seq_type):
    
    quads = []
    sents = [s.strip() for s in seq.split('[SSEP]')]
    for s in sents:
        try:
            tok_list = ["[C]", "[S]", "[A]", "[O]"]

            for tok in tok_list:
                if tok not in s:
                    s += " {} null".format(tok)
            index_ac = s.index("[C]")
            index_sp = s.index("[S]")
            index_at = s.index("[A]")
            index_ot = s.index("[O]")

            combined_list = [index_ac, index_sp, index_at, index_ot]
            arg_index_list = list(np.argsort(combined_list))

            result = []
            for i in range(len(combined_list)):
                start = combined_list[i] + 4
                sort_index = arg_index_list.index(i)
                if sort_index < 3:
                    next_ = arg_index_list[sort_index + 1]
                    re = s[start:combined_list[next_]]
                else:
                    re = s[start:]
                result.append(re.strip())

            ac, sp, at, ot = result

            # if the aspect term is implicit
            if at.lower() == 'it':
                at = 'null'
        except ValueError:
            try:
                print(f'In {seq_type} seq, cannot decode: {s}')
                pass
            except UnicodeEncodeError:
                print(f'In {seq_type} seq, a string cannot be decoded')
                pass
            ac, at, sp, ot = '', '', '', ''

        quads.append((ac, at, sp, ot))
        

    return quads


def compute_f1_scores(pred_pt, gold_pt, verbose=True):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1

    if verbose:
        print(
            f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}"
        )

    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (
        precision + recall) if precision != 0 or recall != 0 else 0
    scores = {
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100
    }

    return scores


def compute_scores(pred_seqs, gold_seqs, verbose=True):
    """
    Compute model performance
    """
    assert len(pred_seqs) == len(gold_seqs), (len(pred_seqs), len(gold_seqs))
    num_samples = len(gold_seqs)

    all_labels, all_preds = [], []

    for i in range(num_samples):
        gold_list = extract_spans_para(gold_seqs[i], 'gold')
        pred_list = extract_spans_para(pred_seqs[i], 'pred')
        if verbose and i < 10:

            print("gold ", gold_seqs[i])
            print("pred ", pred_seqs[i])
            print()

        all_labels.append(gold_list)
        all_preds.append(pred_list)

    scores = compute_f1_scores(all_preds, all_labels)

    return scores, all_labels, all_preds


def train_function(task, subtask, n_epochs):
    print("task {} subtask {}".format(task, subtask))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', device)
    cpu_device = torch.device("cpu")

    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    train_dataloader = get_train_dataloader(task, subtask, tokenizer)
    val_dataloader = get_val_dataloader(task, subtask, tokenizer)

    train_losses = []
    val_losses = []
    min_val_loss = float('inf')
    for epoch in range(n_epochs):
        train_loss = 0.0
        model.train()
        count = 0
        for in_ids, in_masks, out_ids, out_masks in train_dataloader:
            in_ids = in_ids.to(device)
            in_masks = in_masks.to(device)
            out_masks = out_masks.to(device)
            out_ids = out_ids.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids = in_ids, attention_mask = in_masks, labels = out_ids)
            in_ids = in_ids.to(cpu_device)
            in_masks = in_masks.to(cpu_device)
            out_masks = out_masks.to(cpu_device)
            out_ids = out_ids.to(cpu_device)
            loss = outputs[0]
            train_loss += loss
            loss.backward()
            optimizer.step()
            count += 1
            if count % 10 == 0:
                print("\t\t Training Epoch {} Iteration {}".format(epoch+1, count))
        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)

        val_loss = 0.0
        count = 0
        with torch.no_grad():
            model.eval()
            for in_ids, in_masks, out_ids, out_masks in val_dataloader:
                in_ids = in_ids.to(device)
                in_masks = in_masks.to(device)
                out_masks = out_masks.to(device)
                out_ids = out_ids.to(device)
                outputs = model(input_ids = in_ids, attention_mask = in_masks, labels = out_ids)
                in_ids = in_ids.to(cpu_device)
                in_masks = in_masks.to(cpu_device)
                out_masks = out_masks.to(cpu_device)
                out_ids = out_ids.to(cpu_device)
                loss = outputs[0]
                val_loss += loss
                count += 1
                if count % 10 == 0:
                    print("\t\t Validation Epoch {} Iteration {}".format(epoch+1, count))
        val_loss /= len(val_dataloader)
        if val_loss<min_val_loss:
            print("Epoch {} model saved".format(epoch+1))
            min_val_loss = val_loss
            save_path = f"models/model_{task}_{subtask}"  # Define the path to save the model
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

        val_losses.append(val_loss)
        print("Epoch {}: Training Loss: {:.4f} Validation LossL {:.4f}".format(epoch+1, train_loss, val_loss))

def callback_fn(task, subtask, source_ids, tokenizer, batch_idx, input_ids):
    token_ids = {
        "A" : [188],
        "C" : [254],
        "O" : [667],
        "S" : [134],
        "SS" : [4256],
        "EP" : [8569],
        "[" : [784],
        "]" : [908],
        "null" : [206,195],
        "it" : [34],
        "</s>": [1],
        "":[3]
    }
    laptop_cate_tokens = []
    for s in laptop_aspect_cate_list:
        # laptop_cate_tokens.extend(tokenizer.__call__([s])["input_ids"][0][:-1])
        laptop_cate_tokens.append(tokenizer.__call__(s)["input_ids"][0])
    rest_cate_tokens = []
    for s in rest_aspect_cate_list:
        # rest_cate_tokens.extend(tokenizer.__call__([s])["input_ids"][0][:-1])
        rest_cate_tokens.append(tokenizer.__call__(s)["input_ids"][0])
    sentiment_tokens = []
    sentiment_tokens.append(tokenizer.__call__("good")["input_ids"][0])
    sentiment_tokens.append(tokenizer.__call__("bad")["input_ids"][0])
    sentiment_tokens.append(tokenizer.__call__("ok")["input_ids"][0])
  
    if token_ids['['][0] not in input_ids:
        return token_ids['['] + token_ids['</s>']
    if input_ids[-1] in token_ids['[']:
        if task == "aste":
            return token_ids['SS'] + token_ids['A'] + token_ids['O'] + token_ids['S']
        if task == "tasd":
            return token_ids['SS'] + token_ids['A'] + token_ids['C'] + token_ids['S']
        return token_ids['SS'] + token_ids['A'] + token_ids['C'] + token_ids['O'] + token_ids['S']
    if input_ids[-1] in token_ids['SS']:
        return token_ids['EP']
    if input_ids[-1] in token_ids['A'] + token_ids['C'] + token_ids['O'] + token_ids['S'] + token_ids['EP']:
        return token_ids[']']
    if len(input_ids)>1 and input_ids[-1] in token_ids[']'] and input_ids[-2] in token_ids['EP']:
        return token_ids['[']
    if len(input_ids)>2:
        if input_ids[-1] in token_ids[']'] and input_ids[-3] in token_ids['[']:
            if input_ids[-2] in token_ids['S']:
                return sentiment_tokens
            if input_ids[-2] in token_ids['C']:
                if "laptop" in subtask:
                    return laptop_cate_tokens
                if "rest" in subtask:
                    return rest_cate_tokens
            if input_ids[-2] in token_ids['A']:
                res = source_ids[batch_idx].tolist()
                res = set(res)
                for s in token_ids:
                    if len(token_ids[s])==1:
                        res.discard(token_ids[s][0])
                    else:
                        res.discard(token_ids[s][0])
                        res.discard(token_ids[s][1])
                res = list(res)
                res += token_ids['it']
                return res
            if input_ids[-2] in token_ids['O']:
                res = source_ids[batch_idx].tolist()
                res = set(res)
                for s in token_ids:
                    if len(token_ids[s])==1:
                        res.discard(token_ids[s][0])
                    else:
                        res.discard(token_ids[s][0])
                        res.discard(token_ids[s][1])
                res = list(res)
                res += token_ids['null']
                return res
          
def evaluate_function(task, subtask, model_name):

    model = T5ForConditionalGeneration.from_pretrained("models/"+model_name)
    tokenizer = T5Tokenizer.from_pretrained("models/"+model_name)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', device)
    cpu_device = torch.device("cpu")

    model = model.to(device)

    test_dataloader = get_test_dataloader(task, subtask, tokenizer)

    decoded_outputs = []
    targets = []
    output =[]
    with torch.no_grad():
        model.eval()
        count = 0
        for in_ids, in_masks, out_ids, out_masks in test_dataloader:
            in_ids = in_ids.to(device)
            in_masks = in_masks.to(device)
            out_masks = out_masks.to(device)
            out_ids = out_ids.to(device)
            generated_tokens = model.generate(
                input_ids=in_ids,
                attention_mask = in_masks,
                max_length = 100,
                prefix_allowed_tokens_fn = partial(callback_fn, task, subtask, in_ids, tokenizer)
            )
            in_ids = in_ids.to(cpu_device)
            in_masks = in_masks.to(cpu_device)
            out_masks = out_masks.to(cpu_device)
            out_ids = out_ids.to(cpu_device)
            for i in range(len(generated_tokens)):  # Iterate through each sequence in the batch
                decoded_target = tokenizer.decode(out_ids[i], skip_special_tokens=True)
                decoded_output = tokenizer.decode(generated_tokens[i], skip_special_tokens=True)
                targets.append(decoded_target)
                output.append(decoded_output)


                decoded_outputs.append((decoded_target, decoded_output))  # Store decoded sequences as tuples
            count += 1

            if count%10==0:
                print(f'Testing Iteration: {count}/{len(test_dataloader)}')


    num_path = 5
    targets = targets[::num_path]

    # get outputs
    _outputs = output # backup

    output = [] # new outputs

    for i in range(0, len(targets)):
        
        o_idx = i * num_path
        multi_outputs = _outputs[o_idx:o_idx + num_path]
        all_quads = []

        for s in multi_outputs:
            all_quads.extend(
                extract_spans_para(seq=s, seq_type='pred'))

        output_quads = []
    
        counter = dict(Counter(all_quads))
        for quad, count in counter.items():
        
            # keep freq >= num_path / 2
            if count >= len(multi_outputs) / 2:
                output_quads.append(quad)
    

        # recover output
        output_ = []
        for q in output_quads:
            
            ac, at, sp, ot = q        
            if task == "aste":
                if 'null' not in [at, ot, sp]:  # aste has no 'null', for zero-shot only
                    output_.append(f'[A] {at} [O] {ot} [S] {sp}')

                elif task == "tasd":
                    output_.append(f"[A] {at} [S] {sp} [C] {ac}")

                elif task in ["asqp", "acos"]:
                    output_.append(f"[A] {at} [O] {ot} [S] {sp} [C] {ac}")

                else:
                    raise NotImplementedError
            

        target_quads = extract_spans_para(seq=targets[i], seq_type='gold')
        
        # if no output, use the first path
        output_str = " [SSEP] ".join(output_) if output_ else multi_outputs[0]

        output.append(output_str)
        

            
    # stats
    labels_counts = Counter([len(l.split('[SSEP]')) for l in output])
    print("pred labels count", labels_counts)

    scores, all_labels, all_preds = compute_scores(output, targets, verbose=True)

    scores["model"] = model_name
    scores["task"] = task
    scores["subtask"] = subtask
    print(scores)
    with open("results/eval_{}_{}_model_{}.json".format(task, subtask, model_name), "w") as f:
        json.dump(scores, f, indent = 4)



