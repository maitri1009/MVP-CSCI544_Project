from data_store import *

def parse_labels(sent, label, task):
    if task == "acos":
        a,c,s,o = label
    elif task == "asqp":
        a,c,o,s = label
    elif task == "aste":
        a,o,s = label
        c = None
        n = len(sent)
        if a:
            if type(a) == list and len(a)>0:
                i = a[0]
                j = a[-1]
                if j==n-1:
                    a = " ".join(sent[i:])
                else:
                    a = " ".join(sent[i:j+1])
            elif type(a) == list:
                a = None
        if o:
            if type(o) == list and len(o)>0:
                i = o[0]
                j = o[-1]
                if j==n-1:
                    o = " ".join(sent[i:])
                else:
                    o = " ".join(sent[i:j+1])
            elif type(o) == list:
                o = None
    elif task == "tasd":
        a,c,s = label
        o = None
    else:
        raise NotImplementedError
    if a == "null" or a.lower() == "null":
        a = "it"
    if s and type(s) == str and s.lower() in {"positive","pos"}:
        s = "good"
    elif s and type(s) == str and s.lower() in {"negative","neg"}:
        s = "bad"
    else:
        s = "ok"
    return {"[O]":o, "[A]":a, "[S]":s, "[C]":c}

def get_ordered_labels(sent, labels, task, subtask):
    # Get best orders from training

    o_and_ls = []
    best_orders = best_orders_stored[task][subtask][:5]
    for order in best_orders:
        targets = []
        for label in labels:

            label_dict = parse_labels(sent.split(), label, task)
            order_parse = order.split()
            target = []
            for term_type in order_parse:
                target.append(term_type + " " + label_dict[term_type])
            targets.append(" ".join(target))
        o_and_ls.append((order, " [SSEP] ".join(targets)))
    return o_and_ls

def convert_dataset(lines, task, subtask):
    res = []
    for line in lines:
        line = line.lower()
        line = line.split("####")
        sent = line[0].strip()
        labels = eval(line[1].strip())
        order_and_labels = get_ordered_labels(sent, labels, task, subtask)
        for order,label in order_and_labels:
            res.append((sent + " " + order, label))
    return res