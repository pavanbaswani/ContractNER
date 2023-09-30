def get_predictions(text):
    chunk_size = 1024
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    tokens = inputs.tokens()
    
    predictions = []
    input_ids = inputs["input_ids"]
    token_chunks = [input_ids[:, i:i+chunk_size] for i in range(0, input_ids.shape[1], chunk_size)]
    
    for chunk in token_chunks:
        with torch.no_grad():
            logits = model(chunk).logits
        
        preds = torch.argmax(logits, dim=2)
        predictions.extend(preds[0].tolist())

    prediction_str = []
    for token, prediction in zip(tokens, predictions):
        predicted_tag = model.config.id2label[prediction]
        prediction_str.append(predicted_tag)
    
    return inputs.input_ids.squeeze(0), tokens, prediction_str


def get_maxlabel(labels):
    label_counts = {lab:labels.count(lab) for lab in labels}

    max_label = ''
    max_val = -1
    for lab, cont in label_counts.items():
        if cont>max_val:
            max_label = lab
            max_val = cont
    
    return max_label


def get_token_mapping(sentence):
    
    inputs, tokens, preds = get_predictions(sentence)

    assert len(tokens)==len(preds)
    space_tokens = [tok.strip() for tok in sentence.split(' ') if tok.strip()!='']

    token_mapping = []
    token_index = 0

    token = ""
    labels = []
    for tok, label in zip(tokens, preds):
        label = label.replace('I-', '').replace('B-', '')
        if tok.startswith('##'):
            token += tok
            labels.append(label)
        else:
            if space_tokens[token_index].lower() == token.replace('##', '').lower():
                token = tok
                labels.append(label)
                max_label = get_maxlabel(labels)
                token_mapping.append({'token': space_tokens[token_index], 'label': max_label})
                token_index += 1
                labels = []
            else:
                token += tok
                labels.append(label)

    if len(token)>0:
        max_label = get_maxlabel(labels)
        token_mapping.append({'token': space_tokens[token_index], 'label': max_label})
    
    return token_mapping

