def load_json(json_path):
	json_data = {}

	with open(json_path, 'r', encoding='utf-8') as fp:
		json_data = json.load(fp)

	return json_data


def write_json(json_path, json_data):
	with open(json_path, 'w', encoding='utf-8') as fp:
		json.dump(json_data, fp)


def reformat_data(text_path):
    data = []

    tokens = []
    tags = []
    if os.path.exists(text_path):
        with open(text_path, 'r', encoding='utf-8') as fp:
            for line in fp.readlines():
                line = line.strip().replace('\n', '').replace('\r', '')
                if line=='':
                    assert len(tokens) == len(tags)
                    flag = False
                    for tag in tags:
                        if tag!=3:
                            flag = True
                            break
                    if flag:
                        data.append({'tokens': tokens, 'tags': tags})
                    tokens = []
                    tags = []
                else:
                    temp = line.split('\t')
                    tokens.append(temp[0])
                    tags.append(label2id.get(str(temp[1]), label2id.get('O')))
    else:
        print("Path: {} do not exists...!".format(text_path))
    
    return data


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs