#!python3
from collections import defaultdict
import numpy as np

def load_ud(path:str, limit:int, feat_values:dict, tokenizer=None):
    """
    Load data from Universal Dependencies.
    Arguments:
    ----------
    path(str): path to the UD file
    limit(int): limit on word length; the word representations will be cropped/padded to fit this
    feat_values(dict): dictionary; list_of_feature_values = <feat_values>[feature]
    tokenizer: Words can be tokenized using the tokenizer (to subword ids instead of characters)

    Returns:
    --------
    input_data, output_data: 
        input_data: numpy array of shape (no_of_examples,<word_limit>)
        output_data: list of numpy arrays (one for each feature), each of shape (no_of_examples, feature_values + 1 (feature not present))
    """

    data = []
    multi_cnt, multi_form, multi_lemma, multi_feats = 0, "", "", {}
    with open(path, "r") as r:
        for line in r:
            # Skip headers and empty lines
            if line[:1] == "#" or len(line) == 0:
                continue

            ln = line.strip().split("\t")

            # Check length of given line is correct
            if len(ln) < 6 :
                continue

            # Start of multiword token
            elif "-" in ln[0]:
                x = ln[0].split("-")
                a, b = int(x[0]), int(x[1])
                multi_cnt = b-a
                multi_form = ln[1].lower()
                multi_lemma = ln[2]

            # If no morphological features, skip
            elif ln[5] == "_":
                continue

            # Accumulate multiword token features, eventually add to training data
            elif multi_cnt > 0:
                features = ln[5].split("|")
                features = [feature.split("=") for feature in features]
                for item in features:
                    if len(item) < 2:
                        print(item)
                    multi_feats[item[0]] = item[1]
                multi_cnt -= 1

                if multi_cnt == 0:
                    data.append([multi_form, multi_feats])
                    multi_form, multi_lemma, multi_feats = "", "", {}              

            # Add word to training data. 
            else:
                form = ln[1].lower()
                lemma = ln[2]
                features = ln[5].split("|")
                features = [feature.split("=") for feature in features]
                features = {x[0]:x[1] for x in features}
                data.append([form, features])


    # Data transformations (to fit the goal format)
    x = []
    y = [[] for f in feat_values.keys()]

    # list of feature names
    features = list(feat_values.keys())
    features.sort()
    feat_lens = [len(feat_values[x]) + 1 for x in features]

    for item in data:
        # Transform words to (padded and trimmed) lists of characters with symbols for start and end of sequence
        # Use <tokenizer>, if provided
        if tokenizer == None:
            w = [ord(x) for x in "<" + item[0] + ">"] + [0 for _ in range(limit - len(item[0]))]
        else:
            w = tokenizer.encode_ids(["<" + item[0] + ">"]) + [0 for _ in range(limit)]
        w = w[:limit]
        x.append(w)

        # Append each feature label to their respective output data.
        for i in range(len(features)):
            feature = features[i]
            if feature in item[1].keys():
                y[i].append(feat_values[feature].index(item[1][feature]) + 1)
            else:
                y[i].append(0)

    # transform to numpy arrays / list of numpy arrays
    x = np.array(x)
    y = [np.eye(feat_lens[i])[y[i]] for i in range(len(features))]
    return x, y
