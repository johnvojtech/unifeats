#!python3
import sys
import preprocess
import models
import numpy as np
import keras


# Dict of possible morphological features and values (for Czech)

CS_FV = {'Abbr': ['Yes'], 'AdpType': ['Comprep', 'Prep', 'Voc'], 'Animacy': ['Anim', 'Inan'], 'Aspect': ['Imp', 'Perf'], 'Case': ['Acc', 'Dat', 'Gen', 'Ins', 'Loc', 'Nom', 'Voc'], 'ConjType': ['Oper'], 'Degree': ['Cmp', 'Pos', 'Sup'], 'Foreign': ['Yes'], 'Gender': ['Fem', 'Fem,Masc', 'Fem,Neut', 'Masc', 'Masc,Neut', 'Neut'], 'Gender[psor]': ['Fem', 'Masc', 'Masc,Neut'], 'Hyph': ['Yes'], 'Mood': ['Cnd', 'Imp', 'Ind'], 'NameType': ['Com', 'Com,Geo', 'Com,Giv', 'Com,Giv,Sur', 'Com,Nat', 'Com,Oth', 'Com,Pro', 'Com,Pro,Sur', 'Com,Sur', 'Geo', 'Geo,Giv', 'Geo,Giv,Sur', 'Geo,Oth', 'Geo,Pro', 'Geo,Sur', 'Giv', 'Giv,Nat', 'Giv,Oth', 'Giv,Pro', 'Giv,Pro,Sur', 'Giv,Sur', 'Nat', 'Nat,Sur', 'Oth', 'Oth,Sur', 'Pro', 'Pro,Sur', 'Sur'], 'Number': ['Dual', 'Plur', 'Plur,Sing', 'Sing'], 'Number[psor]': ['Plur', 'Sing'], 'NumForm': ['Digit', 'Roman', 'Word'], 'NumType': ['Card', 'Frac', 'Mult', 'Mult,Sets', 'Ord', 'Sets'], 'NumValue': ['1', '1,2,3'], 'Person': ['1', '2', '3'], 'Polarity': ['Neg', 'Pos'], 'Poss': ['Yes'], 'PrepCase': ['Npr', 'Pre'], 'PronType': ['Dem', 'Emp', 'Ind', 'Int,Rel', 'Neg', 'Prs', 'Rel', 'Tot'], 'Reflex': ['Yes'], 'Style': ['Arch', 'Coll', 'Expr', 'Rare', 'Slng', 'Vrnc', 'Vulg'], 'Tense': ['Fut', 'Past', 'Pres'], 'Typo': ['Yes'], 'Variant': ['Short'], 'VerbForm': ['Conv', 'Fin', 'Inf', 'Part'], 'Voice': ['Act', 'Pass']}

# Maximum length of input word; trimmed if longer. 
WL = 30

# Location of the UD files in the file system.
UD_PATH = "/lnet/ms/data/universal-dependencies-2.9/"

def print_attentions(model, name, data_x, data_y):
    """
    Pretty-print attention scores.
    Arguments:
    ----------
    model: keras model, returning attention scores on output
    name: string - name of the feature
    data_x: numpy array - input data, i.e. ord encoded words
    data_y: numpy array - output data, i.e. feature labels (to print attention only if feature is represented in the word)
    """
    predicted = model.predict(data_x)[:,0,:,:]
    print(predicted.shape)
    print(data_x.shape)
    for word, item in zip(data_x, zip(list(predicted), data_y)):
        word = " ".join([chr(x) for x in word if x != 0])
        if item[1][0] == 1:
            continue
        else:
            print("###", word, name)
            print("horizontal_sum", [float(x) for x in np.sum(item[0], axis=-2)])
            print("vertical_sum", [float(x) for x in np.sum(item[0], axis=-1)])
            print("diagonal", [item[0][i, i] for i in range(WL)])
            print("attention_whole:")
            for i in range(item[0].shape[0]):
                print(list(item[0][i, :]))
            print()


def train_and_evaluate(train_path, test_path=None, lang="unk"):
    features = list(CS_FV.keys())
    features.sort()
    feat_lens = [len(CS_FV[key]) + 1 for key in features]
    x, y = preprocess.load_ud(UD_PATH+train_path, WL, CS_FV)
    if test_path != None:
        x2, y2 = preprocess.load_ud(UD_PATH+test_path, WL, CS_FV)
    else:
        split = int(len(x)*0.75)
        x, x2 = x[:split, :], x[split:, :]
        y, y2 = [z[:split, :] for z in y], [z[split:, :] for z in y]

    for i in range(len(features)):
        name = features[i].replace("[","_").replace("]","")
        print(name)
        callback = keras.callbacks.EarlyStopping(monitor='loss')
        model, model_attention = models.feature_attention_model(WL, feat_lens[i])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=[keras.losses.CategoricalCrossentropy()], metrics=[keras.metrics.CategoricalAccuracy()])
        model.fit(x=x, y=y[i], validation_split=0.05, batch_size=128, epochs=25, verbose=2, callbacks=[callback])
        model.evaluate(x2, y2[i])
        model.save("_".join(["models/model", lang, name, "_attention.keras"]))
        model_attention.compile()
        model_attention.save("_".join(["models/model", lang, name, "_attention.keras"]))
        print_attentions(model_attention, features[i], x2[:500], y2[i][:500])

if __name__ == "__main__":
    lang = sys.argv[1]
    train_path = sys.argv[2]
    if len(sys.argv) > 3:
        test_path = sys.argv[3]
    else:
        test_path = None
    train_and_evaluate(train_path, test_path, lang)
