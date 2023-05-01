import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import pickle

def inputs_targets_from_data(df):
    """ separate dictionary of document embeddings, labels into (numpy) inputs and targets"""

    # create np array with shape (samples, features)
    x, y = np.stack(df["embedding"].values), np.stack(df["label"].values)

    # check that shapes match
    assert y.shape[0] == x.shape[0]
    print(x.shape, y.shape)

    return x,y

def plot_clf_results(scores, models):
    """ plots scores of classifiers"""

    # normalize scores to [0,1]
    norm = matplotlib.colors.Normalize(vmin=min(scores), vmax=max(scores), clip=True)

    # set colors
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.PiYG)

    plt.bar(
        x = range(len(scores)),
        height= scores,
        tick_label=models,
        color = mapper.to_rgba(scores)
    )
    plt.xticks(rotation=45)
    plt.show()

def clf_train_loop(models,train_data,test_data):
    """ train and evaluate models from a dictionary of models """

    scores, labels = [], []

    # separate data to inputs, targets
    x_train, y_train = inputs_targets_from_data(train_data)
    x_test, y_test = inputs_targets_from_data(test_data)

    for name, model in models.items():
      
        try:
            # if already trained, just read model in
            with open(f"pkls/{name}", 'rb') as pkl:
                model = pickle.load(pkl)
            print(f"loaded {name} from pickle")
        
        except:
  
            # if not trained, train, evaluate and pickle
            model["score"] = model["model"].fit(x_train, y_train).score(x_test, y_test)

            # pickle model
            with open(f"pkls/{name}", 'wb') as pkl:
                pickle.dump(model, pkl)
            print(f"{name} trained,evaluated and pickled")
        
        # model contains the model object and score, insert this into dictionary of models
        models[name] = model

    # collect all scores and names of classifiers
    for name, model in models.items():
        scores.append(model["score"])
        labels.append(name)
    
    # plot results
    plot_clf_results(scores, labels)