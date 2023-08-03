import pickle
import random
import importlib

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.metrics import f1_score
from sklearn.manifold import TSNE

from tqdm import tqdm
import sys

random.seed(0)
np.random.seed(0)

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

plt.rcParams["figure.figsize"] = (20, 10)

ds_name = sys.argv[1]

link_prediction = False
if sys.argv[2] == "True":
    link_prediction = True

multilabel = False
if "dblp" in ds_name:
    multilabel = True

is_sparse = False

if link_prediction:
    ds_folder = "link_prediction"
else:
    ds_folder = "node_classification"

with open("data/{}/{}.pkl".format(ds_folder, ds_name), "rb") as f:
    ds = pickle.load(f)

    features = ds["features"]
    adjs = ds["adjs"]
    labels = ds["labels"]

    train_idx = ds["train_idx"]
    valid_idx = ds["val_idx"]
    test_idx = ds["test_idx"]

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_relations = len(adjs)
    nb_classes = labels.shape[1]

from utils import *

# Preprocess features and adjacency matrices
features = preprocess_features(features)
adjs_norm = [normalized_laplacian(adj + np.eye(nb_nodes) * 3.0, is_sparse) for adj in adjs]

# Create tensors
features = torch.FloatTensor(features[np.newaxis])

if not is_sparse:
    adjs_norm = [torch.FloatTensor(adj[np.newaxis]) for adj in adjs_norm]
else:
    adjs_norm = [sparse_mx_to_torch_sparse_tensor(adj) for adj in adjs_norm]

labels = torch.FloatTensor(labels)

if not multilabel:
    labels = torch.argmax(labels, dim=1)

train_idx = torch.LongTensor(train_idx)
valid_idx = torch.LongTensor(valid_idx)
test_idx = torch.LongTensor(test_idx)

train_labels = labels[train_idx].squeeze()
valid_labels = labels[valid_idx].squeeze()
test_labels = labels[test_idx].squeeze()

if torch.cuda.is_available():
    features = features.cuda()
    adjs_norm = [adj.cuda() for adj in adjs_norm]

    train_labels = train_labels.cuda()
    valid_labels = valid_labels.cuda()
    test_labels = test_labels.cuda()

# Get labels for infomax
lbl_1 = torch.ones(nb_nodes)
lbl_0 = torch.zeros(nb_nodes)
infomax_labels = torch.cat((lbl_1, lbl_0))

if torch.cuda.is_available():
    infomax_labels = infomax_labels.cuda()

from hmge import HMGE

if "biogrid_4211" in ds_name:
    hid_units = 32
    drop_prob = 0.0
    is_attn = False
    common_gcn = True
    single_gcn = False
    normalize_z = False
    lr = 0.001
    l2_coef = 1e-05
    n_epochs = 2000
    patience = 100

elif "dblp_5124" in ds_name:
    hid_units = 64
    drop_prob = 0.1
    is_attn = True
    common_gcn = True
    single_gcn = False
    normalize_z = False
    lr = 0.001
    l2_coef = 1e-05
    n_epochs = 2000
    patience = 20

elif "imdb_3000" in ds_name:
    hid_units = 64
    drop_prob = 0.0
    is_attn = False
    common_gcn = True
    single_gcn = False
    normalize_z = False
    lr = 0.001
    l2_coef = 1e-05
    n_epochs = 2000
    patience = 20

elif "STRING-DB_4083" in ds_name:
    hid_units = 64
    drop_prob = 0.1
    is_attn = False
    common_gcn = True
    single_gcn = False
    normalize_z = False
    lr = 0.001
    l2_coef = 1e-05
    n_epochs = 2000
    patience = 20

elif ds_name == "biogrid_4503_bis" and not link_prediction:
    hid_units = 128
    drop_prob = 0.0
    is_attn = False
    common_gcn = True
    single_gcn = True
    normalize_z = False
    lr = 0.001
    l2_coef = 0.0
    n_epochs = 8000
    patience = 8000

elif ds_name == "biogrid_4503_bis" and link_prediction:
    hid_units = 128
    drop_prob = 0.0
    is_attn = False
    common_gcn = True
    single_gcn = False
    normalize_z = False
    lr = 0.001
    l2_coef = 0.0
    n_epochs = 2000
    patience = 20

model = HMGE(ft_size, hid_units, nb_relations, drop_prob, is_attn, common_gcn, single_gcn, normalize_z)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

if torch.cuda.is_available():
    model = model.cuda()

best_loss = 1e9
best_epoch = 0
cnt_wait = 0

bce_loss = nn.BCEWithLogitsLoss()
loss_history = []

for epoch in tqdm(range(n_epochs)):
    model.train()
    optimizer.zero_grad()

    # Shuffle features
    idx = np.random.permutation(nb_nodes)
    fts_shuf = features[:, idx, :]
    if torch.cuda.is_available():
        fts_shuf = fts_shuf.cuda()

    logits = model(features, adjs_norm, fts_shuf, is_sparse)

    # Compute loss
    loss = bce_loss(logits.squeeze(), infomax_labels)
    loss_history.append(loss.item())

    if loss < best_loss:
        best_loss = loss.item()
        best_epoch = epoch
        cnt_wait = 0

        torch.save(model.state_dict(), "results/{}/best_hmge_{}.pkl".format(ds_folder, ds_name))

    else:
        cnt_wait += 1

    if cnt_wait == patience:
        break
    
    loss.backward()
    optimizer.step()

model.load_state_dict(torch.load("results/{}/best_hmge_{}.pkl".format(ds_folder, ds_name)))

if not link_prediction:
    model.eval()
    with torch.no_grad():
        z = model.embed(features, adjs_norm, is_sparse)
        torch.save(z, "results/hmge_embs/{}/{}.pkl".format(ds_folder, ds_name))

        z_train = z[0, train_idx].squeeze()
        z_valid = z[0, valid_idx].squeeze()
        z_test = z[0, test_idx].squeeze()

    if not multilabel:
        from layers import LogRegModel

        n_epochs = 100
        lr = 0.1
        l2_coef = 0
        ce_loss = nn.CrossEntropyLoss()

        valid_accuracy_list = []
        test_accuracy_list = []
        test_f1_macro_list = []
        test_f1_micro_list = []

        for _ in range(50):
            log_reg_model = LogRegModel(hid_units, nb_classes)
            optimizer = torch.optim.Adam(log_reg_model.parameters(), lr=lr, weight_decay=l2_coef)

            if torch.cuda.is_available():
                log_reg_model = log_reg_model.cuda()

            best_accuracy = 0
            best_epoch = 0

            for epoch in range(n_epochs):
                log_reg_model.train()
                optimizer.zero_grad()

                train_logits = log_reg_model(z_train)
                loss = ce_loss(train_logits, train_labels)

                log_reg_model.eval()
                with torch.no_grad():
                    valid_logits = log_reg_model(z_valid)

                    valid_accuracy = torch.sum(valid_labels == torch.argmax(valid_logits, dim=1)) / len(valid_labels)
                    if valid_accuracy > best_accuracy:
                        best_accuracy = valid_accuracy.item()
                        best_epoch = epoch

                        torch.save(log_reg_model.state_dict(), "results/{}/best_log_reg_{}.pkl".format(ds_folder, ds_name))
                
                loss.backward()
                optimizer.step()

            valid_accuracy_list.append(best_accuracy)

            log_reg_model.load_state_dict(torch.load("results/{}/best_log_reg_{}.pkl".format(ds_folder, ds_name)))
            log_reg_model.eval()

            with torch.no_grad():
                test_logits = log_reg_model(z_test)
                test_preds = torch.argmax(test_logits, dim=1)

                test_accuracy = torch.sum(test_labels == test_preds) / len(test_labels)
                test_f1_macro = f1_score(test_labels.cpu(), test_preds.cpu(), average="macro")
                test_f1_micro = f1_score(test_labels.cpu(), test_preds.cpu(), average="micro")

                test_accuracy_list.append(test_accuracy.item())
                test_f1_macro_list.append(test_f1_macro)
                test_f1_micro_list.append(test_f1_micro)

        print(ds_name)
        print("Test accuracy :", np.mean(test_accuracy_list))
        print("Test F1-macro", np.mean(test_f1_macro_list))
        print("Test F1-micro", np.mean(test_f1_micro_list))
        print("")

    else:
        from layers import MultilabelLogRegModel

        n_epochs = 100
        lr = 0.1
        l2_coef = 0
        bce_loss = nn.BCELoss()

        test_f1_macro_list = []
        test_f1_micro_list = []

        for _ in range(50):
            log_reg_model = MultilabelLogRegModel(hid_units, nb_classes)
            optimizer = torch.optim.Adam(log_reg_model.parameters(), lr=lr, weight_decay=l2_coef)

            if torch.cuda.is_available():
                log_reg_model = log_reg_model.cuda()

            best_f1 = 0
            best_epoch = 0

            for epoch in range(n_epochs):
                log_reg_model.train()
                optimizer.zero_grad()

                train_logits = log_reg_model(z_train)
                loss = bce_loss(train_logits, train_labels)

                log_reg_model.eval()
                with torch.no_grad():
                    valid_logits = log_reg_model(z_valid).cpu()
                    valid_preds = np.array(valid_logits > 0.5, dtype=float)

                    valid_f1 = f1_score(valid_labels.cpu(), valid_preds, average="macro")
                    if valid_f1 > best_f1:
                        best_f1 = valid_f1.item()
                        best_epoch = epoch

                        torch.save(log_reg_model.state_dict(), "results/{}/best_multilabellog_reg_{}.pkl".format(ds_folder, ds_name))
                
                loss.backward()
                optimizer.step()

            log_reg_model.load_state_dict(torch.load("results/{}/best_multilabellog_reg_{}.pkl".format(ds_folder, ds_name)))
            log_reg_model.eval()

            with torch.no_grad():
                test_logits = log_reg_model(z_test).cpu()
                test_preds = np.array(test_logits > 0.5, dtype=float)

                test_f1_macro = f1_score(test_labels.cpu(), test_preds, average="macro")
                test_f1_micro = f1_score(test_labels.cpu(), test_preds, average="micro")

                test_f1_macro_list.append(test_f1_macro)
                test_f1_micro_list.append(test_f1_micro)

        print(ds_name)
        print("Test F1-macro", np.mean(test_f1_macro_list))
        print("Test F1-micro", np.mean(test_f1_micro_list))
        print("")

else:
    model.eval()
    with torch.no_grad():
        z = model.embed(features, adjs_norm, is_sparse).cpu()
        torch.save(z, "results/hmge_embs/{}/{}.pkl".format(ds_folder, ds_name))
        
        link_prediction_array = ds["link_prediction_array"]

    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import average_precision_score

    z_a = z[0, link_prediction_array[:, 0]]
    z_b = z[0, link_prediction_array[:, 1]]
    labels = link_prediction_array[:, 2]

    scores = []
    for i in range(len(z_a)):
        scores.append(torch.sigmoid(torch.dot(z_a[i], z_a[i])))

    print(ds_name)
    print("Link prediction ROC-AUC :", roc_auc_score(labels, scores))
    print("Average precision score :", average_precision_score(labels, scores))
    print("")