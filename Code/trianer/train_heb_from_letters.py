import os

import torch
import tqdm
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from Code.data.heb_data import create_street_names_data_iterators
from Code.models.heb_model import HebLetterToSentence
import matplotlib.pyplot as plt

from Code.utils.helpers import save_model


def train(train_iter,model,loss_function,optimizer,epochs,device='cuda'):
    loss_log = []
    for epoch in range(epochs):
        epoch_losses = list()
        for batch in tqdm.tqdm(train_iter):
            optimizer.zero_grad()
            h0 = model.init_state(batch.chars.shape[0])
            h0 = h0.to(device)
            y_pred,(ht) = model(batch.chars.T, h0)
            loss = loss_function(y_pred.squeeze(1), batch.names)

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        mean_loss = np.mean(epoch_losses)
        loss_log.append(mean_loss)
        print('train loss on epoch {} : {:.3f}'.format(epoch, mean_loss))
    return loss_log, model



if __name__ == '__main__':
    # path = ''
    path = "D:\\Alon_temp\\singlang\\singLang_DLProg\\text_data\\split"
    out_path = 'D:\\Alon_temp\\singlang\\singLang_DLProg\\out_puts'
    train_iterator, test_iterator,char_vocab,name_vocab = create_street_names_data_iterators(path)
    print(len(char_vocab))
    print(len(name_vocab))
    # train_iterator.
    # dl
    model = HebLetterToSentence(len(char_vocab),128,128,128,len(name_vocab))
    model = model.cuda()
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.0001)
    epochs = 50
    train_loss,model = train(train_iterator,model,loss_function,optimizer,epochs)
    run_name = ""
    model_filename = os.path.join(out_path, f"final_lstm_{run_name}.pt")
    save_model(model,model_filename)
    plt.plot(train_loss)
    plt.show()
