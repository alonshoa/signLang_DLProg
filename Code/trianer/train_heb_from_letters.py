import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from Code.data.heb_data import create_street_names_data_iterators
from Code.models.heb_model import HebLetterToSentence



def train(train_iter,model,loss_function,optimizer,epochs,device='cuda'):

    for epoch in range(epochs):
        epoch_losses = list()
        for batch in train_iter:
            optimizer.zero_grad()
            h0 = model.init_state(batch.chars.shape[0])
            h0 = h0.to(device)
            y_pred,(ht) = model(batch.chars.T, h0)
            # print(y_pred.shape)
            # print(batch.names.shape)
            loss = loss_function(y_pred.squeeze(1), batch.names)

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
        print('train loss on epoch {} : {:.3f}'.format(epoch, np.mean(epoch_losses)))






if __name__ == '__main__':
    # path = ''
    path = "D:\\Alon_temp\\singlang\\singLang_DLProg\\text_data\\split"
    train_iterator, test_iterator,char_vocab,name_vocab = create_street_names_data_iterators(path)
    print(len(char_vocab))
    print(len(name_vocab))
    # train_iterator.
    # dl
    model = HebLetterToSentence(len(char_vocab),128,128,128,len(name_vocab))
    model = model.cuda()
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.01)
    epochs = 5
    train(train_iterator,model,loss_function,optimizer,epochs)