import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Code.data.heb_data import create_street_names_data_iterators
from Code.models.heb_model import HebLetterToSentence
from Code.trianer.train_images import Trainer
import torch.nn as nn
from torch.optim import Adam


class Heb_trainer(Trainer):

    def __init__(self, train_dataloader, eval_dataloader, model, optim, summary_writer,
                 criteria=torch.nn.CrossEntropyLoss().cuda(), save_checkpoint=200):
        super(Heb_trainer, self).__init__(train_dataloader, eval_dataloader, model, optim, summary_writer,
                                          criteria=criteria, save_checkpoint=save_checkpoint)

    def train_epoch(self):
        avg_loss = 0
        for batch in tqdm(self.train_dataloader):
            self.model.train()
            self.optim.zero_grad()
            x, y = batch.chars.to(device),batch.names.to(device)  #x.to(device), y.to(device)
            h0 = self.model.init_state(x.shape[0]).to(device)
            out, ht = self.model(x.T,h0)
            loss = self.criteria(out.squeeze(1), y)
            avg_loss += loss.item()
            loss.backward()
            self.optim.step()
        return avg_loss / len(self.train_dataloader)

    def log_epoch(self, acc_log, avg_loss, device, epoch, loss_log, summary_writer):
        # display
        # acc_test = accuracy(self.model, self.eval_dataloader, device)
        # acc_train = accuracy(self.model, self.train_dataloader, device)
        print(f"loss[{epoch}] = {avg_loss / self.train_size}")
        # print(f"acc[{epoch}] = {acc_test.item()}")
        loss_log.append(avg_loss / self.train_size)
        summary_writer.add_scalar("loss_train_text", avg_loss, global_step=epoch)
        # summary_writer.add_scalar("accuracy_test", acc_test.item(), global_step=epoch)
        # summary_writer.add_scalar("accuracy_train",acc_train.item())
        # acc_log.append(acc_test)

if __name__ == '__main__':
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    path_to_text_data = "D:\\Alon_temp\\singlang\\singLang_DLProg\\text_data\\split"
    train_iterator, test_iterator, char_vocab, name_vocab, _, _ = create_street_names_data_iterators(path_to_text_data)
    print(train_iterator)
    model = HebLetterToSentence(len(char_vocab), 128, 128, 128, len(name_vocab))
    model = model.cuda()
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.0001)
    epochs = 50
    writer = SummaryWriter(comment="text_train")

    runexample = next(iter(train_iterator))
    print(len(runexample))
    model = model.to(device)
    # print model in tensorboard
    init_state = model.init_state(18)
    writer.add_graph(model, (runexample.chars.to(device).T, init_state.to(device)))

    trainer =  Heb_trainer(train_iterator, test_iterator, model, optimizer, writer,save_checkpoint=5)

    trainer.train("text_run", device=device)

    # def train(self, run_name, epochs=100, device='cpu', out_path='./', ):

#### before trainer class
# import os
#
# import torch
# import tqdm
# import torch.nn as nn
# from torch.optim import Adam
# import numpy as np
# from Code.data.heb_data import create_street_names_data_iterators
# from Code.models.heb_model import HebLetterToSentence
# import matplotlib.pyplot as plt
#
# from Code.utils.helpers import save_model
#
#
# def train(train_iter,model,loss_function,optimizer,epochs,device='cuda'):
#     loss_log = []
#     for epoch in range(epochs):
#         epoch_losses = list()
#         for batch in tqdm.tqdm(train_iter):
#             optimizer.zero_grad()
#             h0 = model.init_state(batch.chars.shape[0])
#             h0 = h0.to(device)
#             y_pred,(ht) = model(batch.chars.T, h0)
#             loss = loss_function(y_pred.squeeze(1), batch.names)
#
#             loss.backward()
#             optimizer.step()
#
#             epoch_losses.append(loss.item())
#
#         mean_loss = np.mean(epoch_losses)
#         loss_log.append(mean_loss)
#         print('train loss on epoch {} : {:.3f}'.format(epoch, mean_loss))
#     return loss_log, model
#
#
#
# if __name__ == '__main__':
#     # path = ''
#     path = "D:\\Alon_temp\\singlang\\singLang_DLProg\\text_data\\split"
#     out_path = 'D:\\Alon_temp\\singlang\\singLang_DLProg\\out_puts'
#     train_iterator, test_iterator,char_vocab,name_vocab = create_street_names_data_iterators(path)
#     print(len(char_vocab))
#     print(len(name_vocab))
#     # train_iterator.
#     # dl
#     model = HebLetterToSentence(len(char_vocab),128,128,128,len(name_vocab))
#     model = model.cuda()
#     loss_function = nn.CrossEntropyLoss()
#     optimizer = Adam(model.parameters(), lr=0.0001)
#     epochs = 50
#     train_loss,model = train(train_iterator,model,loss_function,optimizer,epochs)
#     run_name = ""
#     model_filename = os.path.join(out_path, f"final_lstm_{run_name}.pt")
#     save_model(model,model_filename)
#     plt.plot(train_loss)
#     plt.show()
