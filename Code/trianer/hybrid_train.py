import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Code.data.dataloaders import get_image_dataloader
from Code.data.heb_data import create_street_names_data_iterators
from Code.data.hybrid_data import HybridDataSet
from Code.models.heb_model import HebLetterToSentence
from Code.models.hybrid_model import HybridModel
from Code.trianer.train_images import Trainer
from Code.utils.helpers import load_resnet_model, set_grads_to_false


class HybridTrainer(Trainer):
    def __init__(self, train_dataloader, eval_dataloader, model, optim, summary_writer,
                 criteria=torch.nn.CrossEntropyLoss().cuda(), save_checkpoint=200, device='cuda'):
        super(HybridTrainer, self).__init__(train_dataloader, eval_dataloader, model, optim, summary_writer,
                                          criteria=criteria, save_checkpoint=save_checkpoint)

        self.device = device


    def train_epoch(self):
        avg_loss = 0
        for x, y in tqdm(self.train_dataloader):
            self.model.train()
            self.optim.zero_grad()
            x, y = x.to(self.device), y.to(self.device)
            y_hat, _ = self.model(x)
            softmax = torch.softmax(y_hat.squeeze(1), dim=1)
            loss = self.criteria(softmax, y)
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
    ## run 1
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    path_to_data = "D:\\Alon_temp\\singlang\\singLang_DLProg\\images\\coloredCaptureData\\train"
    # path_to_data = "D:\\Alon_temp\\singlang\\singLang_DLProg\\images\\coloredCaptureData"
    _,imageDataSet = get_image_dataloader(path_to_data)

    # path = "C:\\HW\\singLang_DLProg\\text_data\\split"
    path = "D:\\Alon_temp\\singlang\\singLang_DLProg\\text_data\\split"

    train_iterator, test_iterator,vocab_letters,vocab_words,train_data,test_data,names  = create_street_names_data_iterators(path)

    hybridDataSet = HybridDataSet(imageDataSet,train_data,vocab_words)
    dataloader = DataLoader(dataset=hybridDataSet, batch_size=4, shuffle=True)
    # someItem = hybridDataSet.__getitem__(2)
    # print(someItem[0].shape)
    # print(hybridDataSet.word_max_len)
    # def __init__(self, vocab_size,embedding_dim, lstm_size,hidden_dim, output_dim):
    print("letter",len(vocab_letters))
    print("words",len(vocab_words))
    # text_model = HebLetterToSentence(len(vocab_letters), 128, 512, 128,len(vocab_words),use_self_emmbed=True)
    # image_model = load_resnet_model('')
    # # exit()
    # # image_model = set_grads_to_false(image_model)
    # # exit(12)
    # model = HybridModel(image_model,text_model).to(device)
    # optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    #
    # writer = SummaryWriter(comment="full_train_no_pretrained")
    # # print(model)
    # # x,y = next(iter(dataloader))
    # print(model(x))
    # trainer = HybridTrainer(dataloader, test_iterator, model, optimizer, writer,save_checkpoint=5,device=device)
    #
    # trainer.train("full_train_no_pretrained",epochs=50, device=device)
    # writer.flush()
    # writer.close()
    # # exit(12
    #
    # ##run 2
    # use_gpu = True
    # device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    #
    # print("letter", len(vocab_letters))
    # print("words", len(vocab_words))
    # text_model = HebLetterToSentence(len(vocab_letters), 128, 512, 128, len(vocab_words), use_self_emmbed=True)
    # image_model = load_resnet_model(
    #     'D:\\Alon_temp\\singlang\\singLang_DLProg\\out_puts\\final_resnet_with_aug_colored_pretrained_test_run_64_trainer.pt')
    #
    # model = HybridModel(image_model, text_model).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #
    # writer = SummaryWriter(comment="full_train_pretrained")
    #
    # trainer = HybridTrainer(dataloader, test_iterator, model, optimizer, writer, save_checkpoint=5, device=device)
    #
    # trainer.train("full_run_test_pretrained", epochs=50, device=device)
    # writer.flush()
    # writer.close()
    # exit(12)
    ## run 3
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    print("letter", len(vocab_letters))
    print("words", len(vocab_words))
    text_model = HebLetterToSentence(len(vocab_letters), 128, 512, 128, len(vocab_words), use_self_emmbed=True)
    image_model = load_resnet_model(
        'D:\\Alon_temp\\singlang\\singLang_DLProg\\out_puts\\final_resnet_with_aug_colored_pretrained_test_run_64_trainer.pt')
    # exit()
    image_model = set_grads_to_false(image_model)
    # exit(12)
    model = HybridModel(image_model, text_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    writer = SummaryWriter(comment="full_train_pretrained_resnet_no_image_update")
    # print(model)
    # x,y = next(iter(dataloader))
    # print(model(x))
    trainer = HybridTrainer(dataloader, test_iterator, model, optimizer, writer, save_checkpoint=5, device=device)

    trainer.train("full_run_test_no_image_update", epochs=10, device=device)

    writer.flush()
    writer.close()

