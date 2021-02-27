import os
import torch
from tqdm import tqdm
from Code.data.dataloaders import get_image_dataloader
from Code.trianer.evaluate import accuracy
from Code.utils.helpers import load_resnet_model, get_pretrained_resnet
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, train_dataloader,eval_dataloader, model, optim,summary_writer,
                 criteria=torch.nn.CrossEntropyLoss().cuda(), save_checkpoint=200):
        self.summary_writer = summary_writer
        self.save_checkpoint = save_checkpoint
        self.criteria = criteria
        self.optim = optim
        self.model = model
        self.eval_dataloader = eval_dataloader
        self.train_dataloader = train_dataloader
        self.train_size = len(self.train_dataloader)
        self.eval_size = len(self.eval_dataloader)


    def train(self, run_name,epochs=100,device='cpu',out_path='./',):
        loss_log = []
        acc_log = []
        run_out_path = os.path.join(out_path, run_name)
        os.makedirs(run_out_path,exist_ok=True)
        if self.summary_writer is None:
            summary_writer = SummaryWriter()
        else:
            summary_writer = self.summary_writer
        runexample = iter(self.train_dataloader).next()
        print(len(runexample))

        self.model = self.model.to(device)
        summary_writer.add_graph(self.model, runexample[0].to(device))


        for epoch in range(epochs):
            # run epoch optimization
            avg_loss = self.train_epoch()

            # display
            acc_test = accuracy(self.model, self.eval_dataloader, device)
            # acc_train = accuracy(self.model, self.train_dataloader, device)
            print(f"loss[{epoch}] = {avg_loss / self.train_size}")
            print(f"acc[{epoch}] = {acc_test.item()}")
            loss_log.append(avg_loss / self.train_size)
            summary_writer.add_scalar("loss_train",avg_loss, global_step=epoch)
            summary_writer.add_scalar("accuracy_test",acc_test.item(), global_step=epoch)
            # summary_writer.add_scalar("accuracy_train",acc_train.item())
            acc_log.append(acc_test)
            avg_loss = 0

            # save model
            if epoch > 0 and epoch % self.save_checkpoint == 0:
                torch.save(model.state_dict(), os.path.join(run_out_path, f"resnet_{run_name}_{epoch}.pt"))

            # acc_log.append(accuracy(model,eval_dataloader))

        return loss_log, acc_log

    def train_epoch(self):
        avg_loss = 0
        for x, y in tqdm(self.train_dataloader):
            self.model.train()
            self.optim.zero_grad()
            x, y = x.to(device), y.to(device)
            x_lat, y_hat = self.model(x)
            loss = self.criteria(y_hat, y)
            avg_loss += loss.item()
            loss.backward()
            optim.step()
        return avg_loss / len(self.train_dataloader)

if __name__ == '__main__':

    use_gpu = True
    learning_rate = 0.001
    # train_path = 'D:\\Alon_temp\\singlang\\singLang_DLProg\\images\\train'
    train_path = 'D:\\Alon_temp\\singlang\\singLang_DLProg\\images\\coloredCaptureData\\train'
    test_path = 'D:\\Alon_temp\\singlang\\singLang_DLProg\\images\\coloredCaptureData\\test'
    # # debug
    # train_path = 'D:\\Alon_temp\\singlang\\singLang_DLProg\\images\\coloredCaptureData_debug'
    # test_path = 'D:\\Alon_temp\\singlang\\singLang_DLProg\\images\\coloredCaptureData_debug'
    out_path = 'D:\\Alon_temp\\singlang\\singLang_DLProg\\out_puts'
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    train_dl = get_image_dataloader(train_path, batch_size=128)
    print(train_dl.dataset.classes)
    test_dl = get_image_dataloader(test_path, batch_size=128)
    # model = load_resnet_model('')
    model = get_pretrained_resnet()
    model = model.to(device)
    optim = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    # optim = torch.optim.Adagrad(params=model.parameters(), lr=learning_rate)
    summary_writer = SummaryWriter(comment="pretrained")
    run_name = "with_aug_colored_pretrained_test_run_64_trainer"
    try:
        trainer = Trainer( train_dataloader=train_dl, eval_dataloader=test_dl, model=model, optim=optim,summary_writer=summary_writer, save_checkpoint=1)
        loss_log, acc_log = trainer.train(run_name,  out_path=out_path, epochs=45, device=device)

    except KeyboardInterrupt:
        pass

    with open('results_olg.txt', 'w') as f:
        f.write("loss:\n")
        for item in loss_log:
            f.write("%s\n" % item)
        f.write("acc:\n")
        for item in acc_log:
            f.write("%s\n" % item)


    plt.plot(loss_log)
    plt.show()
    torch.save(model.state_dict(), os.path.join(out_path, f"final_resnet_{run_name}.pt"))
