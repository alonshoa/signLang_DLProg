import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
from Code.data.dataloaders import get_image_dataloader
from Code.models.resnet import SingLangResNet
from Code.trianer.evaluate import accuracy
from Code.utils.helpers import load_resnet_model, get_pretrained_resnet
import matplotlib.pyplot as plt

def train_resnet(run_name,train_dataloader,eval_dataloader, model, optim,criteria=torch.nn.CrossEntropyLoss().cuda(), epochs=100,
                 display_every=200, save_checkpoint=200, device='cpu',out_path='./',    loss_log = [], acc_log = []):

    step = 0
    model = model.to(device)
    avg_loss = 0
    for i in range(epochs):
        for x,y in tqdm(train_dataloader):
            model.train()
            optim.zero_grad()

            x,y = x.to(device),y.to(device)
            x_lat, y_hat = model(x)
            loss = criteria(y_hat,y)

            avg_loss += loss.item()
            loss.backward()
            optim.step()

            if step > 0 and step % display_every == 0:
                acc = accuracy(model,eval_dataloader,device)
                # print(acc)
                print(f"loss[{step}] = {avg_loss/display_every}")
                print(f"acc[{step}] = {acc.item()}")
                loss_log.append(avg_loss/display_every)
                acc_log.append(acc)
                avg_loss = 0

            if step > 0 and step % save_checkpoint == 0:
                torch.save(model.state_dict(), os.path.join(out_path, f"resnet_{run_name}_{step}.pt"))

            step += 1

        # acc_log.append(accuracy(model,eval_dataloader))

    return loss_log, acc_log


if __name__ == '__main__':
    loss_log = []
    acc_log = []
    use_gpu = True
    learning_rate = 0.001
    # train_path = 'D:\\Alon_temp\\singlang\\singLang_DLProg\\images\\train'
    train_path = 'D:\\Alon_temp\\singlang\\singLang_DLProg\\images\\coloredCaptureData\\train'
    test_path = 'D:\\Alon_temp\\singlang\\singLang_DLProg\\images\\coloredCaptureData\\test'
    out_path = 'D:\\Alon_temp\\singlang\\singLang_DLProg\\out_puts'
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    train_dl = get_image_dataloader(train_path, batch_size=128)
    print(train_dl.dataset.classes)
    test_dl = get_image_dataloader(test_path, batch_size=128)
    model = load_resnet_model('')
    # model = get_pretrained_resnet(train_dl)
    model = model.to(device)
    # optim = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    optim = torch.optim.Adagrad(params=model.parameters(), lr=learning_rate)
    run_name = "with_aug_colored_pretrained_test_run_64"
    try:
        loss_log, acc_log = train_resnet(run_name, train_dl, test_dl, model, optim, out_path=out_path, epochs=100, display_every=500,
                     save_checkpoint=2500, device=device,loss_log=loss_log,acc_log=acc_log)

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
