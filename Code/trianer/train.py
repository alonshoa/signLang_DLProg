import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
from singLang_DLProg.Code.data.dataloaders import get_dataloader
from singLang_DLProg.Code.models.resnet import SingLangResNet


def train_resnet(run_name,train_dataloader, model, optim,criteria=torch.nn.BCELoss(), epochs=100,
                 display_every=200, save_checkpoint=200, device='cpu',out_path='./'):
    loss_log = []
    step = 0
    model = model.to(device)
    avg_loss = 0
    for i in range(epochs):
        for x,y in tqdm(train_dataloader):
            model.train()
            optim.zero_grad()

            x,y = x.to(device),y.to(device)
            x_lat, y_hat = model(x)
            print(y_hat.device)
            print(y)
            loss = F.binary_cross_entropy_with_logits(y_hat,y)
            # loss = criteria(y_hat,y)
            avg_loss += loss.item()
            loss.backward()
            optim.step()

            if step > 0 and step % display_every == 0:
                print(f"loss[{step}] = {avg_loss/display_every}")
                loss_log.append(avg_loss/display_every)
                avg_loss = 0

            if step > 0 and step % save_checkpoint == 0:
                torch.save(model.state_dict(), os.path.join(out_path, f"resnet_{run_name}.pt"))


if __name__ == '__main__':
    use_gpu = True
    learning_rate = 0.001
    test_path = 'D:\\Alon_temp\\singlang\\singLang_DLProg\\images\\debug'
    out_path = 'D:\\Alon_temp\\singlang\\singLang_DLProg\\out_puts'
    dl = get_dataloader(test_path)
    # print(len(dl.classes))
    model = SingLangResNet()
    optim = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    train_resnet("test_run",dl,model,optim,out_path=out_path,epochs=2,display_every=1,save_checkpoint=1,device=device)

    # input = torch.randn(3, requires_grad=True)
    # print(input)
    # target = torch.empty(3).random_(2)
    # print(target)
    # loss = F.binary_cross_entropy_with_logits(input, target)
    # loss.backward()
