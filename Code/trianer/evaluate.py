import torch
from tqdm import tqdm
from Code.data.dataloaders import get_dataloader
from Code.utils.helpers import load_resnet_model


def accuracy(model,eval_data,device):
    model = model.to(device)
    with torch.no_grad():
        correct = 0
        all = 0
        for x,y in tqdm(eval_data):
            x,y = x.to(device),y.to(device)
            x_last, y_hat = model(x)
            # print(torch.argmax(y_hat,dim=1))
            # print(y)
            # break
            correct += torch.sum(torch.argmax(y_hat,dim=1)==y)
            all += x.shape[0]
        return correct/all

# set to evaluation mode
# def evaluate(autoencoder):
#     autoencoder.to(device)
#     autoencoder.eval()
#
#     test_loss_avg, num_batches = 0, 0
#     for image_batch, _ in test_dataloader:
#         with torch.no_grad():
#             image_batch = image_batch.to(device)
#
#             # autoencoder reconstruction
#             image_batch_recon = autoencoder(image_batch)
#             # if (image_batch_recon < 0).any():
#             #   print("yes")
#             # reconstruction error
#             loss = F.mse_loss(image_batch_recon[0], image_batch)
#
#             test_loss_avg += loss.item()
#             num_batches += 1
#
#     test_loss_avg /= num_batches
#     print('average reconstruction error: %f' % (test_loss_avg))
#     return test_loss_avg

if __name__ == '__main__':
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    model = load_resnet_model('D:\\Alon_temp\\singlang\\singLang_DLProg\\pretrained\\final_resnet_test_run_64.pt')
    data = get_dataloader('D:\\Alon_temp\\singlang\\singLang_DLProg\\images\\test',64)
    ac = accuracy(model,data,device)
    print("ac=",ac)

