import torch
from tqdm import tqdm
from Code.data.dataloaders import get_image_dataloader
from Code.utils.helpers import load_resnet_model
from sklearn.metrics import plot_confusion_matrix,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def accuracy(model, eval_data, device):
    model = model.to(device)
    with torch.no_grad():
        correct = 0
        all = 0
        for x, y in tqdm(eval_data):
            x, y = x.to(device), y.to(device)
            x_last, y_hat = model(x)
            correct += torch.sum(torch.argmax(y_hat, dim=1) == y)
            all += x.shape[0]
        return correct/all


def generate_confution_matrix(model,dataloader,device):
    model = model.to(device)
    with torch.no_grad():
        correct = 0
        all = 0
        y_all =[]
        y_pred_all =[]
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            x_last, y_hat = model(x)
            y_all.extend(torch.argmax(y_hat,dim=1).cpu().numpy())
            y_pred_all.extend(y.cpu().numpy())
        return confusion_matrix(y_all,y_pred_all)


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
    # model = load_resnet_model('D:\\Alon_temp\\singlang\\singLang_DLProg\\pretrained\\final_resnet_test_run_64.pt')
    model = load_resnet_model('D:\\Alon_temp\\singlang\\singLang_DLProg\\out_puts\\final_resnet_with_aug_colored_pretrained_test_run_64_trainer.pt')
    data = get_image_dataloader('D:\\Alon_temp\\singlang\\singLang_DLProg\\images\\coloredCaptureData\\test', 64)
    # ac = accuracy(model,data,device)
    # print("ac=",ac)

    v = generate_confution_matrix(model,data,device)
    print(v)

    # plot_confusion_matrix()
    # plt.hist2d(v)
    # plt.show()

    plt.figure(figsize=(16,16))
    lables = [c for c in 'אבגדהוזחטיכלמנסעפצקרשת ']
    sns.heatmap(v, cmap='Blues', xticklabels=lables, yticklabels=lables)
    plt.show()

