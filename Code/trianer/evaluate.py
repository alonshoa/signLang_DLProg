import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from Code.data.dataloaders import get_image_dataloader
from Code.data.heb_data import create_street_names_data_iterators
from Code.data.hybrid_data import HybridDataSet
from Code.models.heb_model import HebLetterToSentence
from Code.models.hybrid_model import HybridModel
from Code.utils.helpers import load_resnet_model, load_model
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


def full_model_accuracy(model,eval_data,word_vocab,device):
    model = model.to(device)
    with torch.no_grad():
        correct = 0
        all = 0
        for x, y in tqdm(eval_data):
            x, y = x.to(device), y.to(device)
            y_hat, ht = model(x)
            print(y_hat)
            argmax = torch.argmax(y_hat.squeeze(1), dim=1)
            print(argmax)
            if (all > 100):
                exit(1)
            correct += torch.sum(argmax == y)
            all += x.shape[0]
        return correct / all


def generate_confusion_matrix(model, dataloader, device):
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
    pass
    # eval image data only
    ########################3
    # use_gpu = True
    # device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    # # model = load_resnet_model('D:\\Alon_temp\\singlang\\singLang_DLProg\\pretrained\\final_resnet_test_run_64.pt')
    # model = load_resnet_model('')
    # data = get_image_dataloader('D:\\Alon_temp\\singlang\\singLang_DLProg\\images\\coloredCaptureData\\test', 64)
    # # ac = accuracy(model,data,device)
    # # print("ac=",ac)
    #
    # v = generate_confution_matrix(model,data,device)
    # print(v)
    #
    # # plot_confusion_matrix()
    # # plt.hist2d(v)
    # # plt.show()
    #
    # plt.figure(figsize=(16,16))
    # lables = [c for c in 'אבגדהוזחטיכלמנסעפצקרשת ']
    # sns.heatmap(v, cmap='Blues', xticklabels=lables, yticklabels=lables)
    # plt.show()

    # eval full train
    ######################
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    # model = load_resnet_model('D:\\Alon_temp\\singlang\\singLang_DLProg\\pretrained\\final_resnet_test_run_64.pt')
    image_model = load_resnet_model('')
    _,image_data = get_image_dataloader('D:\\Alon_temp\\singlang\\singLang_DLProg\\images\\coloredCaptureData\\test')
    path = "D:\\Alon_temp\\singlang\\singLang_DLProg\\text_data\\split"

    train_iterator, test_iterator,vocab_letters,vocab_words,train_data,test_data  = create_street_names_data_iterators(path)


    hybridDataSet = HybridDataSet(image_data,test_data,vocab_words)
    dataloader = DataLoader(dataset=hybridDataSet, batch_size=4, shuffle=True)
    # someItem = hybridDataSet.__getitem__(2)
    # print(someItem[0].shape)
    # print(hybridDataSet.word_max_len)
    # def __init__(self, vocab_size,embedding_dim, lstm_size,hidden_dim, output_dim):

    text_model = HebLetterToSentence(len(vocab_letters), 128, 512, 128,len(vocab_words),use_self_emmbed=True)
    # image_model = load_resnet_model('D:\\Alon_temp\\singlang\\singLang_DLProg\\out_puts\\final_resnet_with_aug_colored_pretrained_test_run_64_trainer.pt')

    # image_model = set_grads_to_false(image_model)
    # exit(12)
    model = HybridModel(image_model,text_model).to(device)
    # model_to_eval = "D:\\Alon_temp\\singlang\\singLang_DLProg\\Code\\trianer\\full_run_test\\resnet_full_run_test_50.pt"
    model_names = [
        'D:\\Alon_temp\\singlang\\singLang_DLProg\\Code\\trianer\\full_run_test\\resnet_full_run_test_50.pt',
        'D:\\Alon_temp\\singlang\\singLang_DLProg\\Code\\trianer\\full_run_test_no_image_update\\resnet_full_run_test_no_image_update_45.pt',
        'D:\\Alon_temp\\singlang\\singLang_DLProg\\Code\\trianer\\full_run_test_pretrained_resnet\\resnet_full_run_test_50.pt'
    ]
    run_names = ["full_run","full_run_no_update","full_run_pretrained"]
    for model_to_eval,name in zip(model_names,run_names):
        model = load_model(model,model_to_eval)
        ac = full_model_accuracy(model,dataloader,vocab_words,device)
        print(f"ac[{name}]=",ac)
    #
    # v = generate_confution_matrix(model,data,device)
    # print(v)
    #
    # # plot_confusion_matrix()
    # # plt.hist2d(v)
    # # plt.show()
    #
    # plt.figure(figsize=(16,16))
    # lables = [c for c in 'אבגדהוזחטיכלמנסעפצקרשת ']
    # sns.heatmap(v, cmap='Blues', xticklabels=lables, yticklabels=lables)
    # plt.show()

