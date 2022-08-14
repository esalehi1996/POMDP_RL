# __author__ = 'SherlockLiao'

import os
import time
import argparse

import torch
import torchvision
from torch import nn
# from torch.autograd import Variable
from torch.utils.data import DataLoader
# from torchvision import transforms
from torchvision.utils import save_image

# if not os.path.exists('mlp_img'):
#     os.mkdir('mlp_img')

def unNormalize(data, mean, scale_normalizer, view_row=3, view_col=3):
    # print(mean, std)
    # scale_normalizer can be a multiple of the std dev or the maximum channel value
    uN_data = data.clone().reshape(-1, 3, view_row, view_col)
    for i in range(3):
        uN_data[:, i, :, :] = (uN_data[:, i, :, :]*scale_normalizer[i] + mean[i])*255.0
    return uN_data


class autoencoder(nn.Module):
    def __init__(self, big_obs_grid=False):
        super(autoencoder, self).__init__()
        if not big_obs_grid:
            #For 3x3 observation grid
            self.latent_space_size = 16
            self.encoder = nn.Sequential(
                nn.Linear(27, 25),
                nn.ReLU(),
                nn.Linear(25, self.latent_space_size))
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_space_size, 25),
                nn.ReLU(),
                nn.Linear(25, 27),
                nn.Tanh())
        else:
            #For 7x7 observation grid
            self.latent_space_size = 64
            self.encoder = nn.Sequential(
                nn.Linear(147, 96),
                nn.ReLU(),
                nn.Linear(96, self.latent_space_size))
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_space_size, 96),
                nn.ReLU(),
                nn.Linear(96, 147),
                nn.Tanh())

    def forward(self, x, getLatent=False):
        x = self.encoder(x)
        # print (x.shape)
        if not getLatent:
            x = self.decoder(x)
        return x

if __name__ == "__main__":
    from minigrid_datasets import ObsGrids7x7

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch Size over dataset",
        default=256
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of Epochs to run over dataset",
        default=100
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning Rate of the Autoencoder",
        default=1e-4
    )
    parser.add_argument(
        "--starting_model_path",
        help="path for model to start with"
    )
    parser.add_argument(
        "--path",
        help="folder to save data in",
        default='models/MGER6x6/ExpName_LatentSize16_withLargeStd_withLowLR_WD'
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="GPU selection",
        default=True
    )

    args = parser.parse_args()
    
    use_cuda = torch.cuda.is_available() and args.use_gpu
    device = torch.device("cuda" if use_cuda else "cpu")
    
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate


    big_obs_grid = True

    dataset = ObsGrids7x7(args.path)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = autoencoder(big_obs_grid).to(device)
    if args.starting_model_path is not None:
        model.load_state_dict(torch.load(args.starting_model_path))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)

    start_time = time.time()
    for epoch in range(num_epochs):
        # print ('Now Running Epoch: [{}/{}]'.format(epoch, num_epochs))
        losses = []
        for data in dataloader:
            data = data.to(device)
    #         exit()
            # print (data)
            # print (data.shape)
    #         exit()
    #         print (output[0,:])
    #         output1 = model(data[0, :])
    #         print (output1)
            # unNormalized_data = unNormalize(data, dataset.mean, dataset.std)
            # print('---------------------')
            # for i in unNormalized_data:
            #     print (i)
            # exit()
    #         img = data
    #         img = img.view(img.size(0), -1)
    #         # img = Variable(img).cuda()
    #         # ===================forward=====================
    #         output = model(img)
            output = model(data)
            # print (output)
            # print (output.shape)
            loss = criterion(output, data)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().item())
        # ===================log========================
        print('Epoch [{}/{}], Loss:{:.10f}'.format(epoch + 1, num_epochs, torch.Tensor(losses).mean().item()))
        if epoch % 5 == 0:
            # pic = to_img(output.cpu().data)
            # save_image(pic, './mlp_img/image_{}.png'.format(epoch))
            torch.save(model.state_dict(), os.path.join(args.path, 'autoencoder_final.pth'))
    
    total_time = time.time() - start_time
    print ('Total Time Taken: ', total_time)
    
    torch.save(model.state_dict(), os.path.join(args.path, 'autoencoder_final.pth'))