import torch.nn as nn

class PseudoMixEncoder(nn.Module):
    def __init__(self, chnum_in, n_frames):
        super(PseudoMixEncoder, self).__init__()

        self.chnum_in = chnum_in  # color channels
        feature_num = 128
        feature_num_2 = 96
        feature_num_x2 = 256
        self.encoder = nn.Sequential(
            nn.Conv2d(self.chnum_in * n_frames, feature_num_2, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num_2, feature_num, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num, feature_num_x2, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num_x2, feature_num_x2, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class PseudoMixDecoder(nn.Module):
    def __init__(self, chnum_in, n_frames, double_input=False):
        super(PseudoMixDecoder, self).__init__()

        # Dong Gong's paper code + Tanh
        self.chnum_in = chnum_in
        feature_num = 128
        feature_num_2 = 96
        feature_num_x2 = 256
        inchannel = feature_num_x2 * 2 if double_input else feature_num_x2
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(inchannel, feature_num_x2, (3, 3), stride=(2, 2), padding=(1, 1),
                               output_padding=(1, 1)),
            nn.BatchNorm2d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(feature_num_x2, feature_num, (3, 3), stride=(2, 2), padding=(1, 1),
                               output_padding=(1, 1)),
            nn.BatchNorm2d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(feature_num, feature_num_2, (3, 3), stride=(2, 2), padding=(1, 1),
                               output_padding=(1, 1)),
            nn.BatchNorm2d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(feature_num_2, self.chnum_in * n_frames, (3, 3), stride=(2, 2), padding=(1, 1),
                               output_padding=(1, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class PseudoMixDiscriminator(nn.Module):
    def __init__(self, chnum_in, n_frames):
        super(PseudoMixDiscriminator, self).__init__()
        self.chnum_in = chnum_in
        feature_num = 128
        feature_num_2 = 96
        feature_num_x2 = 256
        self.discriminator = nn.Sequential(
            nn.Conv2d(self.chnum_in * n_frames, feature_num_2, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num_2, feature_num, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num, feature_num_x2, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num_x2, feature_num_x2, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num_x2, feature_num_x2, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num_x2, feature_num_x2, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            Flatten(),
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.discriminator(x)
        return x

# import torch
# inp = torch.ones([4, 16, 256, 256])
# net = PseudoMixDiscriminator(1, 16)
# out = net(inp)
# pass
