import torch

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
    
        ####################################
        #          YOUR CODE HERE          #
        ####################################

        # - convolutional layer with in_channels=3, out_channels=128, kernel=4, stride=2
        # - convolutional layer with in_channels=128, out_channels=256, kernel=4, stride=2
        # - batch norm
        # - convolutional layer with in_channels=256, out_channels=512, kernel=4, stride=2
        # - batch norm
        # - convolutional layer with in_channels=512, out_channels=1024, kernel=4, stride=2
        # - batch norm
        # - convolutional layer with in_channels=1024, out_channels=1, kernel=4, stride=1

        self.d_model = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=1),
        )

        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        x = self.d_model(x)
        
        ##########       END      ##########
        
        return x


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()    
        self.noise_dim = noise_dim
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################

        # - transpose convolution with in_channels=NOISE_DIM, out_channels=1024, kernel=4, stride=1
        # - batch norm
        # - transpose convolution with in_channels=1024, out_channels=512, kernel=4, stride=2
        # - batch norm
        # - transpose convolution with in_channels=512, out_channels=256, kernel=4, stride=2
        # - batch norm
        # - transpose convolution with in_channels=256, out_channels=128, kernel=4, stride=2
        # - batch norm
        # - transpose convolution with in_channels=128, out_channels=3, kernel=4, stride=2
        print(f"noise_dim = {noise_dim}, output_channels = {output_channels}")
        # Campuswire #720
        self.g_model = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(self.noise_dim, 1024, kernel_size=4, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True),

            torch.nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1),
            torch.nn.Tanh()
        )
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        x = self.g_model(x)
        
        ##########       END      ##########
        
        return x
    

