import torch
import torch.nn as nn

class Prompter(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1,
                 **kwargs, ):
        super().__init__()
        self.mid_d = 16
        self.cnn_in = cnn_in = self.mid_d
        self.pool_in = pool_in = self.mid_d
        self.cnn_dim = cnn_dim = cnn_in * 2
        self.pool_dim = pool_dim = pool_in * 2

        self.conv_in = nn.Conv2d(768, self.mid_d * 2, kernel_size=1, stride=1, padding=0)

        self.conv1 = nn.Conv2d(self.mid_d , self.mid_d , kernel_size=1, stride=1, padding=0)
        self.proj1 = nn.Conv2d(self.mid_d , self.mid_d , kernel_size=kernel_size, stride=stride, padding=padding)

        self.mid_gelu1 = nn.GELU()

        self.Maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        self.proj2 = nn.Conv2d(self.mid_d , self.mid_d , kernel_size=1, stride=1, padding=0)
        self.mid_gelu2 = nn.GELU()

        self.conv_fuse = nn.Conv2d(self.mid_d * 2, self.mid_d * 2, kernel_size=1, stride=1, padding=0,)
        self.proj = nn.Conv2d(self.mid_d * 2, 768, kernel_size=1, stride=1, padding=0)

        #self.beta = nn.Parameter(1e-6 * torch.ones(768),requires_grad=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def forward(self, input):
        # B, C H, W
        x = self.conv_in(input)

        cx = x[:, :self.cnn_in, :, :].contiguous()
        cx = self.conv1(cx)
        cx = self.proj1(cx)
        cx = self.mid_gelu1(cx)

        px = x[:, self.cnn_in:, :, :].contiguous()
        px = self.Maxpool(px)
        px = self.proj2(px)
        px = self.mid_gelu2(px)

        hx = torch.cat((cx, px), dim=1)

        hx = self.conv_fuse(hx)

        hx = self.proj(hx)
        return hx #* self.beta.view(1, -1, 1, 1)


