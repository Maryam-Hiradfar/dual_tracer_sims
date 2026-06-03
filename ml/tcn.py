import torch.nn as nn
import torch 

class Model(nn.Module):
    def __init__(self, in_channels = 1, hidden_channels = 16, out_channels = 2, kernel_size = 3): 
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size 
        padding = self.kernel_size // 2
        
        self.conv1 = nn.Conv1d(self.in_channels, self.hidden_channels,self.kernel_size, padding = padding)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, padding = padding)
        self.conv3 = nn.Conv1d(self.hidden_channels, self.out_channels, kernel_size=self.kernel_size, padding=padding)

    def forward(self, x): 
        out1 = self.relu(self.conv1(x))
        out2 = self.relu(self.conv2(out1))
        out3 = self.conv3(out2)
        return out3 
    #---------------------
def main(): 
    model = Model()
    x = torch.randn(4,1,100)
    yhat = model(x)
    print(yhat.shape)
        
if __name__ == "__main__":
        main()
      

