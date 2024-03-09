import torch
import torch.nn.functional as F

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device(dev)


def causal_padding(x, dilation):
    """
    Apply causal padding to the input tensor.

    Args:
        x (Tensor): Input tensor.
        dilation (int): Dilation factor.

    Returns:
        Tensor: Padded tensor.
    """
    padding = (dilation * (x.size(-1) - 1), 0)
    x = F.pad(x, padding, mode='constant', value=0)
    return x


class CausalConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        """
        Causal 1D Convolutional layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            dilation (int): Dilation factor (default: 1).
        """
        super(CausalConv2d, self).__init__()
        
        self.padding = (0,0,(kernel_size[0] - 1) * dilation, (kernel_size[1]-1) // 2) #(0,0,causal_padding,same_padding)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, dilation=dilation).to(device)

    def forward(self, x):
        """
        Forward pass of the CausalConv1d layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        x = F.pad(x,self.padding, mode='constant', value=0)
        x = self.conv(x)
        return x


class AddLayer(torch.nn.Module):
    def __init__(self):
        super(AddLayer, self).__init__()

    def forward(self, prev_x, x):
        """
        Forward pass of the AddLayer.

        Args:
            prev_x (Tensor): Previous tensor.
            x (Tensor): Current tensor.

        Returns:
            Tensor: Sum of the previous and current tensors.
        """
        return prev_x + x


class TCN(torch.nn.Module):
    def __init__(self,
                 nb_filters=64,
                 kernel_size=2,
                 nb_stacks=1,
                 dilations=[1, 2, 4, 8, 16, 32],
                 use_skip_connections=True,
                 output_size = 2,
                 drop_rate=0.0,
                 return_sequences=False,
                 **kwargs
                     ):
        """
        Temporal Convolutional Network (TCN) module.

        Args:
            nb_filters (int): Number of filters in the TCN layers.
            kernel_size (int): Size of the convolutional kernels.
            nb_stacks (int): Number of residual blocks stacks.
            dilations (list): List of dilation factors.
            use_skip_connections (bool): Whether to use skip connections (default: True).
            dropout_rate (float): Dropout rate (default: 0.0).
            return_sequences (bool): Whether to return output sequences or last timestep output (default: False).
        """
        super(TCN, self).__init__()
        self.return_sequences = return_sequences
        self.dropout_rate = drop_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.output_size = output_size
        self.nb_filters = nb_filters
        self.channel_depth = kwargs.get("channel_depth")
        self.conv1_x = CausalConv2d(self.channel_depth, nb_filters, kernel_size=(1,1), dilation=1)

        self.conv2_x = torch.nn.ModuleList()
        self.maxpool = torch.nn.ModuleList()
        self.residual_blocks = torch.nn.ModuleList()
        
        for s in range(self.nb_stacks):
            for d in self.dilations:
                self.residual_blocks.append(torch.nn.Sequential(
                    CausalConv2d(in_channels=nb_filters, out_channels=nb_filters, kernel_size=self.kernel_size,
                                 dilation=d),
                    torch.nn.BatchNorm2d(nb_filters),
                    torch.nn.ReLU(),
                    torch.nn.Dropout2d(p=drop_rate),
                    CausalConv2d(in_channels=nb_filters, out_channels=nb_filters, kernel_size=self.kernel_size,
                                 dilation=d),
                    torch.nn.BatchNorm2d(nb_filters),
                    torch.nn.ReLU(),
                    torch.nn.Dropout2d(p=drop_rate)
                ))
                self.conv2_x.append(torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=nb_filters, out_channels=nb_filters, kernel_size=(1,1), padding='same')
                ))
                self.maxpool.append(torch.nn.Sequential(
                    torch.nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
                ))

        if not self.return_sequences:
            self.final_layer = torch.nn.LazyLinear(self.output_size)
            

    def forward(self, inputs,data,**kwargs):
        """
        Forward pass of the TCN.

        Args:
            inputs (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """

        x = inputs.float().to(device)
        x = self.conv1_x(x)
        
        
        skip_connections = []
        for i, (residual_block, conv2, maxpool) in enumerate(zip(self.residual_blocks, self.conv2_x, self.maxpool)):
            prev_x = x
            x = residual_block(x)            
            prev_x = conv2(prev_x)
            
            if kwargs.get("freq_flag"):
                if i not in [0, 1, 2]:
                    skip_connections.append(x)
            else:
                skip_connections.append(x)
                
            x = prev_x + x
            if kwargs.get("freq_flag"):
                if i == 0 or i == 1 or i == 2:
                    x = maxpool(x)

        if self.use_skip_connections:
            stacked = torch.stack(skip_connections)
            x = torch.sum(stacked, dim=0)

        if not self.return_sequences:
            x = x[:, :, -1,:]  # Select only the last timestep
            x = x.flatten(start_dim=1)
            x = torch.cat((x, data),dim=1).float()
            x = self.final_layer(x)

        return x
    
class EncoderTCN(torch.nn.Module):
    def __init__(self,
                 nb_filters=64,
                 kernel_size=2,
                 nb_stacks=1,
                 dilations=[1, 2, 4, 8, 16, 32],
                 use_skip_connections=True,
                 output_size = 2,
                 drop_rate=0.0,
                 return_sequences=False
                     ):
        """
        Temporal Convolutional Network (TCN) module.

        Args:
            nb_filters (int): Number of filters in the TCN layers.
            kernel_size (int): Size of the convolutional kernels.
            nb_stacks (int): Number of residual blocks stacks.
            dilations (list): List of dilation factors.
            use_skip_connections (bool): Whether to use skip connections (default: True).
            dropout_rate (float): Dropout rate (default: 0.0).
            return_sequences (bool): Whether to return output sequences or last timestep output (default: False).
        """
        super(EncoderTCN, self).__init__()
        self.return_sequences = return_sequences
        self.dropout_rate = drop_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.output_size = output_size
        self.nb_filters = nb_filters
        
        self.conv1_x = CausalConv2d(3, nb_filters, kernel_size=(1,1), dilation=1)

        self.conv2_x = torch.nn.ModuleList()
        self.maxpool = torch.nn.ModuleList()
        self.residual_blocks = torch.nn.ModuleList()
        
        for s in range(self.nb_stacks):
            for d in self.dilations:
                self.residual_blocks.append(torch.nn.Sequential(
                    CausalConv2d(in_channels=nb_filters, out_channels=nb_filters, kernel_size=self.kernel_size,
                                 dilation=d),
                    torch.nn.BatchNorm2d(nb_filters),
                    torch.nn.ReLU(),
                    torch.nn.Dropout2d(p=drop_rate),
                    CausalConv2d(in_channels=nb_filters, out_channels=nb_filters, kernel_size=self.kernel_size,
                                 dilation=d),
                    torch.nn.BatchNorm2d(nb_filters),
                    torch.nn.ReLU(),
                    torch.nn.Dropout2d(p=drop_rate)
                ))
                self.conv2_x.append(torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=nb_filters, out_channels=nb_filters, kernel_size=(1,1), padding='same')
                ))
                self.maxpool.append(torch.nn.Sequential(
                    torch.nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
                ))
                
    def forward(self, inputs,data,**kwargs):
        """
        Forward pass of the TCN.

        Args:
            inputs (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """

        x = inputs.float().to(device)
        x = self.conv1_x(x)
        
        
        skip_connections = []
        for i, (residual_block, conv2, maxpool) in enumerate(zip(self.residual_blocks, self.conv2_x, self.maxpool)):
            prev_x = x
            x = residual_block(x)            
            prev_x = conv2(prev_x)
            
            if kwargs.get("freq_flag"):
                if i not in [0, 1, 2]:
                    skip_connections.append(x)
            else:
                skip_connections.append(x)
                
            x = prev_x + x
            if kwargs.get("freq_flag"):
                if i == 0 or i == 1 or i == 2:
                    x = maxpool(x)

        return x,skip_connections

class DecoderTCN(torch.nn.Module):
    def __init__(self,
                 nb_filters=64,
                 kernel_size=2,
                 nb_stacks=1,
                 dilations=[1, 2, 4, 8, 16, 32],
                 use_skip_connections=True,
                 output_size = 2,
                 drop_rate=0.0,
                 return_sequences=False
                     ):
        super(DecoderTCN, self).__init__()
        self.return_sequences = return_sequences
        self.output_size = output_size
        self.use_skip_connections = use_skip_connections
        
        if not self.return_sequences:
            self.final_layer = torch.nn.LazyLinear(output_size)

    def forward(self, x,data,skip_connections,**kwargs):
        """
        Forward pass of the TCN.

        Args:
            inputs (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
       

        if self.use_skip_connections:
            stacked = torch.stack(skip_connections)
            x = torch.sum(stacked, dim=0)
        
        if not self.return_sequences:
            x = x[:, :, -1,:]  # Select only the last timestep
            x = x.flatten(start_dim=1)
            x = torch.cat((x, data),dim=1).float()
            x = self.final_layer(x)

        return x
    
class TransferTCN(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(TransferTCN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x,stat_info,**kwargs):
        with torch.no_grad():
            encoder_output,skip_connections = self.encoder(x,stat_info,**kwargs)
        x = self.decoder(encoder_output,stat_info,skip_connections,**kwargs)
        return x 