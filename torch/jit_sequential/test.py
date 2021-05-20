import torch

class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()

    def forward(self, input):
        return torch.tensor(1)

    @torch.jit.export
    def set_para(self):
        print('calling set_para(): tensor is ', torch.tensor(1))

module = torch.jit.script(MyModule())
sequential = torch.jit.script(torch.nn.Sequential(torch.nn.Identity(), MyModule()))
empty = torch.empty(0)

b = module(empty)
# set_para() is called from torchscript
module.set_para()
b = module(empty)

b = sequential(empty)
# set_para() is called from python, instead of torchscript
sequential[1].set_para()
b = sequential(empty)
