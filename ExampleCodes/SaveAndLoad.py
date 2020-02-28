#Save model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

#load model
net = Net()
net.load_state_dict(torch.load(PATH))