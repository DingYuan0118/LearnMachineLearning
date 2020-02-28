#Save model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

#load model
net = Net()
net.load_state_dict(torch.load(PATH))

'''
When you call torch.load() on a file which contains GPU tensors, 
those tensors will be loaded to GPU by default. You can call 
torch.load(.., map_location='cpu') and then load_state_dict() 
to avoid GPU RAM surge when loading a model checkpoint.
'''