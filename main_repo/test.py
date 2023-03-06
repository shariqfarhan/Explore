import torch
import torch.nn as nn
import torch.nn.functional as F

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()


    test_loss /= len(test_loader.dataset)
#     test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
#     test_acc.append(100. * correct / len(test_loader.dataset))

def mis_classified_images(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    incorr_X, incorr_y, incorr_argmax = list(), list(), list()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            # The below code is added to identify misclassified images

            for idx, i in enumerate(output):
                if torch.argmax(i) == target[idx]:
                    correct += 0 
                # To avoid double counting we don't add 1 here. If the above line of code ``` correct += pred.eq(target.view_as(pred)).sum().item() ```
                # wasn't there, we would add 1 here
                else:
                    incorr_X.append(data)
                    incorr_y.append(target)
                    incorr_argmax.append(torch.argmax(i))

    test_loss /= len(test_loader.dataset)
#     test_losses.append(test_loss)

    return incorr_X, incorr_y, incorr_argmax
