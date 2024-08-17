import torch

def Accuracy(data, target_severities):
    tensordata = torch.argmax(data,dim=3)
    one_hot_encoded = torch.zeros(tensordata.size(0), tensordata.size(1), tensordata.size(2), 3)
    for i in range(tensordata.size(0)):
        for j in range(tensordata.size(1)):
            for k in range(tensordata.size(2)):
                value = tensordata[i, j, k].int()
                one_hot_encoded[i, j, k, value] = 1


    predicted_severities = torch.flatten(one_hot_encoded, start_dim=1)

    hits = 0
    fails = 0
    for i in range(predicted_severities.size(0)):
        for j in range(predicted_severities.size(1)):
            if (predicted_severities[i,j]==target_severities[i,j].int()):
                hits = hits + 1
            else:
                fails = fails + 1

    #accuracy = hits/(hits+fails) * 100
    return hits


def AccuracyEfficient(data, target_severities):
    # Compute the predicted labels by taking the argmax along dimension 3
    predicted_labels = torch.argmax(data, dim=3)
    
    # Convert target_severities to integer type
    target_severities = target_severities.int()
    
    # Compute the accuracy by comparing predicted_labels and target_severities
    hits = (predicted_labels == target_severities).sum().item()
    total = target_severities.numel()
    accuracy = hits / total * 100
    
    return accuracy