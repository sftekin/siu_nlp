import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(net, batch_gen):
    net.eval()

    hidden = net.init_hidden()
    total_correct = 0
    log_probs = []
    for idx, (x, y) in enumerate(batch_gen.generate('test')):
        print('\rtest:{}'.format(idx), flush=True, end='')

        hidden = net.repackage_hidden(hidden)

        x, y = x.to(device), y.to(device)

        output, hidden = net(x, hidden)

        log_probs.append(output)

        pred = torch.round(output)
        total_correct += np.sum(pred.eq(y).numpy())

    accuracy = total_correct / len(batch_gen.dataset_dict['test'])
    return log_probs, accuracy
