import torch
import pickle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(net, X_test, y_test):
    net.eval()

    hidden = net.init_hidden(1)
    log_probs = []
    for idx, (x, y) in enumerate(zip(X_test, y_test)):
        print('\rtest:{}'.format(idx), flush=True, end='')

        hidden = net.repackage_hidden(hidden)

        x = torch.tensor(x).unsqueeze(0).to(device)

        output, hidden = net(x, hidden)

        log_probs.append(output.item())

    print('dumping log_probs')
    model_file = open('log_probs.pkl', 'wb')
    pickle.dump(log_probs, model_file)

    return log_probs
