import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_deep(net, batch_gen, **kwargs):
    net.train()
    net.to(device)

    opt = optim.Adam(net.parameters(), lr=kwargs['lr'])
    batch_size = batch_gen.batch_size
    criterion = nn.BCELoss()

    train_loss_list = []
    val_loss_list = []
    for epoch in range(kwargs['n_epoch']):
        running_loss = 0
        hidden = net.init_hidden(batch_size)
        for idx, (x, y) in enumerate(batch_gen.generate('train')):

            print('\rtrain:{}'.format(idx), flush=True, end='')

            hidden = net.repackage_hidden(hidden)

            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            output, hidden = net(x, hidden)

            loss = criterion(output, y.float())
            loss.backward()

            nn.utils.clip_grad_norm_(net.parameters(), kwargs['clip'])
            opt.step()

            running_loss += loss.item()

            if (idx + 1) % kwargs['eval_every'] == 0:
                print('\n')
                val_loss, acc = evaluate(net, batch_gen)
                print("\nEpoch: {}/{}...".format(epoch + 1, kwargs['n_epoch']),
                      "Step: {}...".format(idx),
                      "Loss: {:.4f}...".format(running_loss / idx),
                      "Val Loss: {:.4f}".format(val_loss),
                      'Val Acc: {:.4f}'.format(acc))

        train_loss_list.append(running_loss / idx)
        val_loss_list.append(val_loss)

        loss_file = open('losses.pkl', 'wb')
        model_file = open('bilstm_model.pkl', 'wb')
        pickle.dump([train_loss_list, val_loss_list], loss_file)
        pickle.dump(net, model_file)

    print('Training finished, saving the model')
    model_file = open('bilstm_model.pkl', 'wb')
    pickle.dump(net, model_file)

    # plot losses
    plt.figure()
    plt.plot(range(kwargs['n_epoch']), train_loss_list, label='train')
    plt.plot(range(kwargs['n_epoch']), val_loss_list, label='validation')
    plt.legend()
    plt.show()


def evaluate(net, batch_gen):
    net.eval()

    criterion = nn.BCELoss()
    batch_size = batch_gen.batch_size

    val_losses = []
    hidden = net.init_hidden(batch_size)
    total_correct = 0
    for idx, (x, y) in enumerate(batch_gen.generate('validation')):

        print('\rval:{}'.format(idx), flush=True, end='')

        hidden = net.repackage_hidden(hidden)

        x, y = x.to(device), y.to(device)

        output, hidden = net(x, hidden)
        val_loss = criterion(output, y.float())

        val_losses.append(val_loss.item())

        pred = torch.round(output)
        total_correct += np.sum(pred.eq(y.float()).detach().to('cpu').numpy())

    accuracy = total_correct / len(batch_gen.dataset_dict['validation'])
    net.train()
    return np.mean(val_losses), accuracy
