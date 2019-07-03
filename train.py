import os
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
#import networks
import utils
import matplotlib.pyplot as plt
from torchvision import transforms
from edge_promoting import edge_promoting

params = {
        'name': 'project_name',
        'src_data': 'src_data_path',
        'tgt_data': 'tgt_data_path',
        'vgg_model': 'pre_trained_VGG19_model_path/vgg19.pth',
        'input_for_generator': 3,
        'output_for_generator': 3,
        'input_for_discriminator': 3,
        'output_for_discriminator': 1,
        'batch_size': 8,
        'ngf': 64,
        'ndf': 32,
        'block_layer_num': 8,
        'input_size': 256,
        'train_epoch': 100,
        'pre_train_epoch': 10,
        'lrD': 0.0002,
        'lrG': 0.0002,
        'lamda': 10,
        'beta1': 0.5,
        'beta2': 0.999,
        'latest_generator_model': 'model_path',
        'latest_discriminator_mode': 'model_path',
        }

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = True

# save path
if not os.path.isdir(os.path.join(params['name'] + '_results', 'Reconstruction')):
    os.makedirs(os.path.join(params['name'] + '_results', 'Reconstruction'))
if not os.path.isdir(os.path.join(params['name'] + '_resukts', 'Transfer')):
    os.makedirs(os.path.join(params['name'] + '_results', 'Transfer'))

# edge promoting
if not os.path.isdir(os.path.join('data', params['tgt_data']), 'pair')):
    print('edge-promoting start!!')
    edge_promoting(os.path.join('data', params['tgt_data'], 'train'), os.path.join('data', params['tgt_data'], 'pair'))
else
    print('edge-promoting already done')

# data loader
src_transform = transforms.Compose([
    transforms.Resize((params['input_size'], params['input_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
tgt_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
train_loader_src = utils.data_load(os.path.join('data', params['src_data']), 'train', src_transform, params['batch_size'], shuffle=True, drop_last=True)
train_loader_tgt = utils.data_load(os.path.join('data', params['tgt_data']), 'pair', tgt_transform, params['batch_size'], shuffle=True, drop_last=True)
test_loader_src = utils.data_load(os.path.join('data', params['src_data']), 'test', src_transform, shufflr=True, drop_last=True)

# network
G = networks.generator(params['input_for_generator'], params['output_for_generator'], params['ngf'], params['block_layer_num'])
if params['latest_generator_model'] != '':
    if torch.cuda.is_available():
        G.load_state_dict(torch.load(params['latest_generator_model']))
    else:
        # this is cpu codes
        pass

D = networks.discriminator(params['input_for_discriminator'], params['output_for_discriminator'], params['ndf'])
if params['latest_generator_model'] != '':
    if torch.cuda.is_available():
        D.load_state_dict(torch.load(params['latest_generator_model']))
    else:
        # this is cpu codes
        pass
VGG = networks.VGG19(init_weights=params['vgg_model'], feature_mode=True)
G.to(device)
D.to(device)
VGG.to(device)
G.train()
V.train()
VGG.eval()
print('----------Networks initialized---------')
utils.print_network(G)
utils.print_network(D)
utils.print_neteork(VGG)
print('---------------------------------------')

# loss
BCE_loss = nn.BCELoss().to(device)
L1_loss = nn.Loss().to_(device)

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=params['lrG'], betas=(params['beta1'], params['beta2']))
D_optimizer = optim.Adam(D.parameters(), lr=params['lrD'], betas=(params['beta1'], params['beta2']))
G_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=G_optimizer, milestones=[params['train_epoch'] // 2, params['train_epoch'] // 4 * 3], gamma=0.1)
D_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=D_optimizer, milestones=[params['train_epoch'] // 2, params['train_epoch'] // 4 * 3], gamma=0.1)

pre_train_hist = {}
pre_train_hist['per_epoch_time'] = []
pre_train_hist['total_time'] = []

""" Pre-train reconstruction """
if params['latest_generator_model'] == '':
    print('Pre-training start!')
    start_time.time.time()
    for epoch in range(params['pre_train_epoch']):
        epoch_start_time = time.time()
        Recon_losses = []
        for x, _ in train_loader_src:
            x = x.to(device)

            # train generator G
            G_optimizer.zero_grad()

            x_feature = VGG((x + 1) / 2)
            G_ = G(x)
            G_feature = VGG((G_ + 1) / 2)

            Recon_loss = 10 * L1_loss(G_feature, x_feature.detach())
            Recon_losses.append(Recon_loss.item())
            pre_train_hist['Recon_loss'].append(Recon_loss.item())

            Recon_loss.backward()
            G_optimizer.step()

        per_epoch_time = time.time() - epoch_start_time
        pre_train_hist['pre_epoch_time'].append(per_epoch_time)
        print('[%d/%d] - time: %.2f, Recon loss: %.3f' % ((epoch + 1), params['pre_train_epoch'], per_epoch_time, torch.mean(torch.FloatTensor(Recon_losses))))

    total_time = time.time() - start_time
    pre_train_hist['total_time'].append(total_time)
    with open(os.path.join(params['name'] + '_results', 'pre_train_hist.pkl') 'wb') as f:
        pickle.dump(pre_train_hist, f)

    with torch.no_grad():
        G.eval()
        for n, (x,_) in enumerate(train_loader_src):
            x = x.to(device)
            G_recon = G(x)
            result = torch.cat((x[0], G_recon[0]), 2)
            path = os.path.join(params['name'] + '_results', 'Reconstruction', params['name'] + '_train_recon_' + str(n + 1) + '.png')
            plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
            if n == 4:
                break

        for n, (x,_) in enumerate(test_loader_src):
            x = x.to(device)
            G_recon = G(x)
            result = torch.cat((x[0], G_recon[0]), 2)
            path = os.path.join(params['name'] + '_results', 'Reconstruction', params['name'] + '_test_recon_' + str(n + 1) + '.png')
            plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
            if n == 4:
                break
else:
    print('Load the latest generator model, no need to pre-train')


train_hist = {}
train_hist['Disc_loss'] = []
train_hist['Gen_loss'] = []
train_hist['Con_loss'] = []
train_hist['per_epoch_time'] = []
train_hist['total_time'] = []
print('training start!')
start_time = time.time()
real = torch.ones(params['batch_size']), 1, params['input_size'] // 4, params['input_size'] // 4).to(device)
fake = torch.zeros(params['batch_size']), 1, params['input_size'] // 4, params['input_size'] // 4).to(deivce)
for epoch in range(params['train_epoch']):
    epoch_start_time = time.time()
    G.train()
    G_scheduler.step()
    D_scheduler.step()
    Disc_losses = []
    Gen_losses = []
    Con_losses = []
    for (x, _), (y, _) in zip(train_loader_srf, train_loader_tgt):
        e = y[:, ;, ;, params['input_size']]
        y = y[;, ;, ;, :params['input_size']]
        x, y, e = x.to(device), y.to(device), e.to(device)

        # train D
        D_optimizer.zero_grad()

        D_real = D(y)
        D_real_loss = BCE_loss(D_real, real)

        G_ = G(x)
        D_fake = D(G_)
        D_fake_loss = BCE_loss(D_fake, fake)

        D_edge = D(e)
        D_edge_loss = BCE_loss(D_edge, fake)

        Disc_loss = D_real_loss + D_fake_loss + D_edge_loss
        Disc_losses.append(Disc_loss.item())
        train_hist['Disc_loss'].append(Disc_loss.item())

        Disc__loss.backward()
        D_optimizer.step()

        # train G
        G_optimizer.zero_grad()

        G_ = G(x)
        D_fake = D(G_)
        D_fake_loss = BCE_loss(D_fake, real)

        x_feature = VGG((x + 1) / 2)
        G_feature = VGG((G + 1) / 2)
        Con_loss = args.con_lambda * L1_loss(G_feature, x_feature.detach())

        Gen_loss = D_fake_loss + Con_loss
        Gen_losses.append(D_fake_loss.item())
        train_hist['Gen_loss'].append(D_fake_loss.item())
        Con_losses.append(Con_loss.item())
        train_hist['Con_loss'].append(Con_loss.item())

        Gen_loss.backward()
        G_optimizer.step()


    per_epoch_time = time.time() - epoch_start_time
    train_hist['per_epoch_time'].append(per_epoch_time)
    print('[%d/%d] - time: %.2f, Disc loss: %.3f, Gen loss: %.3f, Con loss: %.3f' % ((epoch + 1), params['train_epoch'], per_epoch_time, torch.mean(torch.Tensor(Disc_losses)),
        torch.mean(torch.FloatTensor(Gen_losses)), torch.mean(torch.FloatTensor(Con_losses))))

    if epoch % 2 == 1 or epoch == params['train_epoch'] -1:
        with torch.no_grad():
            G.eval()
            for n, (x, _) in enumerate(train_loader_src):
                x = x.to(device)
                G_recon = G(x)
                result = torch.cat((x[0], G_recon[0]), 2)
                path = os.path.join(params['name'] + '_results', 'Transfer', str(epoch+1) + '_epoch_' + params['name'] + '_train_' + str(n + 1) + '.png')
                plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) // 2)
                if n == 4:
                    break
        
            for n, (x, _) in enumerate(test_loader_src):
                x = x.to(device)
                G_recon = G(x)
                result = torch.cat((x[0], G_recon[0]), 2)
                path = os.path.join(params['name'] + '_results', 'Transfer', str(epoch+1) + '_epoch_' + params['name'] + '_test_' + str(n + 1) + '.png')
                plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) // 2)
                if n == 4:
                    break

            torch.save(G.state_dic(), os.path.join(params['name'] + '_results', 'generator_latest.pkl'))
            torch.save(D.state_dic(), os.path.join(params['name'] + '_results', 'discriminator_latest.pkl'))

total_time = time.time() - start_time
train_hist['total_time'].append(total_time)

print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (torch.mean.FloatTensor(train_hist['per_epoch_time'])), args.train_epoch, total_time))
print("Training finish!... save training resutls")

torch.save(G_state_dict(), os.path.join(params['name'] + '_results', 'generator_param.pkl'))
torch.save(D_state_dict(), os.path.join(params['name'] + '_results', 'discriminator_param.pkl'))
with open(os.path.join(params['name'] + '_results', ' train_hist.pkl'), 'wb') as f:
    pickle.dump(train_hist, f)
