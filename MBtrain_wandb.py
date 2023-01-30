import wandb
import argparse
import os
import copy
import gc

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models import SRCNN
from MB16datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr, calc_ssim, calc_ergas, calc_scc

# to use with wandb
# os.environ["WANDB_API_KEY"] = 
# wandb.init()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    torch.manual_seed(args.seed)

    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)

    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    best_ssim = 0.0
    best_ergas = 100.0
    best_scc = 0.0

    best_epoch_e = 0
    best_psnr_e = 0.0
    best_ssim_e = 0.0
    best_ergas_e = 100.0
    best_scc_e = 0.0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()
        epoch_psnr = AverageMeter()
        epoch_ssim = AverageMeter()
        epoch_ergas = AverageMeter()
        epoch_scc = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses_av = epoch_losses.avg
                t.set_postfix(loss='{:.8f}'.format(epoch_losses_av))
                t.update(len(inputs))
                epoch_psnr.update(calc_psnr(preds, labels).item(), len(inputs))
                epoch_ssim.update(calc_ssim(preds, labels).item(), len(inputs))
                epoch_ergas.update(calc_ergas(preds, labels).item(), len(inputs))
                epoch_scc.update(calc_scc(preds, labels).item(), len(inputs))
                
                del inputs, labels, preds, loss, data          
                
                torch.cuda.empty_cache()
                
            epoch_psnr_av = epoch_psnr.avg
            epoch_ssim_av = epoch_ssim.avg
            epoch_ergas_av = epoch_ergas.avg
            epoch_scc_av = epoch_scc.avg            
            print('train psnr: {:.4f}'.format(epoch_psnr.avg))
            print('train ssim: {:.4f}'.format(epoch_ssim.avg))
            print('train ergas: {:.4f}'.format(epoch_ergas.avg))
            print('train scc: {:.4f}'.format(epoch_scc.avg))
            print('train losses: {:.8f}'.format(epoch_losses.avg))
                           

        torch.cuda.empty_cache()
        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_losses_e = AverageMeter()
        epoch_psnr_e = AverageMeter()
        epoch_ssim_e = AverageMeter()
        epoch_ergas_e = AverageMeter()
        epoch_scc_e = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

                loss_e = criterion(preds, labels)
                epoch_losses_e.update(loss_e.item(), len(inputs))
                epoch_losses_e_av = epoch_losses_e.avg

                epoch_psnr_e.update(calc_psnr(preds, labels).item(), len(inputs))
                epoch_ssim_e.update(calc_ssim(preds, labels).item(), len(inputs))
                epoch_ergas_e.update(calc_ergas(preds, labels).item(), len(inputs))
                epoch_scc_e.update(calc_scc(preds, labels).item(), len(inputs))
            
            del inputs, labels, preds, loss_e, data
            
            torch.cuda.empty_cache()
            

        epoch_psnr_e_av = epoch_psnr_e.avg
        epoch_ssim_e_av = epoch_ssim_e.avg
        epoch_ergas_e_av = epoch_ergas_e.avg
        epoch_scc_e_av = epoch_scc_e.avg
        print('eval psnr: {:.4f}'.format(epoch_psnr_e.avg))
        print('eval ssim: {:.4f}'.format(epoch_ssim_e.avg))
        print('eval ergas: {:.4f}'.format(epoch_ergas_e.avg))
        print('eval scc: {:.4f}'.format(epoch_scc_e.avg))
        print('eval losses: {:.8f}'.format(epoch_losses_e.avg))

        if epoch_psnr_e_av > best_psnr_e:
            best_epoch_psnr_e = epoch
            best_psnr_e = epoch_psnr_e_av
            best_weights = copy.deepcopy(model.state_dict())        
          
        # wandb.log({'loss_e':epoch_losses_e_av, 'loss':epoch_losses_av, 'epoch':epoch, 'psnr_e':epoch_psnr_e_av, 'psnr':epoch_psnr_av, 'ssim_e':epoch_ssim_e_av, 'ssim':epoch_ssim_av, 'ergas_e':epoch_ergas_e_av, 'ergas':epoch_ergas_av, 'scc_e':epoch_scc_e_av, 'scc':epoch_scc_av})
      
    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch_psnr_e, best_psnr_e))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))
