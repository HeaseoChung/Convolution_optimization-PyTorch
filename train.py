import argparse
import os
import math
import logging
import hydra
import wandb

import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

from torch.utils.data.dataloader import DataLoader
from torch import nn
from torch.cuda import amp

from models import EDSR
from utils import calc_psnr, sample_data
from dataset import Dataset

@hydra.main(config_path="./configs", config_name="config")
def main(cfg):
    if cfg.training.use_wandb:
        wandb.init(project=cfg.model.name)
        wandb.config.update(cfg)

    """ weight를 저장 할 경로 설정 """ 
    cfg.training.ckpt_dir = os.path.join(cfg.training.ckpt_dir,  f"{cfg.model.name}x{cfg.model.scale}")

    if not os.path.exists(cfg.training.ckpt_dir):
        os.makedirs(cfg.training.ckpt_dir)

    """ GPU 디바이스 설정 """
    cudnn.benchmark = True
    cudnn.deterministic = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    """ Torch Seed 설정 """
    torch.manual_seed(cfg.training.seed)

    model = EDSR(scale=cfg.model.scale, num_channels=cfg.model.n_channels, num_feats=cfg.model.n_features, num_blocks=cfg.model.n_blocks, res_scale=cfg.model.res_scale, block_type=cfg.model.res_block_type).to(device)
    """ Loss 및 Optimizer 설정 """
    pixel_criterion = nn.MSELoss().to(device)
    psnr_optimizer = torch.optim.Adam(model.parameters(), cfg.training.lr, (0.9, 0.999))

    total_iters = cfg.training.n_iters
    start_iters = 0
    best_psnr = 0

    """ 체크포인트 weight 불러오기 """
    if os.path.exists(cfg.training.resume_ckpt):
        checkpoint = torch.load(cfg.training.resume_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        psnr_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iters = checkpoint['iters'] + 1
        loss = checkpoint['loss']
        best_psnr = checkpoint['best_psnr']

    """ 스케줄러 설정 """
    scaler = amp.GradScaler()

    """ 데이터셋 & 데이터셋 설정 """
    train_dataset = Dataset(cfg.training.train_dir, cfg.training.patch_size, scale=cfg.model.scale)
    train_dataloader = DataLoader(
                            dataset=train_dataset,
                            batch_size=cfg.training.batch_size,
                            shuffle=True,
                            num_workers=cfg.training.n_workers,
                            pin_memory=True
                        )
    loader = sample_data(train_dataloader)

    eval_dataset = Dataset(cfg.training.eval_dir, cfg.training.patch_size, scale=cfg.model.scale)
    eval_dataloader = DataLoader(
                                dataset=eval_dataset, 
                                batch_size=cfg.training.batch_size,
                                shuffle=False,
                                num_workers=cfg.training.n_workers,
                                pin_memory=True
                                )
    


    """ 트레이닝 시작 & 테스트 시작"""
    for idx in range(start_iters, total_iters):
        idx = idx + start_iters
        
        if i > total_iters:
            break

        """  트레이닝 시작 """
        model.train()

        """ 데이터 로드 """
        lr, hr = next(loader)
        lr = lr.to(device)
        hr = hr.to(device)

        psnr_optimizer.zero_grad()

        with amp.autocast():
            preds = model(lr)
            loss = pixel_criterion(preds, hr)
        
        scaler.scale(loss).backward()
        scaler.step(psnr_optimizer)
        scaler.update()

        if cfg.training.use_wandb:
            wandb.log({"loss": loss})

        if idx % 100 == 0:
            """  테스트 시작 """
            model.eval()
            psnr = 0.0

            for i, (lr, hr) in enumerate(eval_dataloader):
                lr = lr.to(device)
                hr = hr.to(device)

                with torch.no_grad():
                    preds = model(lr)

                vutils.save_image(
                    lr.detach(), os.path.join(cfg.training.ckpt_dir, f"LR_{idx}-{i}.jpg")
                )
                vutils.save_image(
                    hr.detach(), os.path.join(cfg.training.ckpt_dir, f"HR_{idx}-{i}.jpg")
                )
                vutils.save_image(
                    preds.detach(), os.path.join(cfg.training.ckpt_dir, f"preds_{idx}-{i}.jpg")
                )

                psnr += calc_psnr(hr, preds)

            psnr_avg = psnr/len(eval_dataloader)

            if cfg.training.use_wandb:
                wandb.log({"PSNR": psnr_avg})

            if psnr_avg > best_psnr:
                best_psnr = psnr_avg
                torch.save(
                    model.state_dict(), os.path.join(cfg.training.ckpt_dir, 'best.pth')
                )

        """ iters 100번 마다 저장 """
        if idx % 100 == 0:
            torch.save(
                {
                    'iters': idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': psnr_optimizer.state_dict(),
                    'loss': loss,
                    'best_psnr': best_psnr,
                }, os.path.join(cfg.training.ckpt_dir, '{}.pth'.format(idx))
            )

if __name__ == '__main__':
    main()