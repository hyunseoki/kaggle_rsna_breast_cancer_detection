import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
import pandas as pd
import torch
from dataset import RSNADataset
from trainer import ModelTrainer
import loss as losses
import model as models
import metric as metrics
from transforms import (
    get_train_transforms,
    get_valid_transforms,
)
from util import (
    seed_everything,
    str2bool,
    get_sampler
)


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    base_path = r"./data/train/1024"
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--parallel', type=str2bool, default=False)
    parser.add_argument('--kfold', type=int, default=0, choices=[0, 1, 2, 3, 4])

    parser.add_argument('--base_path', type=str, default=base_path)
    parser.add_argument('--save_folder', type=str, default='./checkpoint')
    parser.add_argument('--use_wandb', type=str2bool, default=False)

    parser.add_argument('--backbone', type=str, default='efficientnet', choices=['efficientnet', 'resnet'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--resume', type=str, default=None)

    parser.add_argument('--loss_weight', type=float, default=1.0)
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'])

    parser.add_argument('--train_oversample', type=str2bool, default=False)
    parser.add_argument('--val_oversample', type=str2bool, default=False)

    parser.add_argument('--comments', type=str, default=None)
    args = parser.parse_args()
    
    if args.comments is not None:
        args.save_folder = os.path.join(args.save_folder, args.comments)

    print('=' * 50)
    print('[info msg] arguments')
    for key, value in vars(args).items():
        print(key, ":", value)
    print('=' * 50)

    train_df = pd.read_csv(rf'./data/5fold/train{args.kfold}.csv')
    valid_df = pd.read_csv(rf'./data/5fold/val{args.kfold}.csv')
    
    train_dataset = RSNADataset(
            base_path=args.base_path,
            label_df=train_df,
            transforms=get_train_transforms()
        )
    valid_dataset = RSNADataset(
            base_path=args.base_path,
            label_df=valid_df,
            transforms=get_valid_transforms()
        )

    if args.train_oversample:
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=get_sampler(df=train_df),
            drop_last=True,
            num_workers=2,
        )
    else:
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=2,
        )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size * 4,
        shuffle=False,
        num_workers=2,
    )

    if args.backbone=='efficientnet':
        model = models.Classifier()
    elif args.backbone=='resnet':
        model = models.ResNetModel()

    if args.resume != None:
        pth_data = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(pth_data['model'])

    if args.val_oversample:
        metric = metrics.probabilistic_f1_oversample
    else:
        metric = metrics.probabilistic_f1

    if args.loss_type=='bce':
        loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor([args.loss_weight], device=args.device))
    else:
        loss = losses.FocalWithLogitLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    trainer = ModelTrainer(
        model=model,
        train_loader=train_data_loader,
        valid_loader=valid_data_loader,
        loss_func=loss,
        metric_func=metric,
        optimizer=optimizer,
        device=args.device,
        parallel=args.parallel,
        save_dir=args.save_folder,
        mode='max',
        scheduler=scheduler, 
        num_epochs=args.epochs,
        use_wandb=args.use_wandb,
    )

    if args.use_wandb:
        trainer.initWandb(
            project_name='KAGGLE_RSNA_MAMMOGRAPHY',
            run_name=args.comments,
            args=args,
        )

    trainer.train()

    with open(os.path.join(trainer.save_dir, 'config.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write('{} : {}\n'.format(key, value)) 


if __name__ == '__main__':
    main()
