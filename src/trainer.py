import sys
import tqdm
import torch
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state_dict, thres, save_dir, fn='model_best.pth'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_fn = os.path.join(save_dir, fn)
    torch.save(
         {
            'model':state_dict,
            'threshold': thres
         },
         save_fn)
    print(f'MODEL IS SAVED TO {save_fn}')


def optimal_f1(labels, predictions, metrics):
    thres = np.linspace(0, 1, 100)
    f1s = [metrics(labels, predictions > thr) for thr in thres]
    idx = np.argmax(f1s)
    return f1s[idx], thres[idx]


def wandb_log(**kwargs):
    for k, v in kwargs.items():
        wandb.log({k: v})


def forward_help(functions, pred, true):
    if isinstance(functions, list):
        loss = 0
        for function in functions:
            loss += function(pred, true)
    else:
        loss = functions(pred, true)

    return loss


def train_one_epoch(train_loader, model, loss_func, metric_func, device, optimizer):
    running_metric = {
        'loss' : AverageMeter(),
        # 'metric' : AverageMeter(),
    }

    model.train()
    scaler = torch.cuda.amp.GradScaler()
    with tqdm.tqdm(train_loader, total=len(train_loader), desc="Train", file=sys.stdout) as iterator:
        for sample in iterator:
            train_x = sample['input'].float().to(device)
            train_y = sample['target'].float().to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                pred_y = model(train_x)
                loss = loss_func(input=pred_y, target=train_y)

            scaler.scale(loss).backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            scaler.step(optimizer)
            scaler.update()

            loss_value = loss.item()
            # metric_value = metric_func(labels=train_y, predictions=pred_y.sigmoid()).item()

            running_metric['loss'].update(loss_value, train_x.size(0))
            # running_metric['metric'].update(metric_value, train_x.size(0))

            log = 'loss - {:.5f}'.format(running_metric['loss'].avg)
            # log = 'loss - {:.5f}, metric - {:.5f}'.format(running_metric['loss'].avg, running_metric['metric'].avg)
            iterator.set_postfix_str(log)

    return running_metric


def validate_one_epoch(valid_loader, model, loss_func, metric_func, device):
    running_metric = {
        'loss' : AverageMeter(),
        # 'metric' : AverageMeter(),
    }

    model.eval()
    pred_score = list()
    true_label = list()

    with tqdm.tqdm(valid_loader, total=len(valid_loader), desc="Valid", file=sys.stdout) as iterator:
        for sample in iterator:
            train_x = sample['input'].float().to(device)
            train_y = sample['target'].float().to(device)

            with torch.no_grad():
                pred_y = model(train_x)

            loss_value = loss_func(input=pred_y, target=train_y).item()
            # metric_value = metric_func(labels=train_y, predictions=pred_y.sigmoid()).item()

            pred_score.extend(pred_y.sigmoid().detach().cpu().numpy().squeeze())
            true_label.extend(train_y.detach().cpu().numpy().squeeze())

            running_metric['loss'].update(loss_value, train_x.size(0))
            # running_metric['metric'].update(metric_value, train_x.size(0))

            # log = 'loss - {:.5f}, metric - {:.5f}'.format(running_metric['loss'].avg, running_metric['metric'].avg)
            log = 'loss - {:.5f}'.format(running_metric['loss'].avg)
            iterator.set_postfix_str(log)

    pf1, thres = optimal_f1(true_label, pred_score, metric_func)
    running_metric['metric'] = pf1
    print(f'METRIC : {pf1}, THRES : {thres}')
    return running_metric, thres


class ModelTrainer:
    def __init__(self, model, train_loader, valid_loader, loss_func, metric_func, optimizer, device, save_dir, 
                       num_epochs, parallel=False, mode='max', scheduler=None, use_wandb=False):

        assert mode in ['min', 'max']

        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.loss_func = loss_func
        self.metric_func = metric_func
        self.optimizer = optimizer
        self.device = device
        self.save_path = str(os.path.join(save_dir, datetime.now().strftime("%m%d%H%M%S")))
        self.mode = mode
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.use_wandb = use_wandb
        self.parallel = parallel

        self.log = {
            'train_loss' : list(),
            'train_metric' : list(),
            'valid_loss' : list(),
            'valid_metric' : list(),
        }

        self.elapsed_time = None
        self.lr_curve = list()

    def initWandb(self, project_name, run_name, args):
        assert self.use_wandb == True
    
        wandb.init(project=project_name)
        wandb.run.name = run_name
        wandb.config.update(args)
        wandb.watch(self.model)

    def train(self):
        if self.device == 'cpu':
            print('[info msg] Start training the model on CPU')
        elif self.parallel and torch.cuda.device_count() > 1:
            print(f'Start training the model on {torch.cuda.device_count()} '
                  f'{torch.cuda.get_device_name(torch.cuda.current_device())} in parallel')
            self.model = torch.nn.DataParallel(self.model)
        else:
            print(f'[info msg] Start training the model on {torch.cuda.get_device_name(torch.cuda.current_device())}')
        print('=' * 50)

        self.model.to(self.device)

        if self.mode =='max':
            best_metric = -float('inf')
        else:
            best_metric = float('inf')

        startTime = datetime.now()

        print('[info msg] training start !!')
        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch+1, self.num_epochs))
            train_metric = train_one_epoch(
                train_loader=self.train_loader,
                model=self.model,
                loss_func=self.loss_func,
                metric_func=self.metric_func,
                device=self.device,
                optimizer=self.optimizer,
            )
            valid_metric, thres = validate_one_epoch(
                valid_loader=self.valid_loader,
                model=self.model,
                loss_func=self.loss_func,
                metric_func=self.metric_func,
                device=self.device,
            )

            self.log['train_loss'].append(train_metric['loss'].avg)
            # self.log['train_metric'].append(train_metric['metric'].avg)
            self.log['train_metric'].append(0)
            self.log['valid_loss'].append(valid_metric['loss'].avg)
            self.log['valid_metric'].append(valid_metric['metric'])
            self.lr_curve.append(self.optimizer.param_groups[0]['lr'])

            if self.use_wandb:
                wandb.log({
                    "Valid Metric": valid_metric['metric'],
                    # "Train Metric": train_metric['metric'].avg,
                    "Valid Loss": valid_metric['loss'].avg,
                    "Train Loss": train_metric['loss'].avg,
                    })

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(valid_metric['metric'].avg)
                else:
                    self.scheduler.step()

            if (self.mode =='min' and valid_metric['metric'] < best_metric) or \
               (self.mode =='max' and valid_metric['metric'] > best_metric) :
                best_metric = valid_metric['metric']

                if self.use_wandb:
                    wandb.run.summary["best_metric"] = valid_metric['metric']

                save_checkpoint(
                    state_dict=self.model.state_dict(),
                    thres=thres,
                    save_dir=self.save_path,
                    fn='model_best.pth'
                )  

        self.elapsed_time = datetime.now() - startTime
        self.__save_result()


    def __save_result(self):
        for key, value in self.log.items():
            self.log[key] = np.array(value)

        if self.mode =='max':
            best_train_metric_pos = np.argmax(self.log['train_metric'])
            best_val_metric_pos = np.argmax(self.log['valid_metric'])

        if self.mode =='min':
            best_train_metric_pos = np.argmin(self.log['train_metric'])
            best_val_metric_pos = np.argmin(self.log['valid_metric'])

        best_train_loss = self.log['train_loss'][best_train_metric_pos]
        best_val_loss = self.log['valid_loss'][best_val_metric_pos]
        best_train_metric = self.log['train_metric'][best_train_metric_pos]
        best_val_metric = self.log['valid_metric'][best_val_metric_pos]

        print('=' * 50)
        print('[info msg] training is done')
        print("Time taken: {}".format(self.elapsed_time))
        print("best metric is {} w/ loss {} at epoch : {}".format(best_val_metric, best_val_loss, best_val_metric_pos))    

        print('=' * 50)
        print('[info msg] model weight and log is save to {}'.format(self.save_path))

        with open(os.path.join(self.save_path, 'log.txt'), 'w') as f:         
            f.write(f'total ecpochs : {self.num_epochs}\n')
            f.write(f'time taken : {self.elapsed_time}\n')
            f.write(f'best_train_metric {best_train_metric} w/ loss {best_train_loss} at epoch : {best_train_metric_pos}\n')
            f.write(f'best_valid_metric {best_val_metric} w/ loss {best_val_loss} at epoch : {best_val_metric_pos}\n')

        df_learning_curves = pd.DataFrame.from_dict({
                    'loss_train': self.log['train_loss'],
                    'loss_val': self.log['valid_loss'],
                    'metric_train': self.log['train_metric'],
                    'metric_val': self.log['valid_metric'],
                })

        df_learning_curves.to_csv(os.path.join(self.save_path, 'learning_curves.csv'), sep=',')

        plt.figure(figsize=(15,5))
        plt.subplot(1, 2, 1)
        plt.title('loss')
        plt.plot(self.log['train_loss'], label='train loss')
        plt.plot(self.log['valid_loss'], label='valid loss')
        plt.axvline(x=best_val_metric_pos, color='r', linestyle='--', linewidth=1.5, label='best_val_metric')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.title('metric')
        plt.plot(self.log['train_metric'], label='train metric')
        plt.plot(self.log['valid_metric'], label='valid metric')
        plt.axvline(x=best_val_metric_pos, color='r', linestyle='--', linewidth=1.5, label='best_val_metric')
        plt.legend()
        plt.savefig(os.path.join(self.save_path, 'history.png'))
        # plt.show()
        
        plt.figure(figsize=(15,5))
        plt.title('lr_rate curve')
        plt.plot(self.lr_curve)
        plt.savefig(os.path.join(self.save_path, 'lr_history.png'))
        # plt.show()

    @property
    def save_dir(self):
        return self.save_path