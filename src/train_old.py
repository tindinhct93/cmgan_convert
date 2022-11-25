from models.generator import TSCNet
from models import discriminator
import os
import time 
import numpy as np
from data import dataloader
import torch.distributed as dist
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
from utils import *
import logging
import evaluation
from torchinfo import summary
import argparse
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=120, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--log_interval", type=int, default=500)
parser.add_argument("--decay_epoch", type=int, default=30, help="epoch from which to start lr decay")
parser.add_argument("--init_lr", type=float, default=5e-4, help="initial learning rate")
parser.add_argument("--cut_len", type=int, default=16000*2, help="cut length, default is 2 seconds in denoise "
                                                                 "and dereverberation")
parser.add_argument("--data_dir", type=str, default='/home/minhkhanh/Desktop/work/denoiser/dataset/voice_bank_demand',
                    help="dir of VCTK+DEMAND dataset")
parser.add_argument("--save_model_dir", type=str, default='./saved_model',
                    help="dir of saved model")
parser.add_argument("--loss_weights", type=list, default=[0.1, 0.9, 0.2, 0.05],
                    help="weights of RI components, magnitude, time loss, and Metric Disc")
parser.add_argument("--save_dir", type=str, default='./saved_tracks_best', help="where enhanced tracks to be saved") 

args = parser.parse_args()
logging.basicConfig(level=logging.INFO)


class Trainer:
    def __init__(self, rank, train_ds, test_ds):
        self.n_fft = 400
        self.hop = 100
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.rank = rank
        self.use_amp = True
        self.interval_eval = 1
        self.max_clip_grad_norm = 0.1
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        model = TSCNet(num_channel=64, num_features=self.n_fft // 2 + 1)
        self.model = DistributedDataParallel(model.to(rank), device_ids=[rank])
        
        model_discriminator = discriminator.Discriminator(ndf=16)
        self.model_discriminator = DistributedDataParallel(model_discriminator.to(rank), device_ids=[rank])
                                    
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.init_lr)
        self.optimizer_disc = torch.optim.AdamW(self.model_discriminator.parameters(), lr=2*args.init_lr)
        self.save_model_dir = os.path.join(os.getcwd(), f"n_fft-{self.n_fft},hop-{self.hop}")
        
        self.best_state = None
        self.resume = True 
        self.epoch_start = 0
        self.best_loss = float('inf')
        if self.resume:
            self._reset()

        if self.rank == 0:
            summary(self.model, [(4, 2, args.cut_len//self.hop+1, int(self.n_fft/2)+1)])
            summary(self.model_discriminator, [(1, 1, int(self.n_fft/2)+1, args.cut_len//self.hop+1),
                                     (1, 1, int(self.n_fft/2)+1, args.cut_len//self.hop+1)])

            os.makedirs(self.save_model_dir, exist_ok=True)
            self.writer = SummaryWriter()
    
    def _serialize(self, epoch):
        '''
        function help to save new general checkpoint
        '''
        package = {}
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            package["model"] = self.model.module.state_dict()
            package['discriminator'] = self.model_discriminator.module.state_dict()
        else:
            package["model"] = self.model.state_dict()
            package['discriminator'] = self.model_discriminator.state_dict()
        
        package['optimizer'] = self.optimizer.state_dict()
        package['optimizer_disc'] = self.optimizer_disc.state_dict()
        package['best_state'] = self.best_state
        package['loss'] = self.best_loss
        package['epoch'] = epoch
        tmp_path = os.path.join(self.save_model_dir, "checkpoint.tar")
        torch.save(package, tmp_path)

        # Save only the best model, pay attention that don't need to save best discriminator
        # because when infer time, you only need model to predict, and if better the discriminator
        # the worse the model ((:
        model = package['best_state']
        tmp_path = os.path.join(self.save_model_dir, "best.th")
        torch.save(model, tmp_path)


    def _reset(self):
        dist.barrier()
        if os.path.exists(self.save_model_dir) and os.path.isfile(self.save_model_dir + "/checkpoint.tar"):
            if self.rank == 0:
                print("\n<<<<<<<<<<<<<<<<<< Load pretrain >>>>>>>>>>>>>>>>>>\n")
            package = torch.load(self.save_model_dir + "/checkpoint.tar")
            
            print("Loading last state for resuming training")
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model.module.load_state_dict(package['model'])
                self.model_discriminator.module.load_state_dict(package['discriminator'])
            else:
                self.model.load_state_dict(package['model'])
                self.model_discriminator.load_state_dict(package['discriminator'])
                
            self.optimizer.load_state_dict(package['optimizer'])
            self.optimizer_disc.load_state_dict(package['optimizer_disc'])
            self.epoch_start = package['epoch'] + 1
            self.best_loss = package['loss']
            self.best_state = package['best_state']
            if self.rank == 0:
                print(f"Model checkpoint loaded. Training will begin at {self.epoch_start} epoch.")

    def train_step(self, batch):
        clean = batch[0].cuda()
        noisy = batch[1].cuda()
        one_labels = torch.ones(min(args.batch_size, clean.size(0))).cuda()

        # Normalization
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
        noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(clean * c, 0, 1)

        self.optimizer.zero_grad()
        noisy_spec = torch.stft(noisy, self.n_fft, self.hop, window=torch.hamming_window(self.n_fft).cuda(),
                                onesided=True)
        clean_spec = torch.stft(clean, self.n_fft, self.hop, window=torch.hamming_window(self.n_fft).cuda(),
                                onesided=True)
        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        clean_spec = power_compress(clean_spec)
        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)

        # Runs the forward pass under autocast.
        with autocast(enabled = self.use_amp):
            est_real, est_imag = self.model(noisy_spec)
            # output is float16 because linear layers autocast to float16.
            # assert est_real.dtype is torch.float16, f"est_real's dtype is not torch.float16 but {est_real.dtype}"
            # assert est_imag.dtype is torch.float16, f"est_imag's dtype is not torch.float16 but {est_imag.dtype}"

            est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
            est_mag = torch.sqrt(est_real**2 + est_imag**2)
            clean_mag = torch.sqrt(clean_real**2 + clean_imag**2)

            predict_fake_metric = self.model_discriminator(clean_mag, est_mag)
            gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())

            loss_mag = F.mse_loss(est_mag, clean_mag)
            loss_ri = F.mse_loss(est_real, clean_real) + F.mse_loss(est_imag, clean_imag)

            est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
            est_audio = torch.istft(est_spec_uncompress, self.n_fft, self.hop,
                                    window=torch.hamming_window(self.n_fft).cuda(), onesided=True)

            time_loss = torch.mean(torch.abs(est_audio - clean))
            length = est_audio.size(-1)
            loss = args.loss_weights[0] * loss_ri + args.loss_weights[1] * loss_mag + args.loss_weights[2] * time_loss \
                + args.loss_weights[3] * gen_loss_GAN

            # loss is float32 because mse_loss layers autocast to float32.
        assert loss.dtype is torch.float32, f"loss's dtype is not torch.float32 but {loss.dtype}"
        self.scaler.scale(loss).backward(retain_graph=True)
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_clip_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        est_audio_list = list(est_audio.detach().cpu().numpy())
        clean_audio_list = list(clean.cpu().numpy()[:, :length])
        pesq_score = discriminator.batch_pesq(clean_audio_list, est_audio_list)

        # The calculation of PESQ can be None due to silent part
        if pesq_score is not None:
            self.optimizer_disc.zero_grad()
            predict_enhance_metric = self.model_discriminator(clean_mag, est_mag.detach())
            (clean_mag, est_mag.detach())
            predict_max_metric = self.model_discriminator(clean_mag, clean_mag)
            discrim_loss_metric = F.mse_loss(predict_max_metric.flatten(), one_labels) + \
                                  F.mse_loss(predict_enhance_metric.flatten(), pesq_score)
            discrim_loss_metric.backward()
            self.optimizer_disc.step()
        else:
            discrim_loss_metric = torch.tensor([0.])

        return loss.item(), discrim_loss_metric.item()

    @torch.no_grad()
    def test_step(self, batch):
        clean = batch[0].cuda()
        noisy = batch[1].cuda()
        one_labels = torch.ones(min(args.batch_size, clean.size(0))).cuda()

        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
        noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(clean * c, 0, 1)

        noisy_spec = torch.stft(noisy, self.n_fft, self.hop, window=torch.hamming_window(self.n_fft).cuda(),
                                onesided=True)
        clean_spec = torch.stft(clean, self.n_fft, self.hop, window=torch.hamming_window(self.n_fft).cuda(),
                                onesided=True)
        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        clean_spec = power_compress(clean_spec)
        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)

        est_real, est_imag = self.model(noisy_spec)
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
        est_mag = torch.sqrt(est_real ** 2 + est_imag ** 2)
        clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2)

        predict_fake_metric = self.model_discriminator(clean_mag, est_mag)
        gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())

        loss_mag = F.mse_loss(est_mag, clean_mag)
        loss_ri = F.mse_loss(est_real, clean_real) + F.mse_loss(est_imag, clean_imag)

        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
        est_audio = torch.istft(est_spec_uncompress, self.n_fft, self.hop,
                                window=torch.hamming_window(self.n_fft).cuda(), onesided=True)

        time_loss = torch.mean(torch.abs(est_audio - clean))
        length = est_audio.size(-1)
        loss = args.loss_weights[0] * loss_ri + args.loss_weights[1] * loss_mag + args.loss_weights[2] * time_loss \
               + args.loss_weights[3] * gen_loss_GAN

        est_audio_list = list(est_audio.detach().cpu().numpy())
        clean_audio_list = list(clean.cpu().numpy()[:, :length])
        pesq_score = discriminator.batch_pesq(clean_audio_list, est_audio_list)
        if pesq_score is not None:
            predict_enhance_metric = self.model_discriminator(clean_mag, est_mag.detach())
            predict_max_metric = self.model_discriminator(clean_mag, clean_mag)
            discrim_loss_metric = F.mse_loss(predict_max_metric.flatten(), one_labels) + \
                                  F.mse_loss(predict_enhance_metric.flatten(), pesq_score)
        else:
            discrim_loss_metric = torch.tensor([0.])

        return loss.item(), discrim_loss_metric.item()

    def test(self):
        self.model.eval()
        self.model_discriminator.eval()
        gen_loss_total = 0.
        disc_loss_total = 0.
        for idx, batch in enumerate(self.test_ds):
            step = idx + 1
            loss, disc_loss = self.test_step(batch)
            gen_loss_total += loss
            disc_loss_total += disc_loss
        gen_loss_avg = gen_loss_total / step
        disc_loss_avg = disc_loss_total / step
        
        template = 'VALID STAGE: Generator loss: {}, Discriminator loss: {}'
        logging.info(
            template.format(gen_loss_avg, disc_loss_avg))

        return gen_loss_avg, disc_loss_avg

    def train(self):
        scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.decay_epoch, gamma=0.5)
        scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_disc, step_size=args.decay_epoch, gamma=0.5)
        
        for epoch in range(self.epoch_start, args.epochs):
            self.model.train()
            self.model_discriminator.train()
            gen_loss_train, disc_loss_train = [], []
            start_time = time.time()
            for idx, batch in enumerate(self.train_ds):
                step = idx + 1
                loss, disc_loss = self.train_step(batch)
                gen_loss_train.append(loss )
                disc_loss_train.append(disc_loss)
                if self.rank == 0:
                    template = 'Epoch {}, Step {}, Exe_time {}, loss: {}, disc_loss: {}'
                    if (step % args.log_interval) == 0:
                        end_time = time.time()
                        logging.info(template.format(epoch, step, end_time-start_time, loss, disc_loss))
                        start_time = end_time

            if self.rank == 0:
                print("---OVERALL SUMMARY EPOCH {}".format(epoch))
                gen_loss_train = np.mean(gen_loss_train)
                disc_loss_train = np.mean(disc_loss_train)
                template = 'TRAINING STAGE: Generator loss: {}, Discriminator loss: {}'
                logging.info(
                    template.format(gen_loss_train, disc_loss_train))

                self.writer.add_scalar("Loss_gen/train", gen_loss_train, epoch)
                self.writer.add_scalar("Loss_disc/train", disc_loss_train, epoch)

                gen_loss_valid, disc_loss_valid = self.test()
                self.writer.add_scalar("Loss_gen/valid", gen_loss_valid, epoch)
                self.writer.add_scalar("Loss_disc/valid", disc_loss_valid, epoch)
                # save best checkpoint
                # path = os.path.join(args.save_model_dir, 'best.th' + str(epoch) + '_' + str(gen_loss_valid)[:5])
                self.best_loss = min(self.best_loss, gen_loss_valid)
                if gen_loss_valid == self.best_loss:
                    self.best_state = copy_state(self.model.state_dict())

                self._serialize(epoch)

                if epoch % self.interval_eval == 0:
                    evaluation(self.model, 
                                args.data_dir + "/test/noisy", 
                                args.data_dir + "/test/clean",
                                True, 
                                args.save_dir)

            
            scheduler_G.step()
            scheduler_D.step()


def entry(rank, world_size):
    torch.cuda.set_device(rank)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank)

    train_ds, test_ds = dataloader.load_data(ds_dir = args.data_dir, 
                                            batch_size = args.batch_size, 
                                            n_cpu = 20, 
                                            cut_len = args.cut_len, 
                                            rank = rank, 
                                            world_size = world_size)
    
    trainer = Trainer(rank, train_ds, test_ds)
    trainer.train()

    

if __name__ == '__main__':
    
    print(args)
    available_gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print(available_gpus)
    world_size = len(available_gpus)

    mp.spawn(entry,
             args=(world_size,),
             nprocs=world_size,
             join=True)
