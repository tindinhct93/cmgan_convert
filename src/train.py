from models.generator import TSCNet
from models import discriminator
import os
from time import gmtime, strftime
from data import dataloader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from utils import *
import toml
from torchinfo import summary
import argparse
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import warnings
import logging
warnings.filterwarnings('ignore')



def cleanup():
    dist.destroy_process_group()

def setup(rank, world_size):
    torch.cuda.set_device(rank)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank)

def entry(rank, world_size, config):
    # init distributed training
    os.environ["CUDA_VISIBLE_DEVICES"] = config["main"]["device_ids"]
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    setup(rank, world_size)

    ## load config
    # train
    epochs = config["main"]["epochs"]
    batch_size = config['dataset_train']['dataloader']['batchsize']
    use_amp = config["main"]["use_amp"]
    max_clip_grad_norm = config["main"]["max_clip_grad_norm"]
    interval_eval = config["main"]["interval_eval"]
    resume = config['main']['resume']
    num_prints = config["main"]["num_prints"]
    data_test_dir = config['dataset_test']['path']

    init_lr = config['optimizer']['init_lr']
    gamma = config["scheduler"]["gamma"]
    decay_epoch = config['scheduler']['decay_epoch']
    num_channel = config['model']['num_channel']

    # feature
    n_fft = config["feature"]["n_fft"]
    hop = config["feature"]["hop"]
    ndf = config["feature"]["ndf"]

    cut_len = int(config["main"]["cut_len"])
    save_model_dir = os.path.join(config["main"]["save_model_dir"], config["main"]['name'] + '/checkpoints')

    if rank == 0:
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)
    
        # Store config file
        config_name = strftime("%Y-%m-%d %H:%M:%S", gmtime()).replace(' ', '_') + '.toml'
        with open(os.path.join(config["main"]["save_model_dir"], config["main"]['name'] + '/' + config_name), 'w+') as f:
            toml.dump(config, f)
            f.close()

        logging.basicConfig(filename=f"{save_model_dir}/train.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
    
    logger = logging.getLogger(__name__)

    # This should be needed to be reproducible https://discuss.pytorch.org/t/setting-seed-in-torch-ddp/126638
    config["main"]["seed"] += rank 
    set_seed(config["main"]["seed"])

    # Create train dataloader
    train_ds = dataloader.load_data(
                                        config['dataset_train']['path'], 
                                        batch_size = config['dataset_train']['dataloader']['batchsize'], 
                                        n_cpu = config['dataset_train']['dataloader']['n_worker'], 
                                        cut_len = cut_len, 
                                        rank = rank, 
                                        world_size = world_size,
                                        shuffle = config['dataset_train']['sampler']['shuffle']
                                    )

    test_ds = dataloader.load_data (
                                        config['dataset_valid']['path'], 
                                        batch_size = config['dataset_valid']['dataloader']['batchsize'], 
                                        n_cpu = config['dataset_valid']['dataloader']['n_worker'], 
                                        cut_len = cut_len, 
                                        rank = rank, 
                                        world_size = world_size,
                                        shuffle = config['dataset_valid']['sampler']['shuffle']
                                    )
    # model
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model = TSCNet(num_channel=num_channel, num_features=n_fft // 2 + 1)
    model = DistributedDataParallel(model.to(rank), device_ids=[rank])
    
    model_discriminator = discriminator.Discriminator(ndf=ndf)
    model_discriminator = DistributedDataParallel(model_discriminator.to(rank), device_ids=[rank])

    if rank == 0:
        summary(model, [(4, 2, cut_len//hop+1, int(n_fft/2)+1)])
        summary(model_discriminator, [(1, 1, int(n_fft/2)+1, cut_len//hop+1),
                                    (1, 1, int(n_fft/2)+1, cut_len//hop+1)])


    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr)
    optimizer_disc = torch.optim.AdamW(model_discriminator.parameters(), lr=2*init_lr)

    # scheduler
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_epoch, gamma=gamma)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_disc, step_size=decay_epoch, gamma=gamma)

    # tensorboard writer
    writer = SummaryWriter()
    loss_weights = []
    loss_weights.append(config["main"]['loss_weights']["ri"])
    loss_weights.append(config["main"]['loss_weights']["mag"])
    loss_weights.append(config["main"]['loss_weights']["time"])
    loss_weights.append(config["main"]['loss_weights']["gan"])


    trainer_class = initialize_module(config["trainer"]["path"], initialize=False)
    trainer = trainer_class(
        dist = dist,
        rank = rank,
        resume = resume,
        n_gpus = world_size,
        epochs = epochs,
        batch_size = batch_size,
        model = model,
        discriminator = model_discriminator,
        train_ds = train_ds,
        test_ds = test_ds,
        scheduler_D = scheduler_D,
        scheduler_G = scheduler_G,
        optimizer = optimizer,
        optimizer_disc = optimizer_disc,
        
        loss_weights = loss_weights,

        hop = hop,
        n_fft = n_fft,
        
        scaler = scaler,
        use_amp = use_amp,
        interval_eval = interval_eval,
        max_clip_grad_norm = max_clip_grad_norm,
        save_model_dir = save_model_dir,
        data_test_dir = data_test_dir,
        tsb_writer = writer,
        num_prints = num_prints,
        logger = logger
    )

    trainer.train()

    cleanup()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=False, 
                                        type=str, 
                                        help="config file path (defaul: None)", 
                                        default="/home/minhkhanh/Desktop/work/denoiser/CMGAN/src/config.toml")

    args = parser.parse_args()
    config = toml.load(args.config)

    print(args)
    available_gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print(available_gpus)
    args.n_gpus = len(available_gpus)
    

    mp.spawn(entry,
             args=(args.n_gpus, config),
             nprocs=args.n_gpus,
             join=True)
