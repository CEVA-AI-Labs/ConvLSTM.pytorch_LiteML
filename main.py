# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
from  liteml.retrainer import RetrainerModel,RetrainerConfig
import os
import numpy as np
from utils.dataset import MovingMNISTDataset
from networks.ConvLSTM import ConvLSTM
import torch
from torch.utils.data import DataLoader
from utils.utils import save_checkpoint
from utils.utils import build_logging
from utils.functions import train
from utils.functions import valid
from utils.functions import test
#from networks.CrossEntropyLoss import CrossEntropyLoss
from networks.BinaryDiceLoss import BinaryDiceLoss
import argparse
import matplotlib
import os
from torch.utils.tensorboard import SummaryWriter
from clearml import Task

tensorboard_dir = r"data/runs"
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)

writer = SummaryWriter(
    log_dir=tensorboard_dir
)
task = Task.init(
        project_name="Regression tests/QAT" , task_name="conv_lstm_qat"
    )

matplotlib.use('agg')
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='3x3_16_3x3_32_3x3_64')
    parser.add_argument('--qat', type=bool, default=True)
    parser.add_argument('--to_onnx', type=bool, default=False)

    args = parser.parse_args()
    return args

class CustomTracer1(torch.fx.Tracer):


    def create_args_for_root(self, root_fn, is_module, concrete_args=None):
        def input_size():
            return 1, 10, 1, 12, 12
        fn, args = super().create_args_for_root(root_fn, is_module, concrete_args)
        #args[2] = None
        args[1].size = input_size
        return fn, args

    def call_module(self, m, forward, args, kwargs):
        #if isinstance(m, ConvLSTM):
        def input_size():
            return 1, 10, 1, 12, 12
        if len(args) >0 and isinstance(args[0], torch.fx.Proxy):
            args[0].size = input_size
        return super().call_module(m, forward, args, kwargs)

def main():
    args = get_args()
    name = args.config
    QAT = args.qat
    EXPORT_TO_ONNX = args.to_onnx
    print(f"######QAT is set to: {QAT}###########")
    if name == '3x3_16_3x3_32_3x3_64': from configs.config_3x3_16_3x3_32_3x3_64 import config
    elif name == '3x3_32_3x3_64_3x3_128': from configs.config_3x3_32_3x3_64_3x3_128 import config
    valid_dataset = MovingMNISTDataset(config, split='valid')
    valid_loader = DataLoader(valid_dataset, batch_size=config.valid_batch_size,
                              num_workers=config.num_workers, shuffle=False, pin_memory=True)
    key = lambda model,x:model(x[0].float().cuda())
    logger = build_logging(config)
    model = ConvLSTM(config).to(config.device)

    if EXPORT_TO_ONNX:
        torch.onnx.export(model, next(iter(valid_loader))[0].float().cuda(), 'conv_lstm.onnx')
    if QAT:
        cfg = RetrainerConfig("./configs/liteml_config.yaml", custom_tracer = CustomTracer1)
        cfg.optimizations_config["QAT"]['calibration_loader'] = valid_loader
        cfg.optimizations_config["QAT"]['calibration_loader_key'] = key
        model = RetrainerModel(model, cfg)
        model.initialize_quantizers(valid_loader, key=key)
        model = model.to(config.device)

    #criterion = CrossEntropyLoss().to(config.device)
    #criterion = torch.nn.MSELoss().to(config.device)
    criterion = BinaryDiceLoss().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_dataset = MovingMNISTDataset(config, split='train')
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size,
                            num_workers=config.num_workers, shuffle=True, pin_memory=True)

    test_dataset = MovingMNISTDataset(config, split='test')
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size,
                            num_workers=config.num_workers, shuffle=False, pin_memory=True)
    train_records, valid_records, test_records = [], [], []
    best_test_loss = float("inf")
    for epoch in range(config.epochs):
        epoch_records = train(config, logger, epoch, model, train_loader, criterion, optimizer)
        train_loss = np.mean(epoch_records['loss'])
        train_records.append(train_loss)
        epoch_records = valid(config, logger, epoch, model, valid_loader, criterion)
        valid_loss = np.mean(epoch_records['loss'])

        valid_records.append(valid_loss)
        epoch_records = test(config, logger, epoch, model, test_loader, criterion)
        test_loss = np.mean(epoch_records['loss'])
        test_records.append(test_loss)

        writer.add_scalar("Train/loss", train_loss, epoch)
        writer.add_scalar("Validation/loss", valid_loss, epoch)
        writer.add_scalar("Test/loss", test_loss, epoch)


        test_loss = np.mean(epoch_records['loss'])
        plt.plot(range(epoch + 1), train_records, label='train')
        plt.plot(range(epoch + 1), valid_records, label='valid')
        plt.plot(range(epoch + 1), test_records, label='test')
        plt.legend()
        plt.savefig(os.path.join(config.output_dir, '{}.png'.format(name)))
        plt.close()
        save_checkpoint(model.state_dict(), is_best=test_loss < best_test_loss, filename= f"checkpoint{epoch}.pth.tar")
        if test_loss < best_test_loss:
            best_test_loss = test_loss

if __name__ == '__main__':
    main()
