import copy
import json
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import DRTS
from utils.metrics import metric
from utils.tools import adjust_learning_rate, visual

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    @staticmethod
    def _unwrap_model(model):
        return model.module if isinstance(model, nn.DataParallel) else model

    def _build_model(self):
        model = DRTS.Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        train_data, _ = self._get_data(flag='train')
        vali_data, _ = self._get_data(flag='val')
        test_data, _ = self._get_data(flag='test')

        print('[Retrieval] preparing DRTS component retrieval banks.')
        self._unwrap_model(model).prepare_dataset(train_data, vali_data, test_data)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    @staticmethod
    def _select_criterion():
        return nn.MSELoss()

    def _forward_model(self, batch_x, index, mode):
        return self.model(batch_x, index, mode=mode)

    def _train_base_only_for_benefit_scorer(self, train_loader):
        warmup_epochs = int(getattr(self.args, 'benefit_base_epochs', 0))
        if warmup_epochs <= 0:
            print('[BenefitScorer] skipping base-only warmup.')
            return

        print(f'[BenefitScorer] base-only warmup for {warmup_epochs} epoch(s).')
        model = self._unwrap_model(self.model)
        model.train()
        params = list(model.linear_x.parameters())
        optimizer = optim.Adam(params, lr=self.args.learning_rate)
        criterion = self._select_criterion()

        for epoch in range(warmup_epochs):
            losses = []
            for _, batch_x, batch_y, _, _ in train_loader:
                optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                f_dim = -1 if self.args.features == 'MS' else 0
                target = batch_y[:, -self.args.pred_len:, f_dim:]
                outputs = model.forward_base(batch_x)[:, -self.args.pred_len:, f_dim:]
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            avg_loss = float(np.average(losses)) if len(losses) > 0 else 0.0
            print(f'[BenefitScorer] warmup epoch {epoch + 1}/{warmup_epochs} loss: {avg_loss:.7f}')

    def train(self, setting):
        _, train_loader = self._get_data(flag='train')
        _, vali_loader = self._get_data(flag='val')
        _, test_loader = self._get_data(flag='test')

        if int(getattr(self.args, 'use_benefit_scorer', 0)):
            self._train_base_only_for_benefit_scorer(train_loader)
            self._unwrap_model(self.model).fit_benefit_scorer(device=self.device)

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        best_valid_loss = float('inf')
        best_model = None

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (index, batch_x, batch_y, _, _) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                f_dim = -1 if self.args.features == 'MS' else 0
                target = batch_y[:, -self.args.pred_len:, f_dim:]

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_model(batch_x, index, mode='train')
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        loss = criterion(outputs, target)
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    outputs = self._forward_model(batch_x, index, mode='train')
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    loss = criterion(outputs, target)
                    loss.backward()
                    model_optim.step()

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion, mode='valid')
            test_loss = self.vali(test_loader, criterion, mode='test')
            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | "
                f"Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}"
            )

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            if vali_loss < best_valid_loss:
                best_model = copy.deepcopy(self.model)
                best_valid_loss = vali_loss

        best_model_path = os.path.join(path, 'checkpoint.pth')
        if best_model is None:
            best_model = copy.deepcopy(self.model)
        torch.save(best_model.state_dict(), best_model_path)
        return best_model

    def vali(self, vali_loader, criterion, mode='valid'):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for index, batch_x, batch_y, _, _ in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self._forward_model(batch_x, index, mode=mode)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                target = batch_y[:, -self.args.pred_len:, f_dim:]
                loss = criterion(outputs, target)
                total_loss.append(float(loss.item()))

        total_loss = float(np.average(total_loss)) if len(total_loss) > 0 else 0.0
        self.model.train()
        return total_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        drts_model = self._unwrap_model(self.model)

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            train_data, _ = self._get_data(flag='train')
            vali_data, _ = self._get_data(flag='val')
            drts_model.prepare_dataset(train_data, vali_data, test_data)
            if int(getattr(self.args, 'use_benefit_scorer', 0)):
                drts_model.fit_benefit_scorer(device=self.device)

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (index, batch_x, batch_y, _, _) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self._forward_model(batch_x, index, mode='test')
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :]

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                f_dim = -1 if self.args.features == 'MS' else 0
                pred = outputs[:, :, f_dim:]
                true = batch_y[:, :, f_dim:]
                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    input_series = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input_series.shape
                        input_series = test_data.inverse_transform(
                            input_series.reshape(shape[0] * shape[1], -1)
                        ).reshape(shape)
                    gt = np.concatenate((input_series[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input_series[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        folder_path = './results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'mse:{mse}, mae:{mae}')
        with open("result_long_term_forecast.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write(f'mse:{mse}, mae:{mae}')
            f.write('\n\n')

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        with open(os.path.join(folder_path, 'metrics_summary.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'setting': setting,
                'method': 'DRTS',
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'mspe': float(mspe),
            }, f, indent=2, ensure_ascii=False)
