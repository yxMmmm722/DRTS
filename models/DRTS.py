import torch
import torch.nn as nn

from layers.Retrieval import RetrievalTool


class Model(nn.Module):
    """
    DRTS forecasting model.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.device = torch.device(
            f'cuda:{configs.gpu}' if (getattr(configs, 'use_gpu', False) and torch.cuda.is_available()) else 'cpu'
        )
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in

        self.linear_x = nn.Linear(self.seq_len, self.pred_len)

        self.n_period = configs.n_period

        self._rt_kwargs = dict(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            channels=self.channels,
            n_period=self.n_period,
            topm=configs.topm,
            decomp_kernel=getattr(configs, 'decomp_kernel', 25),
            seasonal_kernel=getattr(configs, 'seasonal_kernel', 8),
            stl_period=getattr(configs, 'stl_period', 0),
            component_weights=getattr(configs, 'component_weights', '0.4,0.4,0.2'),
            spatial_top_m=getattr(configs, 'spatial_top_m', 3),
            retrieval_batch_size=getattr(configs, 'retrieval_batch_size', 0),
            component_lazy_full_series=getattr(configs, 'component_lazy_full_series', 0),
            use_benefit_scorer=getattr(configs, 'use_benefit_scorer', 0),
            benefit_scorer_batch_size=getattr(configs, 'benefit_scorer_batch_size', 0),
            benefit_scorer_ridge=getattr(configs, 'benefit_scorer_ridge', 1e-3),
            benefit_scorer_max_batches=getattr(configs, 'benefit_scorer_max_batches', 0),
            retrieval_diversity_delta=getattr(configs, 'retrieval_diversity_delta', 0),
            retrieval_diversity_mult=getattr(configs, 'retrieval_diversity_mult', 3),
        )
        self.rt = self._build_retrieval_tool()
        self.use_benefit_scorer = bool(int(getattr(configs, 'use_benefit_scorer', 0)))
        self._prepared_datasets = None

        self.period_num = self.rt.period_num[-1 * self.n_period:]
        self.retrieval_pred = nn.ModuleList([
            nn.Linear(self.pred_len // g, self.pred_len)
            for g in self.period_num
        ])
        self.linear_pred = nn.Linear(2 * self.pred_len, self.pred_len)

    def _build_retrieval_tool(self):
        return RetrievalTool(**self._rt_kwargs)

    def _base_predict(self, x):
        x_offset = x[:, -1:, :].detach()
        x_norm = x - x_offset
        x_pred_from_x = self.linear_x(x_norm.permute(0, 2, 1)).permute(0, 2, 1)
        return x_pred_from_x, x_offset

    def forward_base(self, x):
        x_pred_from_x, x_offset = self._base_predict(x)
        return x_pred_from_x + x_offset

    def _fuse_with_retrieval(self, x_pred_from_x, x_offset, pred_from_retrieval):
        bsz, _, channels = x_pred_from_x.shape
        retrieval_pred_list = []
        for i, pr in enumerate(pred_from_retrieval):
            assert (bsz, self.pred_len, channels) == pr.shape
            g = self.period_num[i]
            pr = pr.reshape(bsz, self.pred_len // g, g, channels)
            pr = pr[:, :, 0, :]
            pr = self.retrieval_pred[i](pr.permute(0, 2, 1)).permute(0, 2, 1)
            pr = pr.reshape(bsz, self.pred_len, self.channels)
            retrieval_pred_list.append(pr)
        retrieval_pred = torch.stack(retrieval_pred_list, dim=1).sum(dim=1)
        pred = torch.cat([x_pred_from_x, retrieval_pred], dim=1)
        pred = self.linear_pred(pred.permute(0, 2, 1)).permute(0, 2, 1)
        return pred.reshape(bsz, self.pred_len, self.channels) + x_offset

    def prepare_dataset(self, train_data, valid_data, test_data):
        if self.rt is None:
            self.rt = self._build_retrieval_tool()

        print('[Retrieval] method=DRTS component retrieval')
        self.rt.prepare_dataset(train_data, valid_data, test_data)
        self._prepared_datasets = (train_data, valid_data, test_data)
        self.retrieval_dict = {}

        if self.use_benefit_scorer and not self.rt.benefit_scorer_ready:
            print('[BenefitScorer] retrieval precomputation is deferred until scorer fitting.')
            self.retrieval_dict['train'] = None
            self.retrieval_dict['valid'] = None
            self.retrieval_dict['test'] = None
            return

        self.compute_retrievals(train_data, valid_data, test_data)

    def compute_retrievals(self, train_data=None, valid_data=None, test_data=None):
        if train_data is None or valid_data is None or test_data is None:
            if self._prepared_datasets is None:
                raise RuntimeError('Datasets are not prepared. Call prepare_dataset first.')
            train_data, valid_data, test_data = self._prepared_datasets

        print('Doing Train Retrieval')
        train_rt = self.rt.retrieve_all(train_data, train=True, device=self.device, split_name='train')

        print('Doing Valid Retrieval')
        valid_rt = self.rt.retrieve_all(valid_data, train=False, device=self.device, split_name='valid')

        print('Doing Test Retrieval')
        test_rt = self.rt.retrieve_all(test_data, train=False, device=self.device, split_name='test')

        torch.cuda.empty_cache()
        self.retrieval_dict['train'] = train_rt.detach()
        self.retrieval_dict['valid'] = valid_rt.detach()
        self.retrieval_dict['test'] = test_rt.detach()

    def fit_benefit_scorer(self, device=None):
        if self._prepared_datasets is None:
            raise RuntimeError('Datasets are not prepared. Call prepare_dataset first.')
        train_data, valid_data, test_data = self._prepared_datasets
        fit_device = device if device is not None else self.device
        was_training = self.training
        self.eval()
        self.rt.fit_benefit_scorer(
            train_data=train_data,
            base_predict_fn=lambda batch_x: self.forward_base(batch_x),
            device=fit_device,
        )
        if was_training:
            self.train()
        self.compute_retrievals(train_data, valid_data, test_data)

    def encoder(self, x, index, mode):
        bsz, seq_len, channels = x.shape
        assert seq_len == self.seq_len and channels == self.channels

        x_pred_from_x, x_offset = self._base_predict(x)
        retrieval_bank = self.retrieval_dict[mode]
        if retrieval_bank is None:
            raise RuntimeError(f'Retrieval bank for mode={mode} is not prepared.')
        retrieval_index = index.long().to(retrieval_bank.device)
        pred_from_retrieval = retrieval_bank[:, retrieval_index].to(x.device)
        return self._fuse_with_retrieval(x_pred_from_x, x_offset, pred_from_retrieval)

    def forward(self, x_enc, index, mode='train'):
        return self.encoder(x_enc, index, mode)[:, -self.pred_len:, :]
