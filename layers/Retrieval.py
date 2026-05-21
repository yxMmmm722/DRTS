import copy

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from layers.ComponentDecomposition import decompose_series


class RetrievalTool():
    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        n_period=3,
        topm=20,
        decomp_kernel=25,
        seasonal_kernel=8,
        stl_period=0,
        component_weights='0.4,0.4,0.2',
        spatial_top_m=3,
        retrieval_batch_size=0,
        component_lazy_full_series=0,
        use_benefit_scorer=0,
        benefit_scorer_batch_size=0,
        benefit_scorer_ridge=1e-3,
        benefit_scorer_max_batches=0,
        retrieval_diversity_delta=0,
        retrieval_diversity_mult=3,
    ):
        period_num = [16, 8, 4, 2, 1]
        period_num = period_num[-1 * n_period:]
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        
        self.n_period = n_period
        self.period_num = sorted(period_num, reverse=True)
        self.topm = topm

        self.decomp_kernel = decomp_kernel
        self.seasonal_kernel = seasonal_kernel
        try:
            parsed_stl_period = int(stl_period)
        except (TypeError, ValueError):
            parsed_stl_period = 0
        self.stl_period = parsed_stl_period if parsed_stl_period > 0 else None
        self.component_weights = self._parse_component_weights(component_weights)
        self.spatial_top_m = spatial_top_m
        self.retrieval_batch_size = int(retrieval_batch_size) if retrieval_batch_size else 0
        self.component_lazy_full_series = bool(int(component_lazy_full_series))

        self.use_benefit_scorer = bool(int(use_benefit_scorer))
        self.benefit_scorer_batch_size = int(benefit_scorer_batch_size) if benefit_scorer_batch_size else 0
        self.benefit_scorer_ridge = float(benefit_scorer_ridge)
        self.benefit_scorer_max_batches = int(benefit_scorer_max_batches) if benefit_scorer_max_batches else 0
        self.benefit_feature_dim = 8
        self.retrieval_diversity_delta = max(0, int(retrieval_diversity_delta))
        self.retrieval_diversity_mult = max(1, int(retrieval_diversity_mult))
        self.benefit_coef = None
        self.benefit_bias = None
        self.benefit_scorer_ready = False

        self.n_train = 0
        self.train_component_mg = None
        self.future_component_mg = None
        self.query_component_mg_by_split = {}
        self.full_mg_by_split = {}
        self.component_full_mg_by_split = {}

    @staticmethod
    def _parse_component_weights(component_weights):
        if isinstance(component_weights, str):
            parts = component_weights.split(',')
        elif isinstance(component_weights, (list, tuple)):
            parts = component_weights
        else:
            parts = [1.0, 1.0, 1.0]

        values = []
        for p in parts:
            try:
                values.append(float(p))
            except (TypeError, ValueError):
                continue

        if len(values) != 3:
            values = [1.0, 1.0, 1.0]
        if abs(values[0]) + abs(values[1]) + abs(values[2]) <= 1e-12:
            values = [1.0, 1.0, 1.0]
        return values
        
    def prepare_dataset(self, train_data, valid_data=None, test_data=None):
        self.n_train = len(train_data)
        print(f'[STL] building component banks: period={self.stl_period}')
        self._prepare_full_series_component_caches(train_data, valid_data, test_data)

    def decompose_mg(self, data_all, remove_offset=True):
        data_all = copy.deepcopy(data_all) # T, S, C

        mg = []
        for g in self.period_num:
            cur = data_all.unfold(dimension=1, size=g, step=g).mean(dim=-1)
            cur = cur.repeat_interleave(repeats=g, dim=1)
            
            mg.append(cur)
#             data_all = data_all - cur
            
        mg = torch.stack(mg, dim=0) # G, T, S, C

        if remove_offset:
            offset = []
            for i, data_p in enumerate(mg):
                cur_offset = data_p[:,-1:,:]
                mg[i] = data_p - cur_offset
                offset.append(cur_offset)
            offset = torch.stack(offset, dim=0)
        else:
            offset = None
            
        return mg, offset

    def _build_full_series_mg(self, data):
        if not torch.is_tensor(data):
            data = torch.tensor(data)
        data = data.float()
        if data.dim() != 2:
            raise ValueError(f'Full-series component cache expects [T, C], got {tuple(data.shape)}')

        series_len, channels = data.shape
        mg = []
        for g in self.period_num:
            if g <= 1:
                cur = data
            else:
                n_full = series_len // g
                full_len = n_full * g
                pieces = []
                if full_len > 0:
                    head = data[:full_len].reshape(n_full, g, channels).mean(dim=1)
                    pieces.append(head.repeat_interleave(repeats=g, dim=0))
                if full_len < series_len:
                    tail = data[full_len:].mean(dim=0, keepdim=True)
                    pieces.append(tail.repeat(series_len - full_len, 1))
                cur = torch.cat(pieces, dim=0) if len(pieces) > 0 else data
            mg.append(cur)
        return torch.stack(mg, dim=0)

    def _decompose_full_series_component_mg(self, full_mg, split_name):
        trend_bank = []
        seasonal_bank = []
        residual_bank = []
        for i in range(full_mg.shape[0]):
            trend, seasonal, residual = decompose_series(
                full_mg[i].unsqueeze(0),
                trend_kernel=self.decomp_kernel,
                seasonal_kernel=self.seasonal_kernel,
                stl_period=self.stl_period,
                show_progress=True,
                progress_desc=f'STL full {split_name} scale {i + 1}/{full_mg.shape[0]}',
            )
            trend_bank.append(trend.squeeze(0))
            seasonal_bank.append(seasonal.squeeze(0))
            residual_bank.append(residual.squeeze(0))
        return {
            'trend': torch.stack(trend_bank, dim=0),
            'seasonal': torch.stack(seasonal_bank, dim=0),
            'residual': torch.stack(residual_bank, dim=0),
        }

    def _slice_component_windows(self, component_full_mg, full_mg, starts, length):
        starts = torch.as_tensor(starts, dtype=torch.long)
        time_index = starts.unsqueeze(1) + torch.arange(length, dtype=torch.long).unsqueeze(0)
        component_windows = {}
        for comp_name, comp_full in component_full_mg.items():
            scale_windows = []
            for scale_i in range(comp_full.shape[0]):
                window = comp_full[scale_i][time_index].clone()
                if comp_name == 'trend':
                    offset = full_mg[scale_i, starts + length - 1].unsqueeze(1)
                    window = window - offset
                scale_windows.append(torch.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0))
            component_windows[comp_name] = torch.stack(scale_windows, dim=0)
        return component_windows

    def _build_split_component_windows(self, data, split_name, include_future=False):
        if data is None:
            return None, None
        if not hasattr(data, 'data_x'):
            raise RuntimeError('Full-series STL component cache requires dataset.data_x.')

        full_series = torch.tensor(np.asarray(data.data_x), dtype=torch.float32)
        full_mg = self._build_full_series_mg(full_series)
        component_full_mg = self._decompose_full_series_component_mg(full_mg, split_name=split_name)
        self.full_mg_by_split[split_name] = full_mg
        self.component_full_mg_by_split[split_name] = component_full_mg

        starts = torch.arange(len(data), dtype=torch.long)
        query_components = self._slice_component_windows(
            component_full_mg=component_full_mg,
            full_mg=full_mg,
            starts=starts,
            length=self.seq_len,
        )

        future_components = None
        if include_future:
            future_components = self._slice_component_windows(
                component_full_mg=component_full_mg,
                full_mg=full_mg,
                starts=starts + self.seq_len,
                length=self.pred_len,
            )
        return query_components, future_components

    def _build_split_component_full_cache(self, data, split_name):
        if data is None:
            return
        if not hasattr(data, 'data_x'):
            raise RuntimeError('Full-series STL component cache requires dataset.data_x.')

        full_series = torch.tensor(np.asarray(data.data_x), dtype=torch.float32)
        full_mg = self._build_full_series_mg(full_series)
        component_full_mg = self._decompose_full_series_component_mg(full_mg, split_name=split_name)
        self.full_mg_by_split[split_name] = full_mg
        self.component_full_mg_by_split[split_name] = component_full_mg

    def _slice_single_component_window(self, component_full_mg, full_mg, comp_name, scale_i, starts, length):
        starts = torch.as_tensor(starts, dtype=torch.long)
        time_index = starts.unsqueeze(1) + torch.arange(length, dtype=torch.long).unsqueeze(0)
        window = component_full_mg[comp_name][scale_i][time_index].clone()
        if comp_name == 'trend':
            offset = full_mg[scale_i, starts + length - 1].unsqueeze(1)
            window = window - offset
        return torch.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)

    def _slice_lazy_query_components(self, split_name, index, scale_i, comp_order, device):
        if split_name not in self.component_full_mg_by_split:
            raise RuntimeError(f'Lazy component cache for split={split_name} is not prepared.')
        index_cpu = index.detach().cpu().long()
        component_full_mg = self.component_full_mg_by_split[split_name]
        full_mg = self.full_mg_by_split[split_name]
        return {
            comp_name: self._slice_single_component_window(
                component_full_mg=component_full_mg,
                full_mg=full_mg,
                comp_name=comp_name,
                scale_i=scale_i,
                starts=index_cpu,
                length=self.seq_len,
            ).to(device)
            for comp_name in comp_order
        }

    def _component_channelwise_reference_from_full(self, comp_name, scale_i, topk_idx, device, weights=None):
        component_full_mg = self.component_full_mg_by_split['train']
        full_mg = self.full_mg_by_split['train']
        comp_full = component_full_mg[comp_name][scale_i]
        full_scale = full_mg[scale_i]

        bsz, topk, channels = topk_idx.shape
        ref = torch.zeros((bsz, self.pred_len, channels), device=device, dtype=comp_full.dtype)
        pred_offsets = torch.arange(self.pred_len, dtype=torch.long)
        topk_cpu = topk_idx.detach().cpu().long()

        for c in range(channels):
            future_starts = topk_cpu[:, :, c] + self.seq_len
            time_index = future_starts.unsqueeze(-1) + pred_offsets.view(1, 1, -1)
            cand_c = comp_full[:, c][time_index]  # [B, K, P]
            if comp_name == 'trend':
                offset_idx = future_starts + self.pred_len - 1
                cand_c = cand_c - full_scale[:, c][offset_idx].unsqueeze(-1)
            if weights is None:
                ref[:, :, c] = cand_c.mean(dim=1).to(device)
            else:
                w_c = weights[:, :, c].detach().cpu().to(dtype=cand_c.dtype)
                ref[:, :, c] = (cand_c * w_c.unsqueeze(-1)).sum(dim=1).to(device)
        return torch.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0)

    def _prepare_full_series_component_caches(self, train_data, valid_data=None, test_data=None):
        self.query_component_mg_by_split = {}
        self.full_mg_by_split = {}
        self.component_full_mg_by_split = {}

        if self.component_lazy_full_series:
            print('[STL] lazy full-series component windows enabled.')
            self.train_component_mg = None
            self.future_component_mg = None
            self._build_split_component_full_cache(train_data, split_name='train')
            self._build_split_component_full_cache(valid_data, split_name='valid')
            self._build_split_component_full_cache(test_data, split_name='test')
            return

        train_query, train_future = self._build_split_component_windows(
            train_data,
            split_name='train',
            include_future=True,
        )
        self.train_component_mg = train_query
        self.future_component_mg = train_future
        self.query_component_mg_by_split['train'] = train_query

        if valid_data is not None:
            valid_query, _ = self._build_split_component_windows(
                valid_data,
                split_name='valid',
                include_future=False,
            )
            self.query_component_mg_by_split['valid'] = valid_query

        if test_data is not None:
            test_query, _ = self._build_split_component_windows(
                test_data,
                split_name='test',
                include_future=False,
            )
            self.query_component_mg_by_split['test'] = test_query
    
    def _build_train_self_mask(self, index, bsz, device):
        sliding_index = torch.arange(2 * (self.seq_len + self.pred_len) - 1).to(device)
        sliding_index = sliding_index.unsqueeze(dim=0).repeat(len(index), 1)
        sliding_index = sliding_index + (index - self.seq_len - self.pred_len + 1).unsqueeze(dim=1)

        sliding_index = torch.where(sliding_index >= 0, sliding_index, 0)
        sliding_index = torch.where(sliding_index < self.n_train, sliding_index, self.n_train - 1)

        self_mask = torch.zeros((bsz, self.n_train)).to(device)
        self_mask = self_mask.scatter_(1, sliding_index, 1.)
        return self_mask.bool()

    def _diverse_topk(self, score, topk):
        if self.retrieval_diversity_delta <= 0:
            return torch.topk(score, topk, dim=1, largest=False)

        bsz, n_hist, channels = score.shape
        recall_k = min(n_hist, max(topk, topk * self.retrieval_diversity_mult))
        cand_values, cand_idx = torch.topk(score, recall_k, dim=1, largest=False)

        cand_values_cpu = cand_values.detach().cpu()
        cand_idx_cpu = cand_idx.detach().cpu()
        selected_idx_cpu = torch.empty((bsz, topk, channels), dtype=torch.long)
        selected_values_cpu = torch.empty((bsz, topk, channels), dtype=cand_values_cpu.dtype)
        delta = int(self.retrieval_diversity_delta)

        for b in range(bsz):
            for c in range(channels):
                chosen = []
                chosen_values = []
                used = set()
                for r in range(recall_k):
                    idx = int(cand_idx_cpu[b, r, c])
                    if all(abs(idx - prev) > delta for prev in chosen):
                        chosen.append(idx)
                        chosen_values.append(cand_values_cpu[b, r, c])
                        used.add(r)
                        if len(chosen) == topk:
                            break

                if len(chosen) < topk:
                    for r in range(recall_k):
                        if r in used:
                            continue
                        chosen.append(int(cand_idx_cpu[b, r, c]))
                        chosen_values.append(cand_values_cpu[b, r, c])
                        if len(chosen) == topk:
                            break

                selected_idx_cpu[b, :, c] = torch.tensor(chosen[:topk], dtype=torch.long)
                selected_values_cpu[b, :, c] = torch.stack(chosen_values[:topk])

        return selected_values_cpu.to(score.device), selected_idx_cpu.to(score.device)

    def _keep_k_for_diversity(self, topk):
        if self.retrieval_diversity_delta <= 0:
            return topk
        return min(self.n_train, max(topk, topk * self.retrieval_diversity_mult))

    def _benefit_features(self, topk_values, topk_idx, index, comp_idx, scale_i):
        bsz, topk, channels = topk_idx.shape
        device = topk_idx.device
        dtype = topk_values.dtype

        dist = torch.nan_to_num(topk_values, nan=1e6, posinf=1e6, neginf=1e6).clamp_min(0.0)
        log_dist = torch.log1p(dist)
        inv_dist = 1.0 / (1.0 + dist)

        if topk > 1:
            rank = torch.arange(topk, device=device, dtype=dtype) / float(topk - 1)
        else:
            rank = torch.zeros(topk, device=device, dtype=dtype)
        rank = rank.view(1, topk, 1).expand(bsz, topk, channels)

        dist_mean = dist.mean(dim=1, keepdim=True)
        dist_std = torch.sqrt(dist.var(dim=1, keepdim=True, unbiased=False) + 1e-6)
        rel_dist = (dist - dist_mean) / dist_std

        best_dist = dist.min(dim=1, keepdim=True).values
        best_gap = torch.log1p((dist - best_dist).clamp_min(0.0))

        comp_feats = []
        for i in range(3):
            comp_feats.append(torch.full_like(dist, 1.0 if comp_idx == i else 0.0))

        return torch.stack([log_dist, inv_dist, rank, rel_dist, best_gap] + comp_feats, dim=-1)

    def _score_benefit_candidates(self, features):
        if not self.use_benefit_scorer or not self.benefit_scorer_ready:
            return None
        coef = self.benefit_coef.to(device=features.device, dtype=features.dtype)
        bias = self.benefit_bias.to(device=features.device, dtype=features.dtype)
        return torch.einsum('bkcf,f->bkc', features, coef) + bias

    def _candidate_weights(self, topk_values, topk_idx, index, comp_idx, scale_i):
        features = self._benefit_features(
            topk_values=topk_values,
            topk_idx=topk_idx,
            index=index,
            comp_idx=comp_idx,
            scale_i=scale_i,
        )
        scores = self._score_benefit_candidates(features)
        if scores is None:
            return None
        scores = torch.nan_to_num(scores, nan=-1e6, posinf=1e6, neginf=-1e6)
        return torch.softmax(scores, dim=1)

    def _raw_candidate_mse_by_channel_from_full(self, scale_i, topk_idx, target, device):
        full_mg = self.full_mg_by_split['train']
        full_scale = full_mg[scale_i]
        bsz, topk, channels = topk_idx.shape
        pred_offsets = torch.arange(self.pred_len, dtype=torch.long)
        topk_cpu = topk_idx.detach().cpu().long()
        target_cpu = target.detach().cpu()

        mse = torch.zeros((bsz, topk, channels), dtype=torch.float32)
        for c in range(channels):
            future_starts = topk_cpu[:, :, c] + self.seq_len
            time_index = future_starts.unsqueeze(-1) + pred_offsets.view(1, 1, -1)
            cand_c = full_scale[:, c][time_index]  # [B, K, P]
            target_c = target_cpu[:, :, c].unsqueeze(1)
            mse[:, :, c] = (cand_c - target_c).pow(2).mean(dim=2)
        return torch.nan_to_num(mse, nan=1e6, posinf=1e6, neginf=1e6).to(device)

    def fit_benefit_scorer(self, train_data, base_predict_fn, device=torch.device('cpu')):
        if not self.use_benefit_scorer:
            return
        if 'train' not in self.full_mg_by_split:
            raise RuntimeError('Training full-series cache is required before fitting benefit scorer.')

        print('[BenefitScorer] fitting supervised candidate scorer.')
        feature_dim = self.benefit_feature_dim
        xtx = torch.zeros((feature_dim + 1, feature_dim + 1), dtype=torch.float64)
        xty = torch.zeros((feature_dim + 1,), dtype=torch.float64)
        n_examples = 0

        if self.benefit_scorer_batch_size > 0:
            fit_batch_size = min(self.benefit_scorer_batch_size, len(train_data))
        elif self.retrieval_batch_size > 0:
            fit_batch_size = min(self.retrieval_batch_size, len(train_data))
        else:
            fit_batch_size = min(128, len(train_data))

        loader = DataLoader(
            train_data,
            batch_size=fit_batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        comp_order = ['trend', 'seasonal', 'residual']
        max_flat_examples = 200000
        with torch.no_grad():
            for batch_id, (index, batch_x, batch_y, _, _) in enumerate(tqdm(loader, desc='BenefitScorer', unit='batch')):
                if self.benefit_scorer_max_batches > 0 and batch_id >= self.benefit_scorer_max_batches:
                    break

                batch_x = batch_x.float().to(device)
                target = batch_y.float().to(device)
                base_pred = base_predict_fn(batch_x).detach()
                base_mse = (base_pred - target).pow(2).mean(dim=1)  # [B, C]
                index = index.to(device)
                bsz = batch_x.shape[0]
                self_mask = self._build_train_self_mask(index=index, bsz=bsz, device=device)

                if self.component_lazy_full_series:
                    query_cache = None
                else:
                    query_cache = self.query_component_mg_by_split.get('train')
                    if query_cache is None:
                        x_mg, _ = self.decompose_mg(batch_x)

                for scale_i in range(self.n_period):
                    if self.component_lazy_full_series:
                        query_components = self._slice_lazy_query_components(
                            split_name='train',
                            index=index,
                            scale_i=scale_i,
                            comp_order=comp_order,
                            device=device,
                        )
                    elif query_cache is not None:
                        index_cpu = index.detach().cpu().long()
                        query_components = {
                            comp_name: query_cache[comp_name][scale_i, index_cpu].to(device)
                            for comp_name in comp_order
                        }
                    else:
                        q_scale = x_mg[scale_i]
                        q_trend, q_seasonal, q_residual = decompose_series(
                            q_scale,
                            trend_kernel=self.decomp_kernel,
                            seasonal_kernel=self.seasonal_kernel,
                            stl_period=self.stl_period,
                        )
                        query_components = {'trend': q_trend, 'seasonal': q_seasonal, 'residual': q_residual}

                    for comp_idx, comp_name in enumerate(comp_order):
                        q_comp = query_components[comp_name]
                        graph = self.build_dynamic_spatial_graph(q_comp, top_m=self.spatial_top_m)
                        safe_topk = max(1, min(self.topm, self.n_train))
                        keep_topk = self._keep_k_for_diversity(safe_topk)

                        if self.component_lazy_full_series:
                            topk_values = None
                            topk_idx = None
                            train_component_full_mg = self.component_full_mg_by_split['train']
                            train_full_mg = self.full_mg_by_split['train']
                            lazy_history_chunk = 128
                            for h_start in range(0, self.n_train, lazy_history_chunk):
                                h_end = min(h_start + lazy_history_chunk, self.n_train)
                                h_starts = torch.arange(h_start, h_end, dtype=torch.long)
                                h_comp = self._slice_single_component_window(
                                    component_full_mg=train_component_full_mg,
                                    full_mg=train_full_mg,
                                    comp_name=comp_name,
                                    scale_i=scale_i,
                                    starts=h_starts,
                                    length=self.seq_len,
                                ).to(device)
                                dist_by_channel = self._distance_per_channel(
                                    q=q_comp,
                                    h=h_comp,
                                    metric=self._metric_for_component(comp_name),
                                )
                                score = torch.einsum('bcj,bnj->bnc', graph, dist_by_channel)
                                score = torch.nan_to_num(score, nan=1e6, posinf=1e6, neginf=1e6)
                                score = score.masked_fill(self_mask[:, h_start:h_end].unsqueeze(-1), float('inf'))

                                chunk_idx = torch.arange(h_start, h_end, device=device, dtype=torch.long)
                                chunk_idx = chunk_idx.view(1, -1, 1).expand(bsz, -1, self.channels)
                                if topk_values is None:
                                    cand_values = score
                                    cand_idx = chunk_idx
                                else:
                                    cand_values = torch.cat([topk_values, score], dim=1)
                                    cand_idx = torch.cat([topk_idx, chunk_idx], dim=1)
                                cur_topk = min(keep_topk, cand_values.shape[1])
                                topk_values, topk_pos = torch.topk(cand_values, cur_topk, dim=1, largest=False)
                                topk_idx = torch.gather(cand_idx, 1, topk_pos)
                            topk_values, topk_idx = self._diverse_topk(topk_values, safe_topk)
                        else:
                            h_comp = self.train_component_mg[comp_name][scale_i].to(device)
                            dist_by_channel = self._distance_per_channel(
                                q=q_comp,
                                h=h_comp,
                                metric=self._metric_for_component(comp_name),
                            )
                            score = torch.einsum('bcj,bnj->bnc', graph, dist_by_channel)
                            score = torch.nan_to_num(score, nan=1e6, posinf=1e6, neginf=1e6)
                            score = score.masked_fill(self_mask.unsqueeze(-1), float('inf'))
                            topk_values, topk_idx = self._diverse_topk(score, safe_topk)

                        features = self._benefit_features(
                            topk_values=topk_values,
                            topk_idx=topk_idx,
                            index=index,
                            comp_idx=comp_idx,
                            scale_i=scale_i,
                        )
                        cand_mse = self._raw_candidate_mse_by_channel_from_full(
                            scale_i=scale_i,
                            topk_idx=topk_idx,
                            target=target,
                            device=device,
                        )
                        benefit = base_mse.unsqueeze(1) - cand_mse
                        benefit = benefit - benefit.mean(dim=1, keepdim=True)

                        x_flat = features.reshape(-1, feature_dim)
                        y_flat = benefit.reshape(-1)
                        finite = torch.isfinite(x_flat).all(dim=1) & torch.isfinite(y_flat)
                        x_flat = x_flat[finite]
                        y_flat = y_flat[finite]
                        if x_flat.numel() == 0:
                            continue
                        if x_flat.shape[0] > max_flat_examples:
                            pick = torch.linspace(
                                0,
                                x_flat.shape[0] - 1,
                                steps=max_flat_examples,
                                device=x_flat.device,
                            ).long()
                            x_flat = x_flat[pick]
                            y_flat = y_flat[pick]

                        ones = torch.ones((x_flat.shape[0], 1), device=x_flat.device, dtype=x_flat.dtype)
                        design = torch.cat([x_flat, ones], dim=1).double().cpu()
                        labels = y_flat.double().cpu()
                        xtx += design.t().matmul(design)
                        xty += design.t().matmul(labels)
                        n_examples += int(x_flat.shape[0])

        if n_examples == 0:
            print('[BenefitScorer] no finite training examples; falling back to mean top-k retrieval.')
            self.benefit_scorer_ready = False
            return

        reg = torch.eye(feature_dim + 1, dtype=torch.float64) * max(0.0, self.benefit_scorer_ridge)
        reg[-1, -1] = 0.0
        coef = torch.linalg.solve(xtx + reg, xty)
        self.benefit_coef = coef[:-1].float()
        self.benefit_bias = coef[-1:].float().squeeze(0)
        self.benefit_scorer_ready = True
        print(f'[BenefitScorer] fitted on {n_examples} candidate examples.')
        
    def retrieve(self, x, index, train=True, split_name='unknown'):
        return self._retrieve_component(x=x, index=index, train=train, split_name=split_name)

    @staticmethod
    def _metric_for_component(comp_name):
        return {
            'trend': 'trend',
            'seasonal': 'correlation',
            'residual': 'huber',
        }[comp_name]

    def _distance_per_channel(self, q, h, metric='mse'):
        # q: [B, L, C], h: [N, L, C] -> [B, N, C]
        bsz, seq_len, channels = q.shape
        n_hist = h.shape[0]
        dist_chunks = []

        if metric == 'mse':
            # Memory-safe MSE:
            # mse(q, h) = mean(q^2) + mean(h^2) - 2 * mean(q*h)
            q2 = q.pow(2).mean(dim=1, keepdim=True)  # [B, 1, C]
            chunk_size = min(n_hist, 1024)
            for start in range(0, n_hist, chunk_size):
                end = min(start + chunk_size, n_hist)
                h_chunk = h[start:end]  # [Nc, L, C]
                h2 = h_chunk.pow(2).mean(dim=1).unsqueeze(0)  # [1, Nc, C]
                cross = torch.einsum('blc,nlc->bnc', q, h_chunk) / float(max(1, seq_len))  # [B, Nc, C]
                dist_chunk = q2 + h2 - 2.0 * cross
                dist_chunks.append(torch.clamp(dist_chunk, min=0.0))

        elif metric == 'trend':
            value_dist = self._distance_per_channel(q, h, metric='mse')
            if seq_len <= 1:
                return value_dist
            slope_q = q[:, 1:, :] - q[:, :-1, :]
            slope_h = h[:, 1:, :] - h[:, :-1, :]
            slope_dist = self._distance_per_channel(slope_q, slope_h, metric='mse')
            return value_dist + 0.5 * slope_dist

        elif metric == 'huber':
            # Huber is more robust for residual spikes than MSE.
            delta = 1.0
            chunk_size = max(1, min(n_hist, 64))
            for start in range(0, n_hist, chunk_size):
                end = min(start + chunk_size, n_hist)
                h_chunk = h[start:end]  # [Nc, L, C]
                abs_err = (q.unsqueeze(1) - h_chunk.unsqueeze(0)).abs()
                quadratic = torch.clamp(abs_err, max=delta)
                linear = abs_err - quadratic
                dist_chunk = (0.5 * quadratic.pow(2) + delta * linear).mean(dim=2)  # [B, Nc, C]
                dist_chunks.append(dist_chunk)

        elif metric in ['corr', 'correlation']:
            eps = 1e-6
            q_center = q - q.mean(dim=1, keepdim=True)
            q_norm = q_center / torch.sqrt(q_center.pow(2).mean(dim=1, keepdim=True) + eps)
            chunk_size = min(n_hist, 1024)
            for start in range(0, n_hist, chunk_size):
                end = min(start + chunk_size, n_hist)
                h_chunk = h[start:end]
                h_center = h_chunk - h_chunk.mean(dim=1, keepdim=True)
                h_norm = h_center / torch.sqrt(h_center.pow(2).mean(dim=1, keepdim=True) + eps)
                corr_chunk = torch.einsum('blc,nlc->bnc', q_norm, h_norm) / float(max(1, seq_len))
                dist_chunks.append(1.0 - corr_chunk)
        else:
            raise ValueError(f'Unsupported component metric: {metric}')

        dist = torch.cat(dist_chunks, dim=1)
        return torch.nan_to_num(dist, nan=0.0, posinf=1e6, neginf=1e6)

    def build_dynamic_spatial_graph(self, component_x, top_m=3):
        """
        component_x: [B, L, C]
        return: [B, C, C]
        """
        if component_x.dim() != 3:
            raise ValueError(f'build_dynamic_spatial_graph expects [B, L, C], got {tuple(component_x.shape)}')

        bsz, seq_len, channels = component_x.shape
        eps = 1e-6

        centered = component_x - component_x.mean(dim=1, keepdim=True)
        std = torch.sqrt(centered.pow(2).mean(dim=1, keepdim=True) + eps)
        normalized = centered / std
        corr = torch.einsum('blc,bld->bcd', normalized, normalized) / max(1, seq_len)
        corr = torch.clamp(corr, min=0.0)

        eye = torch.eye(channels, device=component_x.device, dtype=component_x.dtype).unsqueeze(0)
        corr = torch.where(eye.bool(), torch.ones_like(corr), corr)

        if top_m < channels:
            keep_k = max(1, top_m)
            top_idx = torch.topk(corr, k=keep_k, dim=2).indices
            keep_mask = torch.zeros_like(corr)
            keep_mask.scatter_(2, top_idx, 1.0)
            corr = corr * keep_mask

        row_sum = corr.sum(dim=2, keepdim=True)
        corr = corr / torch.clamp(row_sum, min=eps)
        corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        return corr

    def _component_channelwise_reference(self, future_bank, topk_idx, weights=None):
        # future_bank: [N, P, C], topk_idx: [B, K, C] -> [B, P, C]
        bsz, topk, channels = topk_idx.shape
        pred_len = future_bank.shape[1]
        ref = torch.zeros((bsz, pred_len, channels), device=future_bank.device, dtype=future_bank.dtype)
        for c in range(channels):
            idx_c = topk_idx[:, :, c]  # [B, K]
            cand_c = future_bank[:, :, c][idx_c]  # [B, K, P]
            if weights is None:
                ref[:, :, c] = cand_c.mean(dim=1)
            else:
                ref[:, :, c] = (cand_c * weights[:, :, c].unsqueeze(-1)).sum(dim=1)
        return ref

    def _retrieve_component(self, x, index, train=True, split_name='unknown'):
        index = index.to(x.device)
        bsz, seq_len, channels = x.shape
        assert seq_len == self.seq_len and channels == self.channels

        if self.component_lazy_full_series:
            if 'train' not in self.component_full_mg_by_split:
                raise RuntimeError('Lazy component full-series banks are not prepared. Call prepare_dataset first.')
        elif self.train_component_mg is None or self.future_component_mg is None:
            raise RuntimeError('Component banks are not prepared. Call prepare_dataset first.')

        query_component_cache = None if self.component_lazy_full_series else self.query_component_mg_by_split.get(split_name)
        x_mg = None
        if query_component_cache is None and not self.component_lazy_full_series:
            x_mg, _ = self.decompose_mg(x)  # [G, B, S, C]
        self_mask = None
        if train:
            self_mask = self._build_train_self_mask(index=index, bsz=bsz, device=x.device)

        weight_trend, weight_seasonal, weight_residual = self.component_weights
        comp_order = ['trend', 'seasonal', 'residual']
        comp_weights = {
            'trend': weight_trend,
            'seasonal': weight_seasonal,
            'residual': weight_residual,
        }

        ref_by_scale = []
        lazy_history_chunk = 128

        for scale_i in range(self.n_period):
            if self.component_lazy_full_series:
                query_components = self._slice_lazy_query_components(
                    split_name=split_name,
                    index=index,
                    scale_i=scale_i,
                    comp_order=comp_order,
                    device=x.device,
                )
            elif query_component_cache is not None:
                index_cpu = index.detach().cpu().long()
                query_components = {
                    comp_name: query_component_cache[comp_name][scale_i, index_cpu].to(x.device)
                    for comp_name in comp_order
                }
            else:
                q_scale = x_mg[scale_i]  # [B, Ls, C]
                q_trend, q_seasonal, q_residual = decompose_series(
                    q_scale,
                    trend_kernel=self.decomp_kernel,
                    seasonal_kernel=self.seasonal_kernel,
                    stl_period=self.stl_period,
                )
                query_components = {
                    'trend': q_trend,
                    'seasonal': q_seasonal,
                    'residual': q_residual,
                }

            ref_components = {}

            for comp_idx, comp_name in enumerate(comp_order):
                q_comp = query_components[comp_name]
                graph = self.build_dynamic_spatial_graph(
                    q_comp,
                    top_m=self.spatial_top_m,
                )  # [B, C, C]
                safe_topk = max(1, min(self.topm, self.n_train))
                keep_topk = self._keep_k_for_diversity(safe_topk)

                if self.component_lazy_full_series:
                    topk_values = None
                    topk_idx = None
                    train_component_full_mg = self.component_full_mg_by_split['train']
                    train_full_mg = self.full_mg_by_split['train']
                    for h_start in range(0, self.n_train, lazy_history_chunk):
                        h_end = min(h_start + lazy_history_chunk, self.n_train)
                        h_starts = torch.arange(h_start, h_end, dtype=torch.long)
                        h_comp = self._slice_single_component_window(
                            component_full_mg=train_component_full_mg,
                            full_mg=train_full_mg,
                            comp_name=comp_name,
                            scale_i=scale_i,
                            starts=h_starts,
                            length=self.seq_len,
                        ).to(x.device)
                        dist_by_channel = self._distance_per_channel(
                            q=q_comp,
                            h=h_comp,
                            metric=self._metric_for_component(comp_name),
                        )  # [B, Nh, C]
                        score = torch.einsum('bcj,bnj->bnc', graph, dist_by_channel)  # [B, Nh, C]
                        score = torch.nan_to_num(score, nan=1e6, posinf=1e6, neginf=1e6)

                        if self_mask is not None:
                            score = score.masked_fill(self_mask[:, h_start:h_end].unsqueeze(-1), float('inf'))

                        chunk_idx = torch.arange(h_start, h_end, device=x.device, dtype=torch.long)
                        chunk_idx = chunk_idx.view(1, -1, 1).expand(bsz, -1, channels)
                        if topk_values is None:
                            cand_values = score
                            cand_idx = chunk_idx
                        else:
                            cand_values = torch.cat([topk_values, score], dim=1)
                            cand_idx = torch.cat([topk_idx, chunk_idx], dim=1)
                        cur_topk = min(keep_topk, cand_values.shape[1])
                        topk_values, topk_pos = torch.topk(cand_values, cur_topk, dim=1, largest=False)
                        topk_idx = torch.gather(cand_idx, 1, topk_pos)
                    topk_values, topk_idx = self._diverse_topk(topk_values, safe_topk)
                else:
                    y_comp = self.future_component_mg[comp_name][scale_i].to(x.device)  # [N, P, C]
                    h_comp = self.train_component_mg[comp_name][scale_i].to(x.device)  # [N, Ls, C]
                    dist_by_channel = self._distance_per_channel(
                        q=q_comp,
                        h=h_comp,
                        metric=self._metric_for_component(comp_name),
                    )  # [B, N, C]
                    score = torch.einsum('bcj,bnj->bnc', graph, dist_by_channel)  # [B, N, C]
                    score = torch.nan_to_num(score, nan=1e6, posinf=1e6, neginf=1e6)

                    # Retrieval score uses only lookback components.
                    if self_mask is not None:
                        score = score.masked_fill(self_mask.unsqueeze(-1), float('inf'))

                    topk_values, topk_idx = self._diverse_topk(score, safe_topk)  # [B,K,C]

                candidate_weights = self._candidate_weights(
                    topk_values=topk_values,
                    topk_idx=topk_idx,
                    index=index,
                    comp_idx=comp_idx,
                    scale_i=scale_i,
                )

                # Future bank is only used after top-k indices are selected.
                if self.component_lazy_full_series:
                    ref_components[comp_name] = self._component_channelwise_reference_from_full(
                        comp_name=comp_name,
                        scale_i=scale_i,
                        topk_idx=topk_idx,
                        device=x.device,
                        weights=candidate_weights,
                    )
                else:
                    ref_components[comp_name] = self._component_channelwise_reference(
                        future_bank=y_comp,
                        topk_idx=topk_idx,
                        weights=candidate_weights,
                    )  # [B, P, C]

            ref_scale = (
                comp_weights['trend'] * ref_components['trend']
                + comp_weights['seasonal'] * ref_components['seasonal']
                + comp_weights['residual'] * ref_components['residual']
            )
            ref_scale = torch.nan_to_num(ref_scale, nan=0.0, posinf=0.0, neginf=0.0)
            ref_by_scale.append(ref_scale)

        pred_from_retrieval = torch.stack(ref_by_scale, dim=0)  # [G,B,P,C]
        pred_from_retrieval = torch.nan_to_num(pred_from_retrieval, nan=0.0, posinf=0.0, neginf=0.0)

        return pred_from_retrieval
    
    def retrieve_all(
        self,
        data,
        train=False,
        device=torch.device('cpu'),
        split_name='unknown',
    ):
        if self.component_lazy_full_series:
            assert 'train' in self.component_full_mg_by_split
        else:
            assert self.train_component_mg is not None and self.future_component_mg is not None
        if self.retrieval_batch_size > 0:
            rt_batch_size = min(self.retrieval_batch_size, len(data))
        else:
            # Component-wise retrieval builds BxN scores; reduce query batch for GPU safety.
            rt_batch_size = min(128, len(data))
        
        rt_loader = DataLoader(
            data,
            batch_size=rt_batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False
        )
        
        retrievals = []
        with torch.no_grad():
            for index, batch_x, _, _, _ in tqdm(rt_loader):
                pred_from_retrieval = self.retrieve(
                    batch_x.float().to(device),
                    index,
                    train=train,
                    split_name=split_name,
                )
                pred_from_retrieval = pred_from_retrieval.cpu()
                retrievals.append(pred_from_retrieval)
                
        return torch.cat(retrievals, dim=1)
