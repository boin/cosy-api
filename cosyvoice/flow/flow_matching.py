# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import threading
import torch
import torch.nn.functional as F
from matcha.models.components.flow_matching import BASECFM


class ConditionalCFM(BASECFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0)
        # Just change the architecture of the estimator here
        self.estimator = estimator
        self.lock = threading.Lock()

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, prompt_len=0, cache=torch.zeros(1, 80, 0, 2)):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """

        z = torch.randn_like(mu).to(mu.device).to(mu.dtype) * temperature
        cache_size = cache.shape[2]
        # fix prompt and overlap part mu and z
        if cache_size != 0:
            z[:, :, :cache_size] = cache[:, :, :, 0]
            mu[:, :, :cache_size] = cache[:, :, :, 1]
        z_cache = torch.concat([z[:, :, :prompt_len], z[:, :, -34:]], dim=2)
        mu_cache = torch.concat([mu[:, :, :prompt_len], mu[:, :, -34:]], dim=2)
        cache = torch.stack([z_cache, mu_cache], dim=-1)

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), cache

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        # Do not use concat, it may cause memory format changed and trt infer with wrong results!
        x_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        mask_in = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=x.dtype)
        mu_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        t_in = torch.zeros([2], device=x.device, dtype=x.dtype)
        spks_in = torch.zeros([2, 80], device=x.device, dtype=x.dtype)
        cond_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        for step in range(1, len(t_span)):
            # Classifier-Free Guidance inference introduced in VoiceBox
            x_in[:] = x
            mask_in[:] = mask
            mu_in[0] = mu
            t_in[:] = t.unsqueeze(0)
            spks_in[0] = spks
            cond_in[0] = cond
            dphi_dt = self.forward_estimator(
                x_in, mask_in,
                mu_in, t_in,
                spks_in,
                cond_in
            )
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1].float()

    def forward_estimator(self, x, mask, mu, t, spks, cond):
        if isinstance(self.estimator, torch.nn.Module):
            return self.estimator(x, mask, mu, t, spks, cond)
        else:
            with self.lock:
                self.estimator.set_input_shape('x', (2, 80, x.size(2)))
                self.estimator.set_input_shape('mask', (2, 1, x.size(2)))
                self.estimator.set_input_shape('mu', (2, 80, x.size(2)))
                self.estimator.set_input_shape('t', (2,))
                self.estimator.set_input_shape('spks', (2, 80))
                self.estimator.set_input_shape('cond', (2, 80, x.size(2)))
                # run trt engine
                assert self.estimator.execute_v2([x.contiguous().data_ptr(),
                                                  mask.contiguous().data_ptr(),
                                                  mu.contiguous().data_ptr(),
                                                  t.contiguous().data_ptr(),
                                                  spks.contiguous().data_ptr(),
                                                  cond.contiguous().data_ptr(),
                                                  x.data_ptr()]) is True
            return x

    def compute_loss(self, x1, mask, mu, spks=None, cond=None, streaming=False):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t = 1 - torch.cos(t * 0.5 * torch.pi)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        # during training, we randomly drop condition to trade off mode coverage and sample fidelity
        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
            mu = mu * cfg_mask.view(-1, 1, 1)
            spks = spks * cfg_mask.view(-1, 1)
            cond = cond * cfg_mask.view(-1, 1, 1)

        pred = self.estimator(y, mask, mu, t.squeeze(), spks, cond, streaming=streaming)
        loss = F.mse_loss(pred * mask, u * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])
        return loss, y


class CausalConditionalCFM(ConditionalCFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(in_channels, cfm_params, n_spks, spk_emb_dim, estimator)
        self.rand_noise = torch.randn([1, 80, 50 * 300])

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, cache={}):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """

        offset = cache.pop('offset')
        z = self.rand_noise[:, :, :mu.size(2) + offset].to(mu.device).to(mu.dtype) * temperature
        z = z[:, :, offset:]
        offset += mu.size(2)
        # fix prompt and overlap part mu and z
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        mel, cache = self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond, cache=cache)
        cache['offset'] = offset
        return mel, cache

    def solve_euler(self, x, t_span, mu, mask, spks, cond, cache):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        # Do not use concat, it may cause memory format changed and trt infer with wrong results!
        x_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        mask_in = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=x.dtype)
        mu_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        t_in = torch.zeros([2], device=x.device, dtype=x.dtype)
        spks_in = torch.zeros([2, 80], device=x.device, dtype=x.dtype)
        cond_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        flow_cache_size = cache['down_blocks_kv_cache'].shape[4]
        for step in range(1, len(t_span)):
            # Classifier-Free Guidance inference introduced in VoiceBox
            x_in[:] = x
            mask_in[:] = mask
            mu_in[0] = mu
            t_in[:] = t.unsqueeze(0)
            spks_in[0] = spks
            cond_in[0] = cond
            cache_step = {k: v[step - 1] for k, v in cache.items()}
            dphi_dt, cache_step = self.forward_estimator(
                x_in, mask_in,
                mu_in, t_in,
                spks_in,
                cond_in,
                cache_step
            )
            # NOTE if smaller than flow_cache_size, means last chunk, no need to cache
            if flow_cache_size != 0 and x_in.shape[2] >= flow_cache_size:
                cache['down_blocks_conv_cache'][step - 1] = cache_step[0]
                cache['down_blocks_kv_cache'][step - 1] = cache_step[1][:, :, :, -flow_cache_size:]
                cache['mid_blocks_conv_cache'][step - 1] = cache_step[2]
                cache['mid_blocks_kv_cache'][step - 1] = cache_step[3][:, :, :, -flow_cache_size:]
                cache['up_blocks_conv_cache'][step - 1] = cache_step[4]
                cache['up_blocks_kv_cache'][step - 1] = cache_step[5][:, :, :, -flow_cache_size:]
                cache['final_blocks_conv_cache'][step - 1] = cache_step[6]
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return sol[-1].float(), cache

    def forward_estimator(self, x, mask, mu, t, spks, cond, cache):
        if isinstance(self.estimator, torch.nn.Module):
            x, cache1, cache2, cache3, cache4, cache5, cache6, cache7 = self.estimator.forward_chunk(x, mask, mu, t, spks, cond, **cache)
            cache = (cache1, cache2, cache3, cache4, cache5, cache6, cache7)
        else:
            with self.lock:
                self.estimator.set_input_shape('x', (2, 80, x.size(2)))
                self.estimator.set_input_shape('mask', (2, 1, x.size(2)))
                self.estimator.set_input_shape('mu', (2, 80, x.size(2)))
                self.estimator.set_input_shape('t', (2,))
                self.estimator.set_input_shape('spks', (2, 80))
                self.estimator.set_input_shape('cond', (2, 80, x.size(2)))
                self.estimator.set_input_shape('down_blocks_conv_cache', cache['down_blocks_conv_cache'].shape)
                self.estimator.set_input_shape('down_blocks_kv_cache', cache['down_blocks_kv_cache'].shape)
                self.estimator.set_input_shape('mid_blocks_conv_cache', cache['mid_blocks_conv_cache'].shape)
                self.estimator.set_input_shape('mid_blocks_kv_cache', cache['mid_blocks_kv_cache'].shape)
                self.estimator.set_input_shape('up_blocks_conv_cache', cache['up_blocks_conv_cache'].shape)
                self.estimator.set_input_shape('up_blocks_kv_cache', cache['up_blocks_kv_cache'].shape)
                self.estimator.set_input_shape('final_blocks_conv_cache', cache['final_blocks_conv_cache'].shape)
                # run trt engine
                down_blocks_kv_cache_out = torch.zeros(1, 4, 2, x.size(2), 512, 2).to(x)
                mid_blocks_kv_cache_out = torch.zeros(12, 4, 2, x.size(2), 512, 2).to(x)
                up_blocks_kv_cache_out = torch.zeros(1, 4, 2, x.size(2), 512, 2).to(x)
                assert self.estimator.execute_v2([x.contiguous().data_ptr(),
                                                  mask.contiguous().data_ptr(),
                                                  mu.contiguous().data_ptr(),
                                                  t.contiguous().data_ptr(),
                                                  spks.contiguous().data_ptr(),
                                                  cond.contiguous().data_ptr(),
                                                  cache['down_blocks_conv_cache'].contiguous().data_ptr(),
                                                  cache['down_blocks_kv_cache'].contiguous().data_ptr(),
                                                  cache['mid_blocks_conv_cache'].contiguous().data_ptr(),
                                                  cache['mid_blocks_kv_cache'].contiguous().data_ptr(),
                                                  cache['up_blocks_conv_cache'].contiguous().data_ptr(),
                                                  cache['up_blocks_kv_cache'].contiguous().data_ptr(),
                                                  cache['final_blocks_conv_cache'].contiguous().data_ptr(),
                                                  x.data_ptr(),
                                                  cache['down_blocks_conv_cache'].data_ptr(),
                                                  down_blocks_kv_cache_out.data_ptr(),
                                                  cache['mid_blocks_conv_cache'].data_ptr(),
                                                  mid_blocks_kv_cache_out.data_ptr(),
                                                  cache['up_blocks_conv_cache'].data_ptr(),
                                                  up_blocks_kv_cache_out.data_ptr(),
                                                  cache['final_blocks_conv_cache'].data_ptr()]) is True
                cache = (cache['down_blocks_conv_cache'],
                         down_blocks_kv_cache_out,
                         cache['mid_blocks_conv_cache'],
                         mid_blocks_kv_cache_out,
                         cache['up_blocks_conv_cache'],
                         up_blocks_kv_cache_out,
                         cache['final_blocks_conv_cache'])
        return x, cache
