"""
PyTorch autograd surrogate differentiation bridge (`QERLayer`) for hybrid models.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .backends import _HAS_TORCH, nn, torch
from .denoiser import QuantumIsingDenoiser

if _HAS_TORCH:

    class _QERSurrogateFunction(torch.autograd.Function):
        """
        O(1)-circuit contrastive response backward surrogate for PyTorch autograd.

        Nudging visible fields by +/-beta * (-grad_output / ||grad_output||_inf) yields
        the forward finite-difference JVP proxy J_{zh} g_out for upstream inputs and the
        conjugate correlator update for commuting diagonal Ising weights theta. Note that
        this is an approximate backward surrogate rather than an exact transpose VJP J_{zh}^T g_out,
        coinciding with exact VJP only up to finite-depth Jacobian/Hessian asymmetry.
        """

        @staticmethod
        def forward(
            ctx,
            x_t: "torch.Tensor",
            h_cond: "torch.Tensor",
            qpu_theta: "torch.Tensor",
            denoiser: QuantumIsingDenoiser,
            t_step: int,
            shots: int,
            beta: float,
            gain: float,
        ):
            x_np = x_t.detach().cpu().numpy().astype(np.float64)
            h_np = h_cond.detach().cpu().numpy().astype(np.float64)
            th_np = qpu_theta.detach().cpu().numpy().astype(np.float64)
            b_size = x_np.shape[0]
            eye_p = np.eye(denoiser.n_visible, dtype=np.float64)
            th_b = np.broadcast_to(th_np, (b_size, denoiser.n_qpu_params))
            cp_b = np.broadcast_to(eye_p, (b_size, denoiser.n_visible, denoiser.n_visible))
            _, z_raw = denoiser.measure_observables_batch(
                x_np, np.full(b_size, t_step, dtype=np.int64), shots=shots,
                cond_batch=h_np, theta_batch=th_b, cond_proj_batch=cp_b,
            )
            outs = np.clip(z_raw / gain, -1.0, 1.0)

            ctx.denoiser = denoiser
            ctx.t_step = t_step
            ctx.shots = shots
            ctx.beta = beta
            ctx.gain = gain
            ctx.save_for_backward(x_t.detach(), h_cond.detach(), qpu_theta.detach())
            return torch.tensor(outs, dtype=x_t.dtype, device=x_t.device)

        @staticmethod
        def backward(ctx, grad_output: "torch.Tensor"):
            x_t, h_cond, qpu_theta = ctx.saved_tensors
            denoiser: QuantumIsingDenoiser = ctx.denoiser
            beta = ctx.beta
            gain = ctx.gain

            g_np = grad_output.detach().cpu().numpy().astype(np.float64)
            x_np = x_t.cpu().numpy().astype(np.float64)
            h_np = h_cond.cpu().numpy().astype(np.float64)
            th_np = qpu_theta.cpu().numpy().astype(np.float64)
            eye_p = np.eye(denoiser.n_visible, dtype=np.float64)

            nudge_dirs = -g_np
            norms = np.max(np.abs(nudge_dirs), axis=1)
            active = np.flatnonzero(norms >= 1e-8)
            grad_h = np.zeros_like(h_np)
            grad_th = np.zeros_like(th_np)
            if active.size > 0:
                unit_nudges = nudge_dirs[active] / norms[active][:, None]
                na = active.size
                x_both = np.concatenate([x_np[active], x_np[active]], axis=0)
                h_both = np.concatenate([h_np[active], h_np[active]], axis=0)
                t_both = np.full(2 * na, ctx.t_step, dtype=np.int64)
                nt_both = np.concatenate([unit_nudges, unit_nudges], axis=0)
                nb_both = np.concatenate([np.full(na, beta), np.full(na, -beta)], axis=0)
                th_both = np.broadcast_to(th_np, (2 * na, denoiser.n_qpu_params))
                cp_both = np.broadcast_to(eye_p, (2 * na, denoiser.n_visible, denoiser.n_visible))
                obs_both, z_both = denoiser.measure_observables_batch(
                    x_both, t_both, shots=ctx.shots, cond_batch=h_both,
                    nudge_target_batch=nt_both, nudge_beta_batch=nb_both,
                    theta_batch=th_both, cond_proj_batch=cp_both,
                )
                scales = (norms[active] / (2.0 * beta))[:, None]
                grad_th = -np.sum((obs_both[:na] - obs_both[na:]) * scales, axis=0) / gain
                grad_h[active] = -((z_both[:na] - z_both[na:]) * scales) / gain

            grad_h_t = torch.tensor(grad_h, dtype=h_cond.dtype, device=h_cond.device)
            grad_th_t = torch.tensor(grad_th, dtype=qpu_theta.dtype, device=qpu_theta.device)
            in_pin = denoiser.in_pinning_base + denoiser.in_pinning_slope * (1.0 - ctx.t_step / denoiser.diffusion_steps)
            grad_x_t = grad_h_t * in_pin
            return grad_x_t, grad_h_t, grad_th_t, None, None, None, None, None

    class _QERSTEFunction(torch.autograd.Function):
        """
        Zero-backward-shot Straight-Through Estimator (STE) baseline for QERLayer.

        Forward executes the identical noisy QPU circuit, while backward passes the upstream
        gradient straight through to visible fields (0 extra QPU circuits; qpu_theta receives 0 grad).
        """

        @staticmethod
        def forward(
            ctx,
            x_t: "torch.Tensor",
            h_cond: "torch.Tensor",
            qpu_theta: "torch.Tensor",
            denoiser: QuantumIsingDenoiser,
            t_step: int,
            shots: int,
            gain: float,
        ):
            x_np = x_t.detach().cpu().numpy().astype(np.float64)
            h_np = h_cond.detach().cpu().numpy().astype(np.float64)
            th_np = qpu_theta.detach().cpu().numpy().astype(np.float64)
            b_size = x_np.shape[0]
            eye_p = np.eye(denoiser.n_visible, dtype=np.float64)
            th_b = np.broadcast_to(th_np, (b_size, denoiser.n_qpu_params))
            cp_b = np.broadcast_to(eye_p, (b_size, denoiser.n_visible, denoiser.n_visible))
            _, z_raw = denoiser.measure_observables_batch(
                x_np, np.full(b_size, t_step, dtype=np.int64), shots=shots,
                cond_batch=h_np, theta_batch=th_b, cond_proj_batch=cp_b,
            )
            outs = np.clip(z_raw / gain, -1.0, 1.0)
            ctx.denoiser = denoiser
            ctx.t_step = t_step
            ctx.gain = gain
            ctx.save_for_backward(qpu_theta.detach())
            return torch.tensor(outs, dtype=x_t.dtype, device=x_t.device)

        @staticmethod
        def backward(ctx, grad_output: "torch.Tensor"):
            (qpu_theta,) = ctx.saved_tensors
            denoiser: QuantumIsingDenoiser = ctx.denoiser
            grad_h_t = grad_output / float(ctx.gain)
            in_pin = denoiser.in_pinning_base + denoiser.in_pinning_slope * (1.0 - ctx.t_step / denoiser.diffusion_steps)
            grad_x_t = grad_h_t * in_pin
            grad_th_t = torch.zeros_like(qpu_theta)
            return grad_x_t, grad_h_t, grad_th_t, None, None, None, None

    class QERLayer(nn.Module):
        """
        Drop-in PyTorch commuting-diagonal Ising QPU layer using O(1) Contrastive
        Field Probing (`_QERSurrogateFunction`) or a zero-backward-shot Straight-Through
        Estimator (`backward_mode='ste'`) for generic hybrid architectures.
        """

        def __init__(
            self,
            denoiser: QuantumIsingDenoiser,
            shots: int = 44,
            beta: float = 0.45,
            gain: float = 0.75,
            backward_mode: str = "qer",
        ):
            super().__init__()
            if backward_mode not in ("qer", "ste"):
                raise ValueError(f"Unknown backward_mode={backward_mode!r}; expected 'qer' or 'ste'.")
            self.denoiser = denoiser
            self.shots = int(shots)
            self.beta = float(beta)
            self.gain = float(gain)
            self.backward_mode = str(backward_mode)
            self.qpu_theta = nn.Parameter(torch.zeros(denoiser.n_qpu_params, dtype=torch.float32))

        def forward(
            self, x_t: "torch.Tensor", h_cond: "torch.Tensor", t_step: Optional[int] = None
        ) -> "torch.Tensor":
            with torch.no_grad():
                self.qpu_theta.clamp_(-1.5, 1.5)
            eff_t = max(1, self.denoiser.diffusion_steps // 2) if t_step is None else int(t_step)
            if self.backward_mode == "ste":
                return _QERSTEFunction.apply(
                    x_t, h_cond, self.qpu_theta, self.denoiser, eff_t, self.shots, self.gain
                )
            return _QERSurrogateFunction.apply(
                x_t, h_cond, self.qpu_theta, self.denoiser, eff_t, self.shots, self.beta, self.gain
            )

    # Backward-compatible aliases
    _QERDFunction = _QERSurrogateFunction
    QERDDiffusionLayer = QERLayer
