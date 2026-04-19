"""Ranger22 — optimized Ranger21 drop-in.

Subclasses ``ranger22.baseline.Ranger21`` with the same constructor
signature. When the optimizer is configured in the "nnue-pytorch common
path" (no GC, no AdaBelief, no AGC, no softplus, no weight decay, madgrad
off, ``momentum_type="pnm"``), ``step`` takes a fused fast path.
Any other configuration falls back to the baseline semantics.

The fast path preserves Ranger21's actual update rule:
- by default, a ``torch.compile`` helper handles both EMA state evolution
  (``grad_ma`` and ``variance_ma``) and the Ranger21 parameter update for
  1-D and 2-D CUDA float32 tensors
- non-compiled tensors use PyTorch fused AdamW as the state-update core plus
  eager Ranger21 post-update
- ``compile_post_update=False`` opts out of compiled full-update

This keeps training behavior aligned with the baseline while reducing launch
overhead in the nnue-pytorch configuration.
"""

from __future__ import annotations

import functools
import math
import os
from collections.abc import Callable

import torch
from torch.optim.adamw import adamw as functional_adamw

from .baseline import Ranger21


def _ensure_triton_libcuda_path():
    if os.path.exists("/sbin/ldconfig") or "TRITON_LIBCUDA_PATH" in os.environ:
        return
    for candidate in (
        "/run/opengl-driver/lib",
        "/run/current-system/sw/share/nix-ld/lib",
    ):
        if os.path.exists(os.path.join(candidate, "libcuda.so.1")):
            os.environ["TRITON_LIBCUDA_PATH"] = candidate
            return


def _compiled_full_update_1d(
    p: torch.Tensor,
    grad: torch.Tensor,
    variance_ma: torch.Tensor,
    gma_cur: torch.Tensor,
    gma_prev: torch.Tensor,
    scalars: torch.Tensor,
):
    lr = scalars[0]
    inv_sqrt_bc2 = scalars[1]
    eps = scalars[2]
    two_nl_factor = scalars[3]
    noise_norm_recip = scalars[4]
    pnm_a = scalars[5]
    pnm_b = scalars[6]
    step_size = scalars[7]
    beta2 = scalars[8]
    one_minus_beta2 = scalars[9]
    beta1_sq = scalars[10]
    one_minus_beta1_sq = scalars[11]

    variance_ma.mul_(beta2)
    variance_ma.add_(grad * grad * one_minus_beta2)
    gma_cur.mul_(beta1_sq)
    gma_cur.add_(grad * one_minus_beta1_sq)

    unorm = torch.linalg.vector_norm(p, ord=2)
    correction = two_nl_factor * (1 - 1 / (unorm + eps))
    denom = variance_ma.sqrt() * inv_sqrt_bc2 + eps
    pn = (gma_cur * pnm_a + gma_prev * pnm_b) * noise_norm_recip
    p.mul_(1 - lr * correction)
    p.sub_((pn / denom) * step_size)


def _compiled_full_update_2d(
    p: torch.Tensor,
    grad: torch.Tensor,
    variance_ma: torch.Tensor,
    gma_cur: torch.Tensor,
    gma_prev: torch.Tensor,
    scalars: torch.Tensor,
):
    lr = scalars[0]
    inv_sqrt_bc2 = scalars[1]
    eps = scalars[2]
    two_nl_factor = scalars[3]
    noise_norm_recip = scalars[4]
    pnm_a = scalars[5]
    pnm_b = scalars[6]
    step_size = scalars[7]
    beta2 = scalars[8]
    one_minus_beta2 = scalars[9]
    beta1_sq = scalars[10]
    one_minus_beta1_sq = scalars[11]

    variance_ma.mul_(beta2)
    variance_ma.add_(grad * grad * one_minus_beta2)
    gma_cur.mul_(beta1_sq)
    gma_cur.add_(grad * one_minus_beta1_sq)

    unorm = torch.linalg.vector_norm(p, ord=2, dim=1, keepdim=True)
    correction = two_nl_factor * (1 - 1 / (unorm + eps))
    denom = variance_ma.sqrt() * inv_sqrt_bc2 + eps
    pn = (gma_cur * pnm_a + gma_prev * pnm_b) * noise_norm_recip
    p.mul_(1 - lr * correction)
    p.sub_((pn / denom) * step_size)


@functools.lru_cache(maxsize=None)
def _get_compiled_full_update(ndim: int):
    _ensure_triton_libcuda_path()
    if ndim == 1:
        return torch.compile(
            _compiled_full_update_1d,
            fullgraph=True,
            dynamic=False,
            mode="reduce-overhead",
        )
    if ndim == 2:
        return torch.compile(
            _compiled_full_update_2d,
            fullgraph=True,
            dynamic=False,
            mode="reduce-overhead",
        )
    raise ValueError(f"no compiled Ranger22 full-update for ndim={ndim}")


class Ranger22(Ranger21):
    """Drop-in replacement for Ranger21 with a fused AdamW state-update core."""

    def __init__(self, *args, **kwargs):
        self._compile_post_update = kwargs.pop("compile_post_update", True)
        self._compile_scalars_cache = {}
        super().__init__(*args, **kwargs)
        self._fused_path_ok = self._check_fused_path()
        if self._fused_path_ok:
            print("Ranger22: fused AdamW-core path ENABLED")
            if self._compile_post_update:
                print("Ranger22: torch.compile full-update ENABLED")
        else:
            print(
                "Ranger22: fused path not available, falling back to baseline semantics"
            )

    # ------------------------------------------------------------------ helpers

    def _check_fused_path(self) -> bool:
        """Detect whether the current config matches the fused-kernel path."""
        return (
            not self.use_gc
            and not self.use_gcnorm
            and not self.use_adabelief
            and not self.use_madgrad
            and not self.agc_active
            and not self.softplus
            and self.decay == 0.0
            and self.momentum_pnm  # must be in the PNM code path
            and self.normloss_active  # kernel assumes norm-loss on
        )

    @staticmethod
    def _param_eligible_for_fused_core(p: torch.Tensor, grad: torch.Tensor) -> bool:
        return p.is_cuda and p.dtype == torch.float32 and not grad.is_sparse

    @staticmethod
    def _param_eligible_for_compiled_post_update(p: torch.Tensor) -> bool:
        return p.is_cuda and p.dtype == torch.float32 and p.ndim in (1, 2)

    def _ensure_fused_core_state(self, p: torch.Tensor, state: dict):
        if "adamw_step" not in state:
            state["adamw_step"] = torch.tensor(
                float(state["step"] - 1),
                dtype=torch.float32,
                device=p.device,
            )
        if "adamw_shadow" not in state:
            state["adamw_shadow"] = torch.zeros_like(
                p, memory_format=torch.preserve_format
            )

    @staticmethod
    def _get_group_buffers(group: dict):
        if "_r22_buffers" not in group:
            group["_r22_buffers"] = {
                "shadow_params": [],
                "grads": [],
                "exp_avgs": [],
                "exp_avg_sqs": [],
                "state_steps": [],
                "compiled_params": [],
                "fused_params": [],
                "fallback_params": [],
            }
        bufs = group["_r22_buffers"]
        bufs["shadow_params"].clear()
        bufs["grads"].clear()
        bufs["exp_avgs"].clear()
        bufs["exp_avg_sqs"].clear()
        bufs["state_steps"].clear()
        bufs["compiled_params"].clear()
        bufs["fused_params"].clear()
        bufs["fallback_params"].clear()
        return bufs

    def _get_compile_scalars_buffer(self, p: torch.Tensor) -> torch.Tensor:
        key = (p.device.type, p.device.index, p.dtype)
        if key not in self._compile_scalars_cache:
            host = torch.empty(12, dtype=p.dtype, pin_memory=True)
            device = torch.empty(12, device=p.device, dtype=p.dtype)
            self._compile_scalars_cache[key] = (host, device)
        return self._compile_scalars_cache[key][1]

    @staticmethod
    def _update_compile_scalars_buffer(
        host_scalars: torch.Tensor,
        scalars: torch.Tensor,
        lr: float,
        inv_sqrt_bc2: float,
        eps: float,
        two_nl_factor: float,
        noise_norm_recip: float,
        pnm_a: float,
        pnm_b: float,
        step_size: float,
        beta2: float,
        one_minus_beta2: float,
        beta1_sq: float,
        one_minus_beta1_sq: float,
    ) -> None:
        host_scalars[0] = lr
        host_scalars[1] = inv_sqrt_bc2
        host_scalars[2] = eps
        host_scalars[3] = two_nl_factor
        host_scalars[4] = noise_norm_recip
        host_scalars[5] = pnm_a
        host_scalars[6] = pnm_b
        host_scalars[7] = step_size
        host_scalars[8] = beta2
        host_scalars[9] = one_minus_beta2
        host_scalars[10] = beta1_sq
        host_scalars[11] = one_minus_beta1_sq
        scalars.copy_(host_scalars, non_blocking=True)

    def _eager_post_update(
        self,
        p: torch.Tensor,
        variance_ma: torch.Tensor,
        gma_cur: torch.Tensor,
        gma_prev: torch.Tensor,
        lr: float,
        inv_sqrt_bc2: float,
        eps: float,
        two_nl_factor: float,
        pnm_a: float,
        pnm_b: float,
        noise_norm_recip: float,
        step_size: float,
    ):
        unorm = self.unit_norm(p.data)
        correction = two_nl_factor * (1 - torch.div(1, unorm + eps))
        p.mul_(1 - lr * correction)

        denom = (variance_ma.sqrt() * inv_sqrt_bc2).add_(eps)
        pn = gma_cur.mul(pnm_a).add(gma_prev, alpha=pnm_b).mul(noise_norm_recip)
        p.addcdiv_(pn, denom, value=-step_size)

    def _eager_state_and_post_update(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        variance_ma: torch.Tensor,
        gma_cur: torch.Tensor,
        gma_prev: torch.Tensor,
        lr: float,
        inv_sqrt_bc2: float,
        eps: float,
        two_nl_factor: float,
        pnm_a: float,
        pnm_b: float,
        noise_norm_recip: float,
        step_size: float,
        beta2: float,
        one_minus_beta2: float,
        beta1_sq: float,
        one_minus_beta1_sq: float,
    ):
        variance_ma.mul_(beta2).addcmul_(grad, grad, value=one_minus_beta2)
        gma_cur.mul_(beta1_sq).add_(grad, alpha=one_minus_beta1_sq)
        self._eager_post_update(
            p,
            variance_ma,
            gma_cur,
            gma_prev,
            lr,
            inv_sqrt_bc2,
            eps,
            two_nl_factor,
            pnm_a,
            pnm_b,
            noise_norm_recip,
            step_size,
        )

    @torch.no_grad()
    def step(self, closure=None):
        if not self._fused_path_ok:
            return super().step(closure)

        loss = None
        if closure is not None and isinstance(closure, Callable):
            with torch.enable_grad():
                loss = closure()

        # One global step counter so all params share a step (matches the
        # baseline which reads `state["step"]` from whichever param has the
        # grad first for lr schedule purposes).
        #
        # We still increment per-param state["step"] below so resuming from
        # checkpoints (or mixing with baseline) continues to work.

        # -------- phase 0: state init + simple bookkeeping --------
        # Running total of numel only on the first step, so the lr schedule
        # and warmup can use the same .total_iterations accounting as
        # baseline.
        param_size = 0

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_size += p.numel()
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["grad_ma"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["variance_ma"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["neg_grad_ma"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # max_variance_ma is a no-op in baseline but we keep the
                    # key for state-dict compatibility.
                    state["max_variance_ma"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if self.lookahead_active:
                        state["lookahead_params"] = torch.zeros_like(p.data)
                        state["lookahead_params"].copy_(p.data)
                state["step"] += 1

        if not self.param_size:
            self.param_size = param_size

        # -------- phase 1: fused state update + Ranger21 param update --------
        for group in self.param_groups:
            group_lr = float(group["lr"])
            first_p_with_grad = next(
                (p for p in group["params"] if p.grad is not None), None
            )
            if first_p_with_grad is None:
                continue

            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            step = self.state[first_p_with_grad]["step"]
            lr_updated = self.update_lr(group_lr, step)
            lr = group_lr if lr_updated is None else float(lr_updated)

            # Precompute per-step scalar constants (identical to baseline).
            one_minus_beta2 = 1.0 - beta2
            beta1_sq = beta1 * beta1
            one_minus_beta1_sq = 1.0 - beta1_sq
            bc1 = 1.0 - beta1**step
            bc2 = 1.0 - beta2**step
            inv_sqrt_bc2 = 1.0 / math.sqrt(bc2)
            step_size = lr / bc1
            two_nl_factor = 2.0 * self.normloss_factor
            # Upstream uses `self.momentum_pnm` (a bool) as a float here.
            pnm_a = float(1 + self.momentum_pnm)  # 2.0 in practice
            pnm_b = float(-self.momentum_pnm)  # -1.0 in practice
            noise_norm = math.sqrt((1.0 + beta2) ** 2 + beta2**2)
            noise_norm_recip = 1.0 / noise_norm

            bufs = self._get_group_buffers(group)
            shadow_params = bufs["shadow_params"]
            grads = bufs["grads"]
            exp_avgs = bufs["exp_avgs"]
            exp_avg_sqs = bufs["exp_avg_sqs"]
            state_steps = bufs["state_steps"]
            compiled_params = bufs["compiled_params"]
            fused_params = bufs["fused_params"]
            fallback_params = bufs["fallback_params"]

            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue

                state = self.state[p]
                if step % 2 == 1:
                    gma_cur = state["grad_ma"]
                    gma_prev = state["neg_grad_ma"]
                else:
                    gma_cur = state["neg_grad_ma"]
                    gma_prev = state["grad_ma"]

                variance_ma = state["variance_ma"]

                if (
                    self._compile_post_update
                    and self._param_eligible_for_compiled_post_update(p)
                    and not grad.is_sparse
                ):
                    compiled_params.append((p, grad, variance_ma, gma_cur, gma_prev))
                elif self._param_eligible_for_fused_core(p, grad):
                    self._ensure_fused_core_state(p, state)
                    shadow_params.append(state["adamw_shadow"])
                    grads.append(grad)
                    exp_avgs.append(gma_cur)
                    exp_avg_sqs.append(variance_ma)
                    state_steps.append(state["adamw_step"])
                    fused_params.append((p, variance_ma, gma_cur, gma_prev))
                else:
                    fallback_params.append(p)

            if shadow_params:
                functional_adamw(
                    shadow_params,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    [],
                    state_steps,
                    foreach=None,
                    capturable=False,
                    differentiable=False,
                    fused=True,
                    grad_scale=None,
                    found_inf=None,
                    has_complex=False,
                    amsgrad=False,
                    beta1=beta1_sq,
                    beta2=beta2,
                    lr=0.0,
                    weight_decay=0.0,
                    eps=eps,
                    maximize=False,
                )

            compile_scalars = None
            if self._compile_post_update and compiled_params:
                key = (
                    compiled_params[0][0].device.type,
                    compiled_params[0][0].device.index,
                    compiled_params[0][0].dtype,
                )
                compile_scalars = self._get_compile_scalars_buffer(
                    compiled_params[0][0]
                )
                host_scalars, _ = self._compile_scalars_cache[key]
                self._update_compile_scalars_buffer(
                    host_scalars,
                    compile_scalars,
                    lr=lr,
                    inv_sqrt_bc2=inv_sqrt_bc2,
                    eps=eps,
                    two_nl_factor=two_nl_factor,
                    noise_norm_recip=noise_norm_recip,
                    pnm_a=pnm_a,
                    pnm_b=pnm_b,
                    step_size=step_size,
                    beta2=beta2,
                    one_minus_beta2=one_minus_beta2,
                    beta1_sq=beta1_sq,
                    one_minus_beta1_sq=one_minus_beta1_sq,
                )

            compiled_done = 0
            if compile_scalars is not None:
                for idx, (p, grad, variance_ma, gma_cur, gma_prev) in enumerate(
                    compiled_params
                ):
                    try:
                        _get_compiled_full_update(p.ndim)(
                            p, grad, variance_ma, gma_cur, gma_prev, compile_scalars
                        )
                        compiled_done = idx + 1
                    except Exception as exc:
                        self._compile_post_update = False
                        compile_scalars = None
                        exc_line = str(exc).splitlines()[0]
                        print(
                            "Ranger22: torch.compile full-update disabled after "
                            f"{type(exc).__name__}: {exc_line}"
                        )
                        break

            if compiled_done < len(compiled_params):
                for p, grad, variance_ma, gma_cur, gma_prev in compiled_params[
                    compiled_done:
                ]:
                    self._eager_state_and_post_update(
                        p,
                        grad,
                        variance_ma,
                        gma_cur,
                        gma_prev,
                        lr,
                        inv_sqrt_bc2,
                        eps,
                        two_nl_factor,
                        pnm_a,
                        pnm_b,
                        noise_norm_recip,
                        step_size,
                        beta2,
                        one_minus_beta2,
                        beta1_sq,
                        one_minus_beta1_sq,
                    )

            for p, variance_ma, gma_cur, gma_prev in fused_params:
                self._eager_post_update(
                    p,
                    variance_ma,
                    gma_cur,
                    gma_prev,
                    lr,
                    inv_sqrt_bc2,
                    eps,
                    two_nl_factor,
                    pnm_a,
                    pnm_b,
                    noise_norm_recip,
                    step_size,
                )

            for p in fallback_params:
                self._fallback_single_param(
                    p,
                    group,
                    step,
                    lr,
                    beta1,
                    beta2,
                    eps,
                    one_minus_beta2,
                    beta1_sq,
                    one_minus_beta1_sq,
                    bc2,
                    step_size,
                    two_nl_factor,
                    noise_norm_recip,
                )

        # -------- phase 2: lookahead & bookkeeping --------
        if self.lookahead_active:
            self.lookahead_process_step()

        # track_epochs reads the global iteration count — reuse baseline.
        self.track_epochs(self.state[next(iter(self.state))]["step"])
        return loss

    def lookahead_process_step(self):
        """Foreach-optimized lookahead update preserving baseline behavior."""
        if not self.lookahead_active:
            return

        self.lookahead_step += 1
        if self.lookahead_step < self.lookahead_mergetime:
            return

        self.lookahead_step = 0
        params = []
        lookahead_params = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                params.append(p.data)
                lookahead_params.append(self.state[p]["lookahead_params"])

        if not params:
            return

        try:
            torch._foreach_lerp_(params, lookahead_params, 1.0 - self.lookahead_alpha)
            torch._foreach_copy_(lookahead_params, params)
        except RuntimeError:
            for p, la in zip(params, lookahead_params):
                p.mul_(self.lookahead_alpha).add_(
                    la,
                    alpha=1.0 - self.lookahead_alpha,
                )
                la.copy_(p)

    # ---------------- fallback (used for unusual shapes / dtypes) -----------

    def _fallback_single_param(
        self,
        p,
        group,
        step,
        lr,
        beta1,
        beta2,
        eps,
        one_minus_beta2,
        beta1_sq,
        one_minus_beta1_sq,
        bc2,
        step_size,
        two_nl_factor,
        noise_norm_recip,
    ):
        """Baseline-equivalent update path for a single parameter.

        Used for shapes / dtypes the fused kernel doesn't handle. Keeps the
        exact same semantics as ``Ranger21.step`` under the nnue-pytorch
        config.
        """
        state = self.state[p]
        grad = p.grad
        variance_ma = state["variance_ma"]
        if step % 2 == 1:
            gma_cur, gma_prev = state["grad_ma"], state["neg_grad_ma"]
        else:
            gma_cur, gma_prev = state["neg_grad_ma"], state["grad_ma"]

        # variance_ma EMA
        variance_ma.mul_(beta2).addcmul_(grad, grad, value=one_minus_beta2)

        # norm loss
        unorm = self.unit_norm(p.data)
        correction = two_nl_factor * (1 - torch.div(1, unorm + self.eps))
        p.mul_(1 - lr * correction)

        # denom
        denom = (variance_ma.sqrt() / math.sqrt(bc2)).add_(eps)

        # grad_ma EMA (uses β₁², matching baseline)
        gma_cur.mul_(beta1_sq).add_(grad, alpha=one_minus_beta1_sq)

        # pnmomentum (upstream bool bug preserved via self.momentum_pnm)
        pn = (
            gma_cur.mul(1 + self.momentum_pnm)
            .add(gma_prev, alpha=-self.momentum_pnm)
            .mul(1 / math.sqrt((1 + beta2) ** 2 + beta2**2))
        )
        p.addcdiv_(pn, denom, value=-step_size)
