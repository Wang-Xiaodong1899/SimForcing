"""CPU regression tests; no pretrained weights or dataset downloads required."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import get_object
from omegaconf import OmegaConf
from simforcing.models.wan22 import SimForcing, ActionConditionedModel
from simforcing.models.wan22 import base
from simforcing.models.wan22.action_dit import ActionDiT
from simforcing.models.wan22.wan_video_dit import WanVideoDiT

ROOT = Path(__file__).resolve().parents[1]
VIDEO = dict(
    has_image_input=False,
    patch_size=(1, 2, 2),
    in_dim=2,
    hidden_dim=24,
    ffn_dim=48,
    freq_dim=16,
    text_dim=16,
    out_dim=2,
    num_heads=2,
    attn_head_dim=12,
    num_layers=2,
    eps=1e-6,
    seperated_timestep=True,
    fuse_vae_embedding_in_latents=True,
    action_conditioned=False,
    action_dim=7,
    video_attention_mask_mode="first_frame_causal",
    action_group_causal_mask_mode="group_diagonal",
)
ACTION = dict(
    action_dim=7,
    hidden_dim=24,
    ffn_dim=48,
    num_heads=2,
    attn_head_dim=12,
    num_layers=2,
    text_dim=16,
    freq_dim=16,
    eps=1e-6,
)


class TinyVAE(torch.nn.Module):
    temporal_downsample_factor = 4
    upsampling_factor = 8
    model = SimpleNamespace(z_dim=2)

    def encode(self, video, **kwargs):
        if isinstance(video, list):
            return [self.encode(x.unsqueeze(0), **kwargs)[0] for x in video]
        return torch.nn.functional.avg_pool3d(video[:, :2, ::4], (1, 8, 8))

    def decode(self, z, **kwargs):
        z = torch.cat([z, z[:, :1]], dim=1)
        return torch.nn.functional.interpolate(
            z, size=((z.shape[2] - 1) * 4 + 1, z.shape[3] * 8, z.shape[4] * 8)
        )


def components(**kwargs):
    return SimpleNamespace(
        dit=WanVideoDiT(**kwargs["dit_config"]),
        vae=TinyVAE(),
        text_encoder=None,
        tokenizer=None,
        dit_path="test",
        vae_path="test",
        text_encoder_path=None,
        tokenizer_path=None,
    )


def frozen(self, **kwargs):
    # Construct independently, with the same initialization sequence as the loader.
    video = WanVideoDiT(**kwargs["video_dit_config"])
    action = ActionDiT(**kwargs["action_dit_config"])
    from simforcing.models.wan22.mot import MoT

    self.frozen_video_expert = video
    self.frozen_action_expert = action
    self.frozen_mot = MoT(
        mixtures={"video": video, "action": action}, mot_checkpoint_mixed_attn=False
    )
    self.frozen_mot.eval().requires_grad_(False)


def make_model(kind=SimForcing, seed=7):
    torch.manual_seed(seed)
    with (
        patch.object(base, "load_wan22_ti2v_5b_components", components),
        patch.object(
            ActionDiT,
            "from_pretrained",
            side_effect=lambda **kw: ActionDiT(**kw["action_dit_config"]),
        ),
        patch.object(SimForcing, "_init_frozen_act_cond_branch", frozen),
    ):
        extra = (
            dict(
                frozen_video_expert_checkpoint="test.pt",
                loss_lambda_delta=0.5,
                loss_lambda_real=1.0,
                loss_lambda_syn=0.1,
                sim_cond_init_scale=0.0,
                sim_cond_prob=0.7,
                action_dropout_prob=0.0,
                first_frame_dropout_prob=0.0,
            )
            if kind is SimForcing
            else {}
        )
        return kind.from_wan22_pretrained(
            model_id="test",
            tokenizer_model_id="test",
            device="cpu",
            torch_dtype=torch.float32,
            video_dit_config=VIDEO,
            action_dit_config=ACTION,
            load_text_encoder=False,
            proprio_dim=7,
            mot_checkpoint_mixed_attn=True,
            **extra,
        )


def sample():
    torch.manual_seed(33)
    return dict(
        video_syn=torch.randn(2, 3, 9, 16, 16),
        video_real=torch.randn(2, 3, 9, 16, 16),
        action=torch.randn(2, 8, 7),
        proprio=torch.randn(2, 9, 7),
        context=torch.randn(2, 4, 16),
        context_mask=torch.ones(2, 4, dtype=torch.bool),
    )


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_configs(self):
        for name, target in [
            ("bridge_real_act_cond_224_1e-4", ActionConditionedModel),
            ("bridge_real_simforcing_224_1e-4", SimForcing),
        ]:
            with initialize_config_dir(
                config_dir=str(ROOT / "configs"), version_base="1.3"
            ):
                cfg = compose(config_name="train", overrides=[f"task={name}"])
                # Resolve model/data without Hydra's run-time-only output_dir resolver.
                OmegaConf.to_container(cfg.model, resolve=True, throw_on_missing=True)
                OmegaConf.to_container(cfg.data, resolve=True, throw_on_missing=True)
                self.assertTrue(callable(get_object(cfg.model._target_)))
                self.assertTrue(callable(get_object(cfg.data.train._target_)))
        self.assertEqual(SimForcing.__bases__, (base.VideoActionModel,))

    def test_training_and_checkpoint(self):
        model = make_model()
        model.eval()
        model.dit.train()  # Match trainer's DiT-only mode.
        batch = sample()
        loss, metrics = model.training_loss(batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        conv = model.video_expert.blocks[0].sim_cond_conv
        self.assertGreater(conv.weight.grad.abs().sum().item(), 0.0)
        self.assertTrue(all(p.grad is None for p in model.frozen_mot.parameters()))
        model.train()
        self.assertFalse(model.frozen_mot.training)
        self.assertGreaterEqual(metrics["loss_delta"], 0.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "weights.pt")
            model.save_checkpoint(path, step=12)
            restored = make_model()
            restored.load_checkpoint(path)
            for key, value in model.mot.state_dict().items():
                torch.testing.assert_close(value, restored.mot.state_dict()[key])

    def test_gate_zero_matches_unconditioned_and_blocks_conv_gradient(self):
        model = make_model()
        model.eval()
        with torch.no_grad():
            for block in model.video_expert.blocks:
                block.sim_cond_conv.weight.normal_()
                block.sim_cond_conv.bias.fill_(0.4)
        inputs = model.build_paired_inputs(sample())
        kwargs = dict(
            latents=inputs["latents_real"],
            timestep_video=torch.ones(2) * 400,
            action=inputs["action"],
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            fuse_vae_embedding_in_latents=True,
        )
        plain = model._forward_student_with_action(**kwargs)
        gated = model._forward_student_with_sim_condition(
            **kwargs, sim_latent=inputs["latents_syn"], sim_cond_gate=torch.zeros(2)
        )
        torch.testing.assert_close(plain, gated, rtol=0, atol=0)
        gated.sum().backward()
        for block in model.video_expert.blocks:
            self.assertEqual(block.sim_cond_conv.weight.grad.abs().sum().item(), 0.0)
            self.assertEqual(block.sim_cond_conv.bias.grad.abs().sum().item(), 0.0)

    def test_baseline_backward(self):
        model = make_model(ActionConditionedModel)
        batch = sample()
        batch["video"] = batch.pop("video_real")
        batch.pop("video_syn")
        model.eval()
        model.dit.train()
        loss, _ = model.training_loss(batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.dit.parameters()))

    def test_two_stage_inference(self):
        model = make_model()
        batch = sample()
        common = dict(
            action=batch["action"][0],
            proprio=batch["proprio"][0, 0],
            context=batch["context"][0],
            context_mask=batch["context_mask"][0],
            num_video_frames=9,
            num_inference_steps=2,
            seed=11,
        )
        latent = model.infer_sim_latents(
            input_image=batch["video_syn"][0, :, 0], **common
        )
        for scale in (0.0, 0.5, 1.0):
            out = model.infer_video(
                prompt=None,
                input_image_syn=None,
                input_image_real=batch["video_real"][0, :, 0],
                sim_latents=latent,
                sim_cond_cfg_scale=scale,
                **common,
            )
            self.assertEqual(len(out["video"]), 9)


if __name__ == "__main__":
    unittest.main()
