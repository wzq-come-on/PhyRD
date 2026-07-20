import torch
from torch import nn

from phyrd.models import PhyRDModel, checkpoint_backbone_spec
from phyrd.models.deterministic import DeterministicLossOutput, register_backbone


class TinyDeterministic(nn.Module):
    def __init__(self, input_frames: int, output_frames: int, *, gain: float = 1.0) -> None:
        super().__init__()
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.gain = nn.Parameter(torch.tensor(float(gain)))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return history[:, -1:].repeat(1, self.output_frames, 1, 1, 1) * self.gain

    def training_loss(
        self, history: torch.Tensor, target: torch.Tensor
    ) -> DeterministicLossOutput:
        prediction = self(history)
        loss = (prediction - target).abs().mean()
        return DeterministicLossOutput(
            loss=loss,
            prediction=prediction,
            metrics={"loss_tiny": loss.detach()},
        )


register_backbone("tiny_test", TinyDeterministic)


def test_phyrd_builds_a_registered_backbone_from_config() -> None:
    model = PhyRDModel(
        input_frames=2,
        output_frames=3,
        base_channels=8,
        diffusion_steps=4,
        freeze_deterministic=False,
        deterministic={"name": "tiny_test", "params": {"gain": 0.5}},
    )
    history = torch.rand(1, 2, 1, 8, 8)
    target = torch.rand(1, 3, 1, 8, 8)
    result = model(history, target, stage="deterministic")

    assert result["trend"].shape == target.shape
    assert set(("loss_gen", "loss_tiny", "trend")) <= result.keys()
    result["loss_gen"].backward()
    assert model.deterministic.gain.grad is not None


def test_checkpoint_protocol_accepts_only_the_known_official_migration() -> None:
    params = {"patch_size": 4, "model_resolution": 128}
    assert checkpoint_backbone_spec(
        {
            "deterministic_backbone": "sdir_official",
            "deterministic_config": params,
        }
    ) == {"name": "sdir_official", "params": params}
    assert checkpoint_backbone_spec(
        {"deterministic_backbone": "sdir", "deterministic_config": params}
    ) is None
