import numpy as np
import pytest
import torch
from affine import Affine
from shapely.geometry import box

from burnmap import analysis, config, inference
from burnmap.imagery import _harmonize
from burnmap.model import BinaryUNet, load_checkpoint


class PassThrough(torch.nn.Module):
    """Logit = channel 0, so stitched output must equal sigmoid(x[0]) everywhere."""

    def forward(self, x):
        return x[:, 0]


@pytest.mark.parametrize("shape", [(300, 200), (512, 512), (1100, 777)])
def test_tiling_reassembles_exactly(shape):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, *shape)).astype(np.float32)
    prob = inference.predict_probabilities(PassThrough(), x, tile=256, overlap=32)
    assert prob.shape == shape
    np.testing.assert_allclose(prob, 1 / (1 + np.exp(-x[0])), atol=1e-5)


def test_unet_shape_and_checkpoint_roundtrip(tmp_path):
    net = BinaryUNet(2)
    assert net(torch.zeros(1, 2, 64, 64)).shape == (1, 64, 64)
    p = tmp_path / "m.pt"
    torch.save({"model_state_dict": net.state_dict(),
                "normalization": {"channel_means": torch.tensor([2500., 1500.]),
                                  "channel_stds": torch.tensor([900., 700.])}}, p)
    m = load_checkpoint(p)
    assert m.channel_means.tolist() == [2500.0, 1500.0]


def test_rejects_phase2_checkpoint(tmp_path):
    p = tmp_path / "m8.pt"
    torch.save({"model_state_dict": BinaryUNet(8).state_dict(), "normalization": {}}, p)
    with pytest.raises(ValueError, match="8 input channels"):
        load_checkpoint(p)


def test_acres_and_compare():
    pred = np.zeros((100, 100), bool); pred[:50] = True
    ref = np.zeros((100, 100), bool); ref[25:75] = True
    a = analysis.compare(pred, ref)
    px_ac = config.PIXEL_M**2 / config.SQM_PER_ACRE
    assert a.predicted_acres == pytest.approx(5000 * px_ac)
    assert a.iou == pytest.approx(2500 / 7500)
    assert a.precision == pytest.approx(0.5) and a.recall == pytest.approx(0.5)


def test_rasterize_aligns_with_grid():
    t = Affine(10, 0, 500000, 0, -10, 4000000)  # 10 m UTM-like grid
    m = analysis.rasterize(box(500100, 3999500, 500300, 3999900), (100, 100), t)
    assert m.sum() == 20 * 40 and m[10:50, 10:30].all()


def test_harmonize_offset():
    dn = np.array([0, 500, 1000, 3500], np.uint16)
    np.testing.assert_array_equal(_harmonize(dn, 5.11), [0, 0, 0, 2500])
    np.testing.assert_array_equal(_harmonize(dn, 2.09), [0, 500, 1000, 3500])


def test_dnbr_flags_burned_pixels():
    pre_nir, pre_swir = np.full((2, 2), 3000.), np.full((2, 2), 1500.)
    post_nir, post_swir = pre_nir.copy(), pre_swir.copy()
    post_nir[0, 0], post_swir[0, 0] = 1200., 2200.  # charred: NIR down, SWIR up
    mask, _ = analysis.dnbr_mask(pre_nir, pre_swir, post_nir, post_swir)
    assert mask.tolist() == [[True, False], [False, False]]
