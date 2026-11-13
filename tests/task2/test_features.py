import numpy as np
import pytest
import torch
from atml_pa3.task2.features import valid_feature, write_feature


def test_fp16_cache_integrity_and_resume(tmp_path):
    path = tmp_path / "1.npy"
    digest = write_feature(path, np.ones((196,768), dtype=np.float16))
    record = {"sha256": digest, "source_sha256": "image", "signature": "config"}
    assert valid_feature(path, record, "image", "config")
    assert not valid_feature(path, record, "different", "config")
    assert not valid_feature(path, record, "image", "changed")
    path.write_bytes(b"corruption")
    assert not valid_feature(path, record, "image", "config")


@pytest.mark.parametrize("array", [np.zeros((49,768),np.float16), np.zeros((196,768),np.float32),
                                   np.full((196,768), np.nan, np.float16)])
def test_cache_rejects_wrong_shape_dtype_or_nonfinite(tmp_path, array):
    with pytest.raises(ValueError, match="FP16"):
        write_feature(tmp_path / "bad.npy", array)


def test_siglip_patch_order_and_layernorm_once():
    from transformers import SiglipVisionConfig, SiglipVisionModel
    config = SiglipVisionConfig(hidden_size=4, intermediate_size=8, num_hidden_layers=0,
                                num_attention_heads=2, image_size=4, patch_size=2)
    model = SiglipVisionModel(config).eval()
    model.use_head = False
    with torch.no_grad():
        model.embeddings.patch_embedding.weight.fill_(1)
        model.embeddings.patch_embedding.bias.zero_()
        model.embeddings.position_embedding.weight.zero_()
    image = torch.tensor([[1.,1.,2.,2.],[1.,1.,2.,2.],[3.,3.,4.,4.],[3.,3.,4.,4.]])[None,None].expand(1,3,4,4)
    patches = model.embeddings(image)
    assert patches[0,:,0].tolist() == [12.,24.,36.,48.]
    calls = []
    hook = model.post_layernorm.register_forward_hook(lambda *_: calls.append(1))
    output = model(pixel_values=image).last_hidden_state
    hook.remove()
    assert calls == [1]
    torch.testing.assert_close(output, model.post_layernorm(patches))


def test_pil_preprocessing_is_deterministic_rgb_224():
    from PIL import Image
    from transformers import SiglipImageProcessorPil
    processor = SiglipImageProcessorPil()
    image = Image.new("RGB", (37, 29), (0,0,0))
    a = processor(images=[image], return_tensors="pt")["pixel_values"]
    b = processor(images=[image], return_tensors="pt")["pixel_values"]
    assert a.shape == (1,3,224,224)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(a, -torch.ones_like(a))
