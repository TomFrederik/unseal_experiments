from grokking_model import GrokkingTransformer

path = './unseal/interface/post-grokking.ckpt'

model = GrokkingTransformer.load_from_checkpoint(path)
w_q, w_k, w_v = model.transformer[0].self_attn.in_proj_weight.chunk(3)
w_o = model.transformer[0].self_attn.out_proj.weight