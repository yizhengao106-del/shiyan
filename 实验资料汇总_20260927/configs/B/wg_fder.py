_base_ = ['wg_base.py']
model_cfg = dict(use_weight_guidance=False, use_frequency=True, use_edge=True,
                 edge_loss_weight=0.3)
info = 'original_plus_frequency_edge_only'
