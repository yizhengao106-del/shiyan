base_seed = 112358

deterministic = True

evaluate = False

has_test = True

load_from = None

log_interval = 20

metric_names = [
    'sm',
    'wfm',
    'mae',
    'em',
    'fmeasure',
]

pretrained = True

save_results = False

test = dict(
    batch_size=4,
    clip_range=None,
    data=dict(
        names=[
            'chameleon',
            'camo_te',
            'cod10k_te',
            'nc4k',
        ],
        shape=dict(h=384, w=384)),
    num_workers=2)

train = dict(
    batch_size=4,
    bn=dict(freeze_affine=True, freeze_encoder=False, freeze_status=True),
    data=dict(names=[
        'cod10k_tr',
        'camo_tr',
    ], shape=dict(h=384, w=384)),
    epoch_based=True,
    grad_acc_step=1,
    lr=0.0001,
    num_epochs=150,
    num_iters=None,
    num_workers=2,
    optimizer=dict(
        cfg=dict(diff_factor=0.1, weight_decay=0),
        group_mode='finetune',
        mode='adam',
        set_to_none=False),
    sche_usebatch=True,
    scheduler=dict(
        cfg=dict(gamma=0.1, milestones=101000),
        mode='step',
        warmup=dict(initial_coef=0.01, mode='linear', num_iters=0)),
    use_amp=True)

use_checkpoint = False

use_custom_worker_init = True

model_name = 'EffB1_ZoomNeXt_FSP'

output_dir = 'outputs_fsp_july13'

info = 'fsp_july13_restored'
