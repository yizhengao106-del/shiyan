base_seed = 112358
data_cfg = 'dataset.yaml'
dataset_infos = dict(
    camo_te=dict(
        image=dict(path='Imgs', suffix='.jpg'),
        mask=dict(path='GT', suffix='.png'),
        root='/home/gyz/ZoomNeXt/datasets/COD-TestDataset/CAMO'),
    camo_tr=dict(
        image=dict(path='Image', suffix='.jpg'),
        mask=dict(path='Mask/GT', suffix='.png'),
        root='/home/gyz/ZoomNeXt/datasets/COD/Train/CAMO-TR'),
    chameleon=dict(
        image=dict(path='Imgs', suffix='.jpg'),
        mask=dict(path='GT', suffix='.png'),
        root='/home/gyz/ZoomNeXt/datasets/COD-TestDataset/CHAMELEON'),
    cod10k_te=dict(
        image=dict(path='Imgs', suffix='.jpg'),
        mask=dict(path='GT', suffix='.png'),
        root='/home/gyz/ZoomNeXt/datasets/COD-TestDataset/COD10K'),
    cod10k_tr=dict(
        image=dict(path='Image', suffix='.jpg'),
        mask=dict(path='Mask', suffix='.png'),
        root='/home/gyz/ZoomNeXt/datasets/COD/Train/COD10K-TR'),
    nc4k=dict(
        image=dict(path='Imgs', suffix='.jpg'),
        mask=dict(path='GT', suffix='.png'),
        root='/home/gyz/ZoomNeXt/datasets/COD-TestDataset/NC4K'))
deterministic = True
device = 'cuda:0'
exp_name = 'EffB1_ZoomNeXt_WeightGuided_BS4_LR0.0001_E150_H384_W384_OPMadam_OPGMfinetune_SCstep_AMP_INFOwg_fder'
has_test = True
info = 'wg_fder'
load_from = None
log_interval = 20
loss_cfg = dict(edge_loss_weight=0.3)
metric_names = [
    'sm',
    'wfm',
    'mae',
    'em',
    'fmeasure',
]
model_cfg = dict(
    edge_loss_weight=0.3,
    guided_stages=(
        3,
        2,
    ),
    use_edge=True,
    use_frequency=True,
    use_weight_guidance=False)
model_name = 'EffB1_ZoomNeXt_WeightGuided'
output_dir = '/home/gyz/ZoomNeXt/weight_guided_runs/20260910_214431_fder_5vdmzi1s/results'
path = dict(
    cfg_copy=
    '/home/gyz/ZoomNeXt/weight_guided_runs/20260910_214431_fder_5vdmzi1s/results/EffB1_ZoomNeXt_WeightGuided_BS4_LR0.0001_E150_H384_W384_OPMadam_OPGMfinetune_SCstep_AMP_INFOwg_fder/exp_0/config.py',
    excel=
    '/home/gyz/ZoomNeXt/weight_guided_runs/20260910_214431_fder_5vdmzi1s/results/EffB1_ZoomNeXt_WeightGuided_BS4_LR0.0001_E150_H384_W384_OPMadam_OPGMfinetune_SCstep_AMP_INFOwg_fder/exp_0/results.xlsx',
    final_full_net=
    '/home/gyz/ZoomNeXt/weight_guided_runs/20260910_214431_fder_5vdmzi1s/results/EffB1_ZoomNeXt_WeightGuided_BS4_LR0.0001_E150_H384_W384_OPMadam_OPGMfinetune_SCstep_AMP_INFOwg_fder/exp_0/pth/checkpoint_final.pth',
    final_state_net=
    '/home/gyz/ZoomNeXt/weight_guided_runs/20260910_214431_fder_5vdmzi1s/results/EffB1_ZoomNeXt_WeightGuided_BS4_LR0.0001_E150_H384_W384_OPMadam_OPGMfinetune_SCstep_AMP_INFOwg_fder/exp_0/pth/state_final.pth',
    log=
    '/home/gyz/ZoomNeXt/weight_guided_runs/20260910_214431_fder_5vdmzi1s/results/EffB1_ZoomNeXt_WeightGuided_BS4_LR0.0001_E150_H384_W384_OPMadam_OPGMfinetune_SCstep_AMP_INFOwg_fder/exp_0/log_2026-09-10.txt',
    output_dir=
    '/home/gyz/ZoomNeXt/weight_guided_runs/20260910_214431_fder_5vdmzi1s/results',
    pth=
    '/home/gyz/ZoomNeXt/weight_guided_runs/20260910_214431_fder_5vdmzi1s/results/EffB1_ZoomNeXt_WeightGuided_BS4_LR0.0001_E150_H384_W384_OPMadam_OPGMfinetune_SCstep_AMP_INFOwg_fder/exp_0/pth',
    pth_log=
    '/home/gyz/ZoomNeXt/weight_guided_runs/20260910_214431_fder_5vdmzi1s/results/EffB1_ZoomNeXt_WeightGuided_BS4_LR0.0001_E150_H384_W384_OPMadam_OPGMfinetune_SCstep_AMP_INFOwg_fder/exp_0',
    save=
    '/home/gyz/ZoomNeXt/weight_guided_runs/20260910_214431_fder_5vdmzi1s/results/EffB1_ZoomNeXt_WeightGuided_BS4_LR0.0001_E150_H384_W384_OPMadam_OPGMfinetune_SCstep_AMP_INFOwg_fder/exp_0/pre',
    tb=
    '/home/gyz/ZoomNeXt/weight_guided_runs/20260910_214431_fder_5vdmzi1s/results/EffB1_ZoomNeXt_WeightGuided_BS4_LR0.0001_E150_H384_W384_OPMadam_OPGMfinetune_SCstep_AMP_INFOwg_fder/exp_0/tb',
    trainer_copy=
    '/home/gyz/ZoomNeXt/weight_guided_runs/20260910_214431_fder_5vdmzi1s/results/EffB1_ZoomNeXt_WeightGuided_BS4_LR0.0001_E150_H384_W384_OPMadam_OPGMfinetune_SCstep_AMP_INFOwg_fder/exp_0/trainer.txt'
)
pretrained = True
proj_root = '/home/gyz/ZoomNeXt/weight_guided_runs/20260910_214431_fder_5vdmzi1s/source'
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
