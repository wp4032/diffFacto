# model settings
cimle=True
cimle_cache_interval=200
cimle_start_epoch=6000
model = dict(
    type='AnchorDiffAE',
    encoder=dict(
        type='PartEncoderForTransformerDecoderKLLossCIMLE',
        encoder=dict(
            type='PointNetV2VAE',
            zdim=256,
            point_dim=3,
        ),
        part_aligner=dict(
            type="PartAlignerTransformer",
            in_channels = 256,
            out_channels=6,
            n_class=4,
            d_head=32,
            depth=5,
            n_heads=8,
            dropout=0.,
            use_checkpoint=False,
            use_linear=True,
            class_cond=True,
            single_attn=True,
            cimle=True,
            noise_scale=30,
        ),
        n_class=4,
        kl_weight=0.001,
        supervise_on_valid_id=False,
        fit_loss_weight=1.0,
        include_z=False,
        include_part_code=True,
        include_params=True,
    ),
    diffusion=dict(
        type='AnchoredDiffusion',
        net = dict(
            type='TransformerNet',
            in_channels=3,
            out_channels=3,
            n_heads=8,
            d_head=16,
            depth=5,
            dropout=0,
            context_dim=256 + 6,
            n_class=4,
            class_cond=True,
            use_linear=True,
            cat_params_to_x=True,
            use_checkpoint=False,
            single_attn=True,
            cat_class_to_x=True,
        ),
        beta_1=1e-4,
        beta_T=.02,
        k=1.0,
        res=False,
        mode='linear',
        use_beta=False,
        rescale_timesteps=False,
        model_mean_type="scaled_epsilon",
        learn_variance=True,
        loss_type='mse',
        include_anchors=False,
        
        classifier_weight=1.,
        guidance=False,
        ddim_sampling=False,
        ddim_nsteps=25,
        ddim_discretize='quad',
        ddim_eta=1.
    ),
    sampler = dict(type='Uniform'),
    num_anchors=4,
    num_timesteps=100,
    npoints = 2048,
    anchor_loss_weight=1.,
    
    gen=True,
    ret_traj = False,
    ret_interval = 10,
    forward_sample=False,
    interpolate=False,
    save_weights=False,
    cimle=True,
    save_dir="/mnt/disk3/wang/diffusion/anchorDIff/work_dirs/pn_aware_attn_both_feat_guidance_200T_no_augment/checkpoints"
)

dataset = dict(
    train=dict(
        type="ShapeNetSegPart",
        batch_size = 128,
        split='trainval',
        root='/mnt/disk1/wang/workspace/datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal',
        npoints=2048,
        scale_mode='shape_unit',
        drop_last=False,
        eval_mode='gen',
        num_workers=4,
        class_choice='Chair',
    ),
    val=dict(
        type="ShapeNetSegPart",
        batch_size=32,
        split='test',
        root='/mnt/disk1/wang/workspace/datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal',
        npoints=2048,
        shuffle=False,
        scale_mode='shape_unit',
        eval_mode='gen',
        drop_last=False,
        num_workers=0,
        class_choice='Chair',
        save_only=True
    ),
)

optimizer = dict(type='Adam', lr=0.002, weight_decay=0.)

scheduler = dict(
    type='StepLR',
    step_size=2666,
    gamma=0.5,
)

logger = dict(
    type="RunLogger")

# when we the trained model from cshuan, image is rgb
resume_path="/mnt/disk1/wang/workspace/anchorDIff/work_dirs/anchordiff_exp308/checkpoints/ckpt_6000.pth"
save_num_batch = 1
max_epoch = 8000
eval_interval = 500
checkpoint_interval = 500
log_interval = 50
max_norm=10
# eval_both=True