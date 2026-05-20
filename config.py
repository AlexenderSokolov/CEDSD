# Training parameters.
class TrainConfig:
    # Core optimization settings.
    lr: float = 1e-4
    batch_size: int = 24
    num_epochs: int = 30
    max_steps_per_epoch: int = 15000
    min_lr: float = 1e-6
    print_every_n_steps: int = 500
    loss_empty_eps: float = 1e-8
    skip_batch_warn_patience: int = 5
    text_empty_batch_skip_ratio: float = 0.50
    text_empty_batch_warn_ratio: float = 0.25
    text_empty_path_patience: int = 3
    text_repeat_path_hard_skip_enable: bool = False
    text_repeat_path_hard_skip_min_count: int = 2
    text_repeat_path_hard_skip_ratio: float = 0.50
    # Neutral Mandarin placeholders used only when ASR returns empty text.
    empty_text_placeholder_candidates = ["嗯", "啊", "好的", "知道了", "收到", "行", "可以", "这样吧"]
    empty_text_placeholder_seed: int = 42
    empty_text_offline_filter_enable: bool = True
    empty_text_offline_min_count: int = 3
    empty_text_offline_audit_file: str = "filtered_empty_text_samples.csv"
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    use_feature_cache: bool = True
    feature_cache_dir: str = "feature_cache"
    train_log_file: str = "train.log"
    # Isolate artifacts by run id to keep concurrent experiments separate.
    output_dir_name: str = "outputs_fapi_viz"
    run_id: str = ""
    auto_timestamp_run_id: bool = True
    isolate_run_artifacts: bool = True
    best_model_file: str = "best_model.pth"
    asr_offline_cache_enable: bool = True
    asr_offline_cache_file: str = "offline_cache_store/offline_asr_cache.csv"
    asr_offline_cache_file_test: str = "offline_cache_store/offline_asr_cache_test.csv"
    fapi_offline_cache_enable: bool = True
    fapi_offline_cache_file: str = "offline_cache_store/offline_fapi_stats_cache.csv"
    fapi_offline_cache_file_test: str = "offline_cache_store/offline_fapi_stats_cache_test.csv"
    e2v_offline_cache_enable: bool = True
    e2v_offline_cache_dir: str = "offline_cache_store/e2v_cache"
    acoustic_offline_cache_enable: bool = True
    acoustic_offline_cache_dir: str = "offline_cache_store/acoustic_cache"
    inference_temperature: float = 1.0
    temperature_grid_search_enable: bool = True
    temperature_candidates = (1.0, 1.2, 1.5, 2.0, 2.5, 3.0)
    temperature_grid_search_apply_best: bool = True

    # Mixed precision training.
    use_amp: bool = True
    amp_dtype: str = "bf16"

    # FAPI penalty policy.
    fapi_penalty_every_n_steps: int = 1
    fapi_penalty_reuse_enable: bool = False
    use_fapi_loss: bool = False

    # DataLoader prefetching and persistent-worker settings.
    dataloader_persistent_workers: bool = False
    dataloader_prefetch_factor: int = 1
    train_num_workers: int = 2
    val_num_workers: int = 1
    test_num_workers: int = 1
    pred_num_workers: int = 1


    # Quick mode shortens runs for iteration and smoke tests.
    quick_mode: bool = False
    quick_train_rows: int = 2000
    quick_val_rows: int = 500
    quick_test_rows: int = 500
    quick_predict_rows: int = 64
    quick_num_workers: int = 6

    # Convergence summary settings.
    convergence_summary_file: str = "convergence_summary.csv"
    convergence_auc_tol: float = 0.002
    convergence_f1_tol: float = 0.002
    convergence_eer_tol: float = 0.002

    # Joint loss weights.
    lambda1: float = 0.2   # KL emotion
    lambda2: float = 0.1   # conflict regularization
    lambda3: float = 0.05  # FAPI penalty
    # Uncertainty-weighting control: keep the uncertainty regularizer from dominating.
    uw_lr_ratio: float = 0.05
    uw_s_min: float = -0.18
    uw_s_max: float = 0.25
    uw_reg_coef: float = 0.5
    uw_ratio_target_low: float = 0.15
    uw_ratio_target_high: float = 0.35
    uw_ratio_warn: float = 0.50
    uw_ratio_critical: float = 0.60
    uw_ratio_ema_beta: float = 0.90
    uw_ratio_patience: int = 3
    # Automatic damping for unstable uncertainty-weighting ratios.
    uw_auto_enable: bool = True
    uw_auto_decay: float = 0.85
    uw_auto_floor: float = 0.15
    uw_auto_cooldown_steps: int = 400
    uw_auto_s_raise: float = 0.02
    # Slow recovery once ratios return to the target interval.
    uw_auto_recover_enable: bool = True
    uw_auto_recover_growth: float = 1.05
    uw_auto_recover_ceiling: float = 0.50
    uw_auto_recover_patience: int = 6
    # ===== UQ AUTO-RECOVER END =====
    # ===== UQ AUTO-FIX END =====
    # ===== UQ CONTROL END =====
    min_lambda: float = 1e-4
    margin: float = 0.5
    fake_label_threshold: float = 0.5
    adaptive_margin: bool = True
    adaptive_margin_ratio: float = 1.05

    # FAPI parameters.
    lm_name: str = "Qwen/Qwen2.5-3B"
    lm_device: str = "cuda"
    lm_precision: str = "fp16"
    gamma_base: float = 10.0
    topk: int = 5
    p_threshold: float = 385.4104
    fapi_lambda_floor: float = 0.15
    fapi_lambda_ceiling: float = 0.85
    fapi_penalty_scale: float = 1e5
    threshold_mode: str = "f1"
    ema_alpha: float = 0.9
    debug_fapi_graph: bool = False

    # IACA trainable initialization values.
    omega1_init: float = 2.0
    omega2_init: float = 1.0
    tau_init: float = 0.5

    # Architecture parameters.
    d_model: int = 256
    n_emotions: int = 9
