import argparse


class Options(object):
    def __init__(self):
        # Handle command line arguments
        self.parser = argparse.ArgumentParser(description="Run a complete training pipeline. Optionally, a JSON configuration file can be used, to overwrite command-line arguments.")

        # Run from config file
        self.parser.add_argument('--config', dest='config_filepath',
                                 help='Configuration .json file (optional). Overwrites existing command-line args!')

        # Run from command-line arguments
        # I/O
        self.parser.add_argument("--output_dir", default="./output", help="Root output directory. Must exist. Time-stamped directories will be created inside.",)
        self.parser.add_argument("--data_dir", default="./data", help="Data directory")
        self.parser.add_argument("--load_model", help="Path to pre-trained model.")
        self.parser.add_argument('--resume', action='store_true',
                                 help='If set, will load `starting_epoch` and state of optimizer, besides model weights.')
        self.parser.add_argument('--change_output', action='store_true',
                                 help='Whether the loaded model will be fine-tuned on a different task (necessitating a different output layer)')
        self.parser.add_argument('--save_all', action='store_true',
                                 help='If set, will save model weights (and optimizer state) for every epoch; otherwise just latest')
        self.parser.add_argument('--name', dest='experiment_name', default='',
                                 help='A string identifier/name for the experiment to be run - it will be appended to the output directory name, before the timestamp')
        self.parser.add_argument(
            '--comment', type=str, default='', help='A comment/description of the experiment')
        self.parser.add_argument('--no_timestamp', action='store_true',
                                 help='If set, a timestamp will not be appended to the output directory name')
        self.parser.add_argument('--records_file', default='./records.xls',
                                 help='Excel file keeping all records of experiments')
        # System
        self.parser.add_argument('--console', action='store_true',
                                 help="Optimize printout for console output; otherwise for file")
        self.parser.add_argument('--print_interval', type=int, default=1,
                                 help='Print batch info every this many batches')
        self.parser.add_argument('--gpu', type=str, default='0',
                                 help='GPU index, -1 for CPU')
        self.parser.add_argument('--n_proc', type=int, default=-1,
                                 help='Number of processes for data loading/preprocessing. By default, equals num. of available cores.')
        self.parser.add_argument('--num_workers', type=int, default=0,
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--seed', type=int,
                                 help='Seed used for splitting sets. None by default, set to an integer for reproducibility')

        # Dataset
        self.parser.add_argument('--limit_size', type=float, default=None,
                                 help="Limit  dataset to specified smaller random sample, e.g. for rapid debugging purposes. "
                                      "If in [0,1], it will be interpreted as a proportion of the dataset, "
                                      "otherwise as an integer absolute number of samples")
        self.parser.add_argument('--test_only', choices={'testset', 'fold_transduction'},
                                 help='If set, no training will take place; instead, trained model will be loaded and evaluated on test set')
        self.parser.add_argument('--data_class', type=str, default='weld',
                                 help="Which type of data should be processed.")
        self.parser.add_argument('--labels', type=str,
                                 help="In case a dataset contains several labels (multi-task), "
                                      "which type of labels should be used in regression or classification, i.e. name of column(s).")
        self.parser.add_argument('--shuffle_timesteps', action='store_true',
                                 help="""Whether timesteps should be shuffled.""")
        self.parser.add_argument('--test_from',
                                 help='If given, will read test IDs from specified text file containing sample IDs one in each row')
        self.parser.add_argument('--test_ratio', type=float, default=0,
                                 help="Set aside this proportion of the dataset as a test set")
        self.parser.add_argument('--val_ratio', type=float, default=0.2,
                                 help="Proportion of the dataset to be used as a validation set")
        self.parser.add_argument('--val_temporal_split', action='store_true', help="If set, split train/val set temporally.")
        self.parser.add_argument('--val_sequential_split', action='store_true', help="If set, split train/val set sequentially.")
        self.parser.add_argument('--pattern', type=str,
                                 help='Regex pattern used to select files contained in `data_dir`. If None, all data will be used.')
        self.parser.add_argument('--val_pattern', type=str, default=None,
                                 help="""Regex pattern used to select files contained in `data_dir` exclusively for the validation set.
                            If None, a positive `val_ratio` will be used to reserve part of the common data set.""")
        self.parser.add_argument('--test_pattern', type=str, default='TEST',
                                 help="""Regex pattern used to select files contained in `data_dir` exclusively for the test set.
                            If None, `test_ratio`, if specified, will be used to reserve part of the common data set.""")
        self.parser.add_argument('--normalization',
                                 choices={'standardization', 'minmax',
                                          'per_sample_std', 'per_sample_minmax'},
                                 default='standardization',
                                 help='If specified, will apply normalization on the input features of a dataset.')
        self.parser.add_argument('--normalize_label', action='store_true', 
                                 help="""If set, normalize the label (so the model predictions can have mean 0, std 1).""")
        self.parser.add_argument('--norm_from',
                                 help="""If given, will read normalization values (e.g. mean, std, min, max) from specified pickle file.
                            The columns correspond to features, rows correspond to mean, std or min, max.""")
        self.parser.add_argument('--subsample_factor', type=int,
                                 help='Sub-sampling factor used for long sequences: keep every kth sample')
        self.parser.add_argument('--baseline', type=int,
                                 help='Test a specific baseline for early prediction')
        self.parser.add_argument('--proportion', type=float,
                                 help='Proportion of dataset used for early prediction or autoregressive pretraining')

        # Training process
        self.parser.add_argument('--task', choices={"imputation", "transduction", "classification", "regression"},
                                 default="imputation",
                                 help=("Training objective/task: imputation of masked values,\n"
                                       "                          transduction of features to other features,\n"
                                       "                          classification of entire time series,\n"
                                       "                          regression of scalar(s) for entire time series"))
        self.parser.add_argument('--masking_ratio', type=float, default=0.15,
                                 help='Imputation: mask this proportion of each variable')
        self.parser.add_argument('--mean_mask_length', type=float, default=3,
                                 help="Imputation: the desired mean length of masked segments. Used only when `mask_distribution` is 'geometric'.")
        self.parser.add_argument('--mask_mode', choices={'separate', 'concurrent'}, default='separate',
                                 help=("Imputation: whether each variable should be masked separately "
                                       "or all variables at a certain positions should be masked concurrently"))
        self.parser.add_argument('--mask_distribution', choices={'geometric', 'bernoulli', 'autoregressive', 'early'}, default='geometric',
                                 help=("Imputation: whether each mask sequence element is sampled independently at random"
                                       "or whether sampling follows a markov chain (stateful), resulting in "
                                       "geometric distributions of masked squences of a desired mean_mask_length"))
        self.parser.add_argument('--exclude_feats', type=str, default=None,
                                 help='Imputation: Comma separated string of indices corresponding to features to be excluded from masking')
        self.parser.add_argument('--mask_feats', type=str, default='0, 1',
                                 help='Transduction: Comma separated string of indices corresponding to features to be masked')
        self.parser.add_argument('--start_hint', type=float, default=0.0,
                                 help='Transduction: proportion at the beginning of time series which will not be masked')
        self.parser.add_argument('--end_hint', type=float, default=0.0,
                                 help='Transduction: proportion at the end of time series which will not be masked')
        self.parser.add_argument('--harden', action='store_true',
                                 help='Makes training objective progressively harder, by masking more of the input')


        self.parser.add_argument('--epochs', type=int, default=400,
                                 help='Number of training epochs')
        self.parser.add_argument('--patience', type=int, default=1000,
                                 help='Early stopping patience (stop if val loss does not reach a new best after this many epochs)')
        self.parser.add_argument('--val_interval', type=int, default=2,
                                 help='Evaluate on validation set every this many epochs. Must be >= 1.')
        self.parser.add_argument(
            '--optimizer', choices={"Adam", "RAdam"}, default="Adam", help="Optimizer")
        self.parser.add_argument('--lr', type=float, default=1e-3,
                                 help='learning rate (default holds for batch size 64)')
        self.parser.add_argument('--lr_step', type=str, default='1000000',
                                 help='Comma separated string of epochs when to reduce learning rate by a factor of 10.'
                                      ' The default is a large value, meaning that the learning rate will not change.' \
                                      ' joshuafan: added option to set this to `plateauX` where X is the number of epochs to wait' \
                                      ' after a new best validation loss, before decreasing learning rate')
        self.parser.add_argument('--lr_factor', type=str, default='0.1',
                                 help=("Comma separated string of multiplicative factors to be applied to lr "
                                       "at corresponding steps specified in `lr_step`. If a single value is provided, "
                                       "it will be replicated to match the number of steps in `lr_step`."))
        self.parser.add_argument('--batch_size', type=int, default=64,
                                 help='Training batch size')
        self.parser.add_argument('--l2_reg', type=float, default=0,
                                 help='L2 weight regularization parameter')
        self.parser.add_argument('--l1_reg', type=float, default=0,
                                 help='L1 weight regularization parameter')
        self.parser.add_argument('--global_reg', action='store_true',
                                 help='If set, L2 regularization will be applied to all weights instead of only the output layer')
        self.parser.add_argument('--key_metric', choices={'loss', 'accuracy', 'precision'}, default='loss',
                                 help='Metric used for defining best epoch')
        self.parser.add_argument('--freeze', action='store_true',
                                 help='If set, freezes all layer parameters except for the output layer. Also removes dropout except before the output layer')
        self.parser.add_argument('--plot_accuracy', action='store_true',
                                 help='If set, plots a scatterplot of predicted and expected values of regression')
        self.parser.add_argument('--plot_loss', action='store_true',
                                 help='If set, plots a scatterplot of loss and loss with attention smoothness during supervised training')

        # Model
        self.parser.add_argument('--model', choices={"swin", "transformer", "LINEAR", "swin_pool", "smooth", "patch", "climax_smooth", "climax", "climax_smooth_pool", "convit", "convit_smooth", "convit_2", "climax_smooth_plot", "climax_max_pool", "climax_seqpool",
                                                     "ridge", "lasso", "random_forest", "xgboost", "local_cnn"}, default="transformer",  # "local_cnn2"
                                 help="Model class")
        self.parser.add_argument('--smooth_attention', action='store_true',
                                 help="""If set, will smooth adjacent attention weights.""")
        self.parser.add_argument('--agg_vars', action='store_true',
                                 help="""Only applicable for ClimaX. If set, creates separate tokens for individual variables, and uses cross-variable attention to aggregate. Otherwise, lumps all variables together into a single token.""")
        self.parser.add_argument('--conv_transformer', action='store_true',
                                 help="""Only applicable for ClimaX. If set, uses depthwise separable convolution as encoder.""")
        self.parser.add_argument('--local_mask', type=int, default=-1,
                                 help="""Only applicable for ClimaX. If set to a non-negative integer, restrict attention to tokens within this distance""")
        self.parser.add_argument('--causal_mask', action='store_true',
                                 help="""Only applicable for ClimaX. If set, uses causal mask in self-attention.""")
        self.parser.add_argument('--reg_lambda', type=float, default=0,
                                 help="""Regularizing weight for loss from attention smoothing.""")
        self.parser.add_argument('--reg_lambda_pool', type=float, default=0,
                                 help="""Regularizing weight for loss from POOLING attention smoothing.""")
        self.parser.add_argument('--lambda_posenc_smoothness', type=float, default=0,
                                 help="""Regularizing weight for loss for POS ENC smoothing.""")
        self.parser.add_argument('--lambda_locality', type=float, default=0, help="Weight for locality loss.")
        self.parser.add_argument('--lambda_erpe_linear', type=float, default=0, help="Weight for ERPE linear loss.")
        self.parser.add_argument('--lambda_focus', type=float, default=0, help="Weight for focus loss.")
        self.parser.add_argument('--lambda_jacobian', type=float, default=0, help="Weight for Jacobian reg loss.")
        self.parser.add_argument('--attention_type', choices=['dot', 'L2'], default='dot',
                                 help="""Type of self-attention mechanism.""")
        self.parser.add_argument('--learnable_scale', action='store_true',
                                 help="""If set, the attention scale factor becomes a learnable parameter.""")
        
        # Oversmoothing tracking
        self.parser.add_argument('--track_oversmoothing', action='store_true',
                                 help="Track oversmoothing metrics (effective rank, cosine similarity, attention entropy, high-freq ratio)")
        self.parser.add_argument('--oversmoothing_log_interval', type=int, default=10,
                                 help="Compute oversmoothing metrics every N batches (default: 10)")
        self.parser.add_argument('--pre_norm', action='store_true',
                                 help="""If set, normalize before attention.""")
        self.parser.add_argument('--qkv_identity_init', action='store_true',
                                 help="""If set, initialize W_Q, W_K, and W_V matrices to identity, so that at the beginning of training, attention is simply dot product similarity between input tokens. Only applicable if not using conv_projection.""")
        self.parser.add_argument('--tied_qk', action='store_true',
                                 help="""If set, use same matrix for W_Q and W_K.""")
        self.parser.add_argument('--key_bias', action='store_true',
                                 help="""If set, add bias (attention sink) as a column in key matrix.""")

        self.parser.add_argument('--max_seq_len', type=int,
                                 help="""Maximum input sequence length. Determines size of transformer layers.
                                 If not provided, then the value defined inside the data class will be used.""")
        self.parser.add_argument('--data_window_len', type=int,
                                 help="""Used instead of the `max_seq_len`, when the data samples must be
                                 segmented into windows. Determines maximum input sequence length
                                 (size of transformer layers).""")
        self.parser.add_argument('--d_model', type=int, default=64,
                                 help='Internal dimension of transformer embeddings')
        self.parser.add_argument('--dim_feedforward', type=int, default=256,
                                 help='Dimension of dense feedforward part of transformer layer')
        self.parser.add_argument('--num_heads', type=int, default=8,
                                 help='Number of multi-headed attention heads')
        self.parser.add_argument('--num_layers', type=int, default=3,
                                 help='Number of transformer encoder layers (blocks)')
        self.parser.add_argument('--num_gpsa_layers', type=int, default=0,
                                 help='Number of GPSA layers')
        self.parser.add_argument('--dropout', type=float, default=0.1,
                                 help='Dropout applied to most transformer encoder layers')
        self.parser.add_argument('--pos_encoding', choices={'fixed_sin', 'learnable', 'learnable_zero_init', 'learnable_uniform_init',
                                                            'learnable_sin_init', 'learnable_tape_init', 'none'}, default='learnable_uniform_init',
                                 help='Method for ABSOLUTE positional encoding. learnable defaults to learnable_uniform_init')
        self.parser.add_argument('--where_to_add_abspos', type=str, choices=["start_add", "start_concat", "before_pool_add", "before_pool_concat",
                                                                             "pooling_before_softmax", "pooling_gating"], default="start_add",
                                 help='Where to inject the absolute positional embedding: at start (add), before seqpool (add/concat), or as learnable offset in the pooling softmax.')
        self.parser.add_argument('--relative_pos_encoding', choices={'alibi', 'erpe_zero_init', 'erpe_uniform_init',
                                                                     'erpe_alibi_init', 'erpe_alibi_init_fixedslopes',
                                                                     'erpe_convit_init', 'erpe_convalibi_init', 'erpe_convalibi_init_dilated',
                                                                     'erpe_convalibi_init_quadratic', 
                                                                     'erpe_convalibi_init_quadratic_clamped', 'erpe_convalibi_init_clamped',
                                                                     'convit', 'convit_half', 'rope', 'none'}, default='none',
                                 help='Method for RELATIVE positional encoding')
        self.parser.add_argument('--where_to_add_relpos', type=str, choices=["before", "after", "after_gating", "only_relpos"], default="before",
                                 help="""Where to add relative position offset (before or after softmax). If `after_gating` is set, do a learnable gating (convit style) where the model can decide how much to weight position & content attention""")
        self.parser.add_argument('--convit_slope', type=float, default=1.0, help="Slope for Convit relative positional encoding. Higher means more focused (narrower receptive field).")
        self.parser.add_argument('--alibi_max_slope', type=float, default=4.0, help="Steepest slope for Alibi (FIRST LAYER). Higher means more focused (narrower receptive field).")
        self.parser.add_argument('--alibi_min_slope', type=float, default=0.25, help="Shallowest slope for Alibi (FIRST LAYER). Higher means more focused (narrower receptive field). TODO: Not sure if this should depend on the data.")

        self.parser.add_argument('--conv_projection', action='store_true',
                                 help="""If true, use conv instead of linear for Q/K/V""")
        self.parser.add_argument('--activation', choices={'relu', 'gelu'}, default='gelu',
                                 help='Activation to be used in transformer encoder')
        self.parser.add_argument('--normalization_layer', choices={'BatchNorm', 'LayerNorm'}, default='BatchNorm',
                                 help='Normalization layer to be used internally in transformer encoder')
        self.parser.add_argument('--class_token', action='store_true',
                                 help='If set, will append class token to help predict global class')
        self.parser.add_argument('--stride', type=int, default=1,
                                 help='Stride between patches or convolutions')
        self.parser.add_argument('--patch_length', type=int, default=1,
                                 help='Number of time steps in each patch')
        self.parser.add_argument('--num_decoder_layers', type=int, default=2,
                                 help='Number of decoder layers')


        # Input pooling
        self.parser.add_argument('--input_pooling_patch', type=int, default=1,
                                 help="Number of time steps in each input pooling patch. Only used with sklearn models at this time.")
        self.parser.add_argument('--input_pooling_stride', type=int, default=1,
                                 help="Stride for input pooling. Only used with sklearn models at this time.")

        # Pooling
        self.parser.add_argument('--pool', type=str, choices=['seqpool', 'average', 'linear', 'seqpool_multihead', 'seqpool_multihead_smoothed', 'seqpool_cls', 'maxpool', 'max_seq_hybrid', 'average_max'], default='linear',
                                 help='Type of final pooling')

        # Oversmoothing related
        self.parser.add_argument('--lambda_neutreno', type=float, default=0,
                                 help="Weight for NeuTRENO regularization. See https://arxiv.org/pdf/2312.00751")
        self.parser.add_argument('--attn_scale', action='store_true',
                                help="""If set, learn scaling factors (per layer/head) for the high-frequency component of attn matrix. See https://arxiv.org/pdf/2203.05962""")
        self.parser.add_argument('--feat_scale', action='store_true',
                                help="""If set, learn scaling factors (per layer/channel) for the high-frequency component of embeddings (post-attention). See https://arxiv.org/pdf/2203.05962""")
        self.parser.add_argument('--centered_attn', action='store_true',
                                help="""If set, subtract 1/T from attention matrix so that it sums to 0 instead of 1. See https://arxiv.org/pdf/2306.01610""")

        # Local-CNN specific
        self.parser.add_argument('--conv_type', type=str, choices=['hierarchical', 'local', 'per_timestep', 'lstm'], default='hierarchical',
                                 help='Type of CNN')
        # self.parser.add_argument('--local_cnn2_batch_norm', action='store_true', help='Set to use batchnorm in LocalCNN2.')
        # self.parser.add_argument('--local_cnn2_spectral_norm', action='store_true', help='Set to use spectral norm in LocalCNN2.')

        # Augmentation
        self.parser.add_argument('--input_noise_std', type=float, default=0.0)

        # C-Mixup specific
        self.parser.add_argument('--mixtype', type=str, default='none', choices=['none', 'kde', 'random', 'dtw'],
                                 help="If set to kde, random, or dtw, use C-Mixup with given sampling probabilities. kde is recommended (interpolate beteen points with nearby labels)")
        self.parser.add_argument('--kde_bandwidth', type=float, default=1.0,
                                 help="bandwidth")
        self.parser.add_argument('--mix_alpha', type=float, default=2)
        self.parser.add_argument('--kde_type', type=str, default='gaussian', help = 'gaussian or tophat')
        self.parser.add_argument('--batch_type', default=0, type=int, help='1 for y batch and 2 for x batch and 3 for representation')
        self.parser.add_argument('--show_process', type=int, default = 0,
                                 help = 'show rmse and r^2 in the process')


    def parse(self):
        args = self.parser.parse_args()

        args.lr_step = [int(i) for i in args.lr_step.split(",")] if not args.lr_step.startswith("plateau") else args.lr_step
        args.lr_factor = [float(i) for i in args.lr_factor.split(",")]
        if (len(args.lr_step) > 1) and (len(args.lr_factor) == 1):
            args.lr_factor = len(args.lr_step) * args.lr_factor  # replicate
        assert len(args.lr_step) == len(
            args.lr_factor
        ), "You must specify as many values in `lr_step` as in `lr_factors`"

        if args.exclude_feats is not None:
            args.exclude_feats = [int(i) for i in args.exclude_feats.split(",")]
        args.mask_feats = [int(i) for i in args.mask_feats.split(",")]

        if args.val_pattern is not None:
            args.val_ratio = 0
            args.test_ratio = 0

        return args
