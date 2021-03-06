import argparse

from str2bool import str2bool


def get_common_args(parser: argparse.ArgumentParser,):
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2-xl",
        help="Name of the model that we want to use.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=256,
        help="Size of each sentence parsed by the model at each "
        "iteration.",
    )
    parser.add_argument(
        "--recompute_data",
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help="If True, training data is recomputed",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(""),
        help="Prompt to evaluate results.",
    )

    args = parser.parse_args()

    return parser, args


def get_train_args():
    parser = argparse.ArgumentParser()
    parser, _args = get_common_args(parser)

    parser.add_argument(
        "--log_dir",
        type=str,
        default="./screenwriter/tensorboard-logs",
        help="Directory where tensorboard logs will be saved.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./screenwriter/checkpoints",
        help="Directory where models will be stored",
    )
    parser.add_argument(
        "--gen_log_file",
        type=str,
        default="./screenwriter/train_generations.txt",
        help="File where training generations will be stored.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=10000,
        help="Number of warmup steps.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size.",
    )
    parser.add_argument(
        "--num_grad_accum",
        type=int,
        default=16,
        help="Steps at what metrics are stored.",
    )
    parser.add_argument(
        "--metrics_freq",
        type=int,
        default=10,
        help="Steps at what metrics are stored.",
    )
    parser.add_argument(
        "--generation_freq",
        type=int,
        default=1000,
        help="Steps at what metrics are stored.",
    )
    parser.add_argument(
        "--generation_limit",
        type=int,
        default=256,
        help="Maximum number of tokens for each generation.",
    )
    parser.add_argument(
        "--saving_freq",
        type=int,
        default=5000,
        help="Iters frequency at which models are stored.",
    )

    args = parser.parse_args()

    return args


def get_generate_args():
    parser = argparse.ArgumentParser()
    parser, _args = get_common_args(parser)
    
    parser.add_argument(
        "--max_context_len",
        type=int,
        default=256,
        help="Context to take into account for each generation.",
    )
    parser.add_argument(
        "--max_generation_len",
        type=int,
        default=4048,
        help="Maximum number of tokens that can be generated.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./screenwriter/generation.txt",
        help="File where the results will be stored.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=50,
        help="Number of max probability tokens to take into account for each "
        "prediction.",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.95,
        help="Percentile where the selected tokens will be sampled from.",
    )

    args = parser.parse_args()

    return args
