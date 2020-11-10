import argparse

from str2bool import str2bool


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2-xl",
        help="Name of the model that we want to use.",
    )
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
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=256,
        help="Size of each sentence parsed by the model at each "
        "iteration.",
    )
    parser.add_argument(
        "--learning_rate",
        type=int,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=2500,
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
        default=4,
        help="Steps at what metrics are stored.",
    )
    parser.add_argument(
        "--metrics_freq",
        type=int,
        default=500,
        help="Steps at what metrics are stored.",
    )
    parser.add_argument(
        "--saving_freq",
        type=int,
        default=2000,
        help="Iters frequency at which models are stored.",
    )
    parser.add_argument(
        "--use_fp16",
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help="If True, the model trains with mixed precision.",
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

    return args
