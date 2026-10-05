import argparse
from pathlib import Path
from .env import ChainConfig
from .experiment import RunConfig, run_comparison
from .trpo import TRPOConfig


def main():
    parser = argparse.ArgumentParser(description="Compare static state and state-action count bonuses using TRPO.")
    parser.add_argument("--output", type=Path, default=Path("results/comparison"))
    parser.add_argument("--modes", nargs="+", choices=["none", "state", "state_action"], default=["none", "state", "state_action"])
    parser.add_argument("--betas", nargs="+", type=float, default=[0.01])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--slip", type=float, default=0.0)
    parser.add_argument("--small", type=float, default=2.0)
    parser.add_argument("--large", type=float, default=10.0)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--hashing", choices=["simhash", "exact"], default="simhash")
    parser.add_argument("--hash-bits", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=1.0)
    parser.add_argument("--max-kl", type=float, default=0.01)
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument("--cg-steps", type=int, default=10)
    parser.add_argument("--backtracks", type=int, default=10)
    parser.add_argument("--hidden", nargs="+", type=int, default=[32, 32])
    args = parser.parse_args()
    try:
        chain = ChainConfig(args.n, args.slip, args.small, args.large, args.horizon)
        optimizer = TRPOConfig(args.gamma, args.gae_lambda, args.max_kl, args.damping,
                               args.cg_steps, args.backtracks, tuple(args.hidden))
        run = RunConfig(args.iterations, args.batch_size, args.hashing, args.hash_bits)
        run_comparison(args.output, chain, optimizer, run, args.modes, args.betas, args.seeds)
    except ValueError as error:
        parser.error(str(error))
    print(f"Saved comparison to {args.output.resolve()}")


if __name__ == "__main__":
    main()
