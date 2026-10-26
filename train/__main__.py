"""Entry point for `python -m train` — HyTwin network RL trainer."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hytwin.rl.network_trainer import train_network_agent
from hytwin.simulation.scenario import Scenario

DEFAULT_CONFIG = str(ROOT / "config" / "italy_network_large.yaml")
DEFAULT_SAVE   = str(ROOT / "output" / "rl_models" / "net_ppo_large")

def main() -> None:
    parser = argparse.ArgumentParser(description="HyTwin — Network RL Trainer (PPO)")
    parser.add_argument("--config",     default=DEFAULT_CONFIG, help="Network scenario YAML")
    parser.add_argument("--timesteps",  type=int,   default=200_000, help="Training timesteps (default: 200000)")
    parser.add_argument("--n-envs",     type=int,   default=4,       help="Parallel environments (default: 4)")
    parser.add_argument("--n-steps",    type=int,   default=576,     help="Rollout steps per env (default: 576)")
    parser.add_argument("--seed",       type=int,   default=0,       help="Random seed (default: 0)")
    parser.add_argument("--save",       default=DEFAULT_SAVE,        help="Output path without .zip extension")
    parser.add_argument("--device",     default="auto",              help="PyTorch device: auto|cpu|cuda (default: auto)")
    parser.add_argument("--verbose",    type=int,   default=1,       help="SB3 verbosity 0/1 (default: 1)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  HyTwin — Network RL Training")
    print(f"{'='*60}")
    print(f"  Config    : {args.config}")
    print(f"  Timesteps : {args.timesteps:,}")
    print(f"  Envs      : {args.n_envs}  (parallel workers)")
    print(f"  n_steps   : {args.n_steps}")
    print(f"  Seed      : {args.seed}")
    print(f"  Device    : {args.device}")
    print(f"  Save to   : {args.save}.zip")
    print(f"  Press Ctrl-C to stop early\n")

    topo = Scenario.from_yaml(args.config).topology()
    print(f"  Network: {len(topo.sites)} sites, {len(topo.links)} links\n")

    train_network_agent(
        topo,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        seed=args.seed,
        save_path=args.save,
        device=args.device,
        verbose=args.verbose,
    )
    print(f"\n  Modello salvato → {args.save}.zip")
    print(f"{'='*60}\n")


if __name__ == "__main__":  # needed by the "spawn" start method (Windows)
    main()
