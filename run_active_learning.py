"""
AUTHOR: Kharanshu Bhojak, Manpreet Singh, Tanvir H. Pantha, Sourav Dutta
University of Texas at Dallas, USA

Date: September 2026

Bayesian Active Learning: suggest the next experiment.

Reads every experiment in an Excel sheet, builds one Figure of Merit (FOM) from the
optimization parameters, fits a Gaussian process of FOM vs. the process parameters,
and prints the next process point to sample (UCB acquisition, FOM is maximized).

Example
-------
python run_active_learning.py --data Lookup_Dataset.xlsx \
    --process-params "Oxide Dep Temp" "Anneal Temp" "Hf Composition" \
    --objectives Vt SS Ion Mobility

NOTE: Parameters whose names are not on the built-in list of common parameters must be
named under --minimize or --maximize.
"""

import argparse
import os

import numpy as np

from al_functions import load_dataset, suggest_next_point


def parse_bounds(items):
    bounds = {}
    for item in items or []:
        try:
            name, rng = item.rsplit("=", 1)
            lo, hi = (float(v) for v in rng.split(","))
        except ValueError:
            raise SystemExit(f"Bad --bounds entry '{item}'. Use NAME=LOW,HIGH")
        bounds[name.strip()] = (lo, hi)
    return bounds


def build_parser():
    p = argparse.ArgumentParser(
        description="Suggest the next experiment with Bayesian active learning "
                    "(GP surrogate + UCB on a combined FOM).")
    p.add_argument("--data", required=True, help="Excel file with all experiments so far")
    p.add_argument("--sheet", default=0, help="Sheet name or index (default: first sheet)")
    p.add_argument("--process-params", nargs="+", required=True, metavar="COL",
                   help="Process-parameter columns (any number)")
    p.add_argument("--objectives", nargs="+", default=[], metavar="COL",
                   help="Optimization-parameter columns; direction is detected from "
                        "the built-in list of common parameter names")
    p.add_argument("--minimize", nargs="+", default=[], metavar="COL",
                   help="Optimization parameters to minimize (overrides auto-detection)")
    p.add_argument("--maximize", nargs="+", default=[], metavar="COL",
                   help="Optimization parameters to maximize (overrides auto-detection)")
    p.add_argument("--vt-target", type=float, default=0.0,
                   help="Target Vt for |Vt - Vt_target| (default 0)")
    p.add_argument("--bounds", nargs="+", metavar="NAME=LOW,HIGH",
                   help="Search range per process parameter (default: min/max in the data)")
    p.add_argument("--kappa", type=float, default=2.0,
                   help="UCB exploration weight (default 2.0)")
    p.add_argument("--n-grid", type=int, nargs="+", default=[7], metavar="N",
                   help="Grid points per process parameter: one value for all axes, or "
                        "one value per parameter in the order of --process-params "
                        "(e.g. 5 7 6). Default 7")
    p.add_argument("--max-candidates", type=int, default=2_000_000,
                   help="Safety cap on the total grid size (default 2,000,000)")
    p.add_argument("--n-restarts", type=int, default=15,
                   help="GP optimizer restarts (default 15)")
    p.add_argument("--top-k", type=int, default=5,
                   help="How many top candidates to report (default 5)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for the GP optimizer restarts (reproducibility)")
    p.add_argument("--output-dir", default="results", help="Folder for CSV output")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if not (args.objectives or args.minimize or args.maximize):
        raise SystemExit("Give at least one optimization parameter via "
                         "--objectives, --minimize or --maximize.")

    if args.seed is not None:
        np.random.seed(args.seed)   # fit_gp draws its restarts from the global RNG

    sheet = args.sheet
    if isinstance(sheet, str) and sheet.isdigit():
        sheet = int(sheet)

    df = load_dataset(args.data, sheet_name=sheet)   # raises FileNotFoundError if missing

    res = suggest_next_point(
        df,
        process_params=args.process_params,
        objective_columns=args.objectives,
        minimize=args.minimize,
        maximize=args.maximize,
        vt_target=args.vt_target,
        bounds=parse_bounds(args.bounds),
        kappa=args.kappa,
        n_grid=args.n_grid[0] if len(args.n_grid) == 1 else args.n_grid,
        max_candidates=args.max_candidates,
        n_restarts=args.n_restarts,
        top_k=args.top_k,
    )

    print(f"Experiments used : {len(res['data'])}")
    print("Directions       : " + ", ".join(
        f"{c} -> {'minimize' if d == 'min' else 'maximize'}"
        for c, d in res["directions"].items()))
    print(f"GP               : LML={res['log_marginal_likelihood']:.2f} | "
          f"LS={np.round(res['length_scales'], 2)}")
    print(f"Best FOM so far  : {res['data']['FOM'].max():.4f}")
    print("\nNext point to sample:")
    for name, val in res["next_point"].items():
        print(f"  {name:<25s} {val:.6g}")
    print(f"\n  predicted FOM = {res['predicted_fom']:.4f} "
          f"± {res['predicted_std']:.4f}   (UCB = {res['ucb']:.4f}, kappa = {args.kappa})")

    os.makedirs(args.output_dir, exist_ok=True)
    res["next_point"].to_frame("next_point").T.assign(
        predicted_FOM=res["predicted_fom"], predicted_std=res["predicted_std"],
        UCB=res["ucb"]).to_csv(os.path.join(args.output_dir, "next_point.csv"), index=False)
    res["top_candidates"].to_csv(os.path.join(args.output_dir, "top_candidates.csv"), index=False)
    res["data"].to_csv(os.path.join(args.output_dir, "fom_table.csv"), index=False)
    print(f"\nSaved next_point.csv, top_candidates.csv, fom_table.csv in '{args.output_dir}/'")


if __name__ == "__main__":
    main()
