import argparse
import random
from concurrent.futures import (
    ThreadPoolExecutor,
    wait,
    FIRST_COMPLETED,
    as_completed,
)
from pathlib import Path

import pm4py
from pm4py.algo.simulation.montecarlo import algorithm as montecarlo
from pm4py.algo.simulation.playout.process_tree import algorithm as tree_playout
from pm4py.algo.simulation.tree_generator import algorithm as tree_generator
from pm4py.objects.random_variables.constant0.random_variable import Constant0
from pm4py.objects.random_variables.exponential.random_variable import Exponential
from pm4py.objects.random_variables.lognormal.random_variable import LogNormal
from pm4py.objects.random_variables.normal.random_variable import Normal
from pm4py.objects.random_variables.uniform.random_variable import Uniform


def _sample_dirichlet(rng, alpha):
    samples = [rng.gammavariate(a, 1.0) for a in alpha]
    total = sum(samples) or 1.0
    return [value / total for value in samples]


def _categorical_sample(rng, options, probs):
    threshold = rng.random()
    cumulative = 0.0
    for option, prob in zip(options, probs):
        cumulative += prob
        if threshold <= cumulative:
            return option
    return options[-1]


def sample_ptandlog_parameters(rng):
    min_acts = max(2, int(round(rng.triangular(6, 14, 9))))
    max_acts = max(min_acts + 2, int(round(rng.triangular(min_acts + 5, min_acts + 25, min_acts + 12))))
    mode = int(round(rng.triangular(min_acts, max_acts, (min_acts + max_acts) / 2)))
    mode = min(max_acts, max(min_acts, mode))

    op_probs = _sample_dirichlet(rng, [3.0, 3.0, 2.0, 2.0, 0.5])

    return {
        "mode": mode,
        "min": min_acts,
        "max": max_acts,
        "sequence": op_probs[0],
        "choice": op_probs[1],
        "parallel": op_probs[2],
        "loop": op_probs[3],
        "or": op_probs[4],
        "silent": rng.betavariate(2, 8),
        "duplicate": rng.betavariate(1, 15),
        "no_models": 1,
    }


def sample_transition_random_variable(rng):
    rv_types = ["NORMAL", "EXPONENTIAL", "UNIFORM", "LOGNORMAL"]
    rv_probs = [0.4, 0.3, 0.2, 0.1]
    rv_type = _categorical_sample(rng, rv_types, rv_probs)

    if rv_type == "NORMAL":
        mu = rng.uniform(1.0, 12.0)
        sigma = rng.uniform(0.2, 3.5)
        rv = Normal(mu=mu, sigma=sigma)
    elif rv_type == "EXPONENTIAL":
        loc = rng.uniform(0.0, 2.0)
        scale = rng.uniform(0.5, 6.0)
        rv = Exponential(loc=loc, scale=scale)
    elif rv_type == "UNIFORM":
        loc = rng.uniform(0.0, 3.0)
        scale = rng.uniform(0.5, 8.0)
        rv = Uniform(loc=loc, scale=scale)
    else:
        s = rng.uniform(0.3, 1.2)
        loc = rng.uniform(0.0, 1.5)
        scale = rng.uniform(0.5, 6.0)
        rv = LogNormal(s=s, loc=loc, scale=scale)

    rv.set_weight(rng.gammavariate(2.0, 1.0))
    return rv


def build_stochastic_map(net, rng):
    smap = {}
    for transition in net.transitions:
        if transition.label is None:
            rv = Constant0()
            rv.set_weight(1.0)
        else:
            rv = sample_transition_random_variable(rng)
        smap[transition] = rv
    return smap


def log_has_empty_trace(log):
    for trace in log:
        if len(trace) == 0:
            return True
    return False


def simulate_single_log(index, output_path, args, max_attempts_per_log):
    rng = random.Random(args.seed + index)
    simulated_log = None
    for _attempt in range(max_attempts_per_log):
        pt_params = sample_ptandlog_parameters(rng)
        tree = tree_generator.apply(
            variant=tree_generator.Variants.PTANDLOGGENERATOR,
            parameters=pt_params,
        )

        net, im, fm = pm4py.convert_to_petri_net(tree)

        base_log = tree_playout.apply(
            tree,
            variant=tree_playout.Variants.BASIC_PLAYOUT,
            parameters={"num_traces": args.base_traces},
        )

        if log_has_empty_trace(base_log):
            continue

        smap = build_stochastic_map(net, rng)

        sim_parameters = {
            "provided_stochastic_map": smap,
            "num_simulations": args.traces_per_log,
            "case_arrival_ratio": args.case_arrival_ratio,
            "enable_diagnostics": False,
        }
        try:
            simulated_log, _sim_results = montecarlo.apply(
                base_log, net, im, fm, parameters=sim_parameters
            )
            break
        except IndexError:
            simulated_log = None
            continue

    if simulated_log is None:
        raise RuntimeError(
            "Unable to simulate a non-empty log after multiple attempts. "
            "Try reducing silent activity probability or increasing max attempts."
        )

    for case_index, trace in enumerate(simulated_log):
        trace.attributes["concept:name"] = str(case_index)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pm4py.write_xes(simulated_log, str(output_path), variant_str="line_by_line")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="PTAndLogGenerator -> Petri net -> Monte Carlo simulation with stochastic map."
    )
    parser.add_argument(
        "--output",
        default="simulated_log.xes",
        help=(
            "File or directory where simulated XES logs are stored. "
            "If multiple logs are requested, numbered files are created."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--num-logs",
        type=int,
        default=5,
        help="Number of event logs to simulate.",
    )
    parser.add_argument(
        "--traces-per-log",
        "--num-simulations",
        dest="traces_per_log",
        type=int,
        default=50,
        help="Number of traces (cases) to simulate per log.",
    )
    parser.add_argument(
        "--case-arrival-ratio",
        type=float,
        default=3600.0,
        help="Average case inter-arrival time in seconds.",
    )
    parser.add_argument(
        "--base-traces",
        type=int,
        default=50,
        help="Traces in the base log used to seed the simulation.",
    )
    parser.add_argument(
        "--threads",
        "--num-threads",
        type=int,
        default=0,
        help="Number of worker threads (0 = auto).",
    )
    parser.add_argument(
        "--log-timeout",
        type=float,
        default=0.0,
        help="Timeout in seconds per log; 0 disables timeout handling.",
    )
    args = parser.parse_args()

    output_base = Path(args.output)
    max_attempts_per_log = 10
    threads = args.threads if args.threads > 0 else min(4, args.num_logs)

    def resolve_output_path(index, total):
        if total == 1:
            return output_base
        output_name = output_base.name.lower()
        if output_name.endswith(".xes.gz"):
            base = output_base.name[:-7]
            suffix = ".xes.gz"
            return output_base.with_name(f"{base}_{index + 1}{suffix}")
        if output_name.endswith(".xes"):
            return output_base.with_name(
                f"{output_base.stem}_{index + 1}{output_base.suffix}"
            )
        return output_base / f"log_{index + 1}.xes"

    futures = {}
    failures = []
    with ThreadPoolExecutor(max_workers=threads) as executor:
        for i in range(args.num_logs):
            output_path = resolve_output_path(i, args.num_logs)
            future = executor.submit(
                simulate_single_log, i, output_path, args, max_attempts_per_log
            )
            futures[future] = (i, output_path)

        if args.log_timeout > 0:
            remaining = dict(futures)
            for i in range(args.num_logs):
                done, _pending = wait(
                    remaining,
                    timeout=args.log_timeout,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    break
                for future in done:
                    index, _output_path = remaining.pop(future)
                    try:
                        result_path = future.result()
                        print(f"Simulated log saved to: {result_path}")
                    except Exception as exc:
                        failures.append((index, str(exc)))
                        print(f"Log {index + 1} failed: {exc}")
            for future, (index, _output_path) in remaining.items():
                future.cancel()
                failures.append((index, "Timeout"))
                print(f"Log {index + 1} timed out and was skipped.")
        else:
            for future in as_completed(futures):
                index, _output_path = futures[future]
                try:
                    result_path = future.result()
                    print(f"Simulated log saved to: {result_path}")
                except Exception as exc:
                    failures.append((index, str(exc)))
                    print(f"Log {index + 1} failed: {exc}")

    if failures:
        failure_count = len(failures)
        print(f"{failure_count} log(s) failed or timed out. See messages above.")


if __name__ == "__main__":
    main()
