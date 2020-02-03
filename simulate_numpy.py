from argparse import ArgumentParser

import cicero


def dispatch_random_simulation(width, height, args):
    gol = cicero.np_game_of_life.ToroidalGOL.from_random_state(
        width=width,
        height=height,
        alive_probability=args.alive_probability,
        convergence_memory=args.convergence_detector_window)
    gol.simulate(steps=args.max_steps, break_on_convergence=args.break_on_convergence)
    return gol.history


def dispatch_user_defined_simulation(args):
    gol = cicero.np_game_of_life.ToroidalGOL.from_json(
        json_path=args.initial_state,
        convergence_memory=args.convergence_detector_window)
    gol.simulate(steps=args.max_steps, break_on_convergence=args.break_on_convergence, verbose=1)
    return gol.history


def main():
    descr = """Toroidal Game of Life
    
    Use this script to execute one-shot simulations.
    """

    parser = ArgumentParser("Cicero - NumPy simulation", description=descr)
    parser.add_argument("--initial-state", default="random", type=str,
                        help="Pass a JSON file. See examples under configuration/. Defaults to 'random'")
    parser.add_argument("--convergence-detector-window", default=30, type=int,
                        help="Slice size, on which convergence will be determined. Default: 30")
    parser.add_argument("--break-on-convergence", default=False, type=bool,
                        help="Whether to end the simulation on convergence. Default: False")
    parser.add_argument("--replay-fps", default=5, type=int,
                        help="Replay speed in frames per second. Default: 5")
    parser.add_argument("--init-probability", default=None, type=int,
                        help="If set, the random initialization will produce active cells in this rate. Default: None")
    parser.add_argument("--size", default="6x6", type=str,
                        help="Simulation grid size. Default: 6x6")
    parser.add_argument("--max-steps", default=100, type=int,
                        help="How many steps to run the simulation for, if 'break-on-convergence' is False. "
                             "Default: 100")

    args = parser.parse_args()
    width, height = map(int, args.size.split("x"))

    if args.initial_state == "random":
        history = dispatch_random_simulation(width=width,
                                             height=height,
                                             args=args)

    else:
        history = dispatch_user_defined_simulation(args=args)

    cicero.visual.plot_flattened_history(history)
    cicero.visual.replay_simulation(history, fps=args.replay_fps, repeats=1)


if __name__ == '__main__':
    main()
