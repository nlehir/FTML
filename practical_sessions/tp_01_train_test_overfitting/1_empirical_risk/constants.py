from argument_parser import args

MEAN_NOISE = 0
STD_NOISE = 10
N_SAMPLES = 600

if args.std:
    STD_NOISE = float(args.std)
