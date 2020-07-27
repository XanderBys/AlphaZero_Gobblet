import logging, sys
fname = "logs/{}.log".format(sys.argv[0][:-3])
logging.basicConfig(filename=fname, level=logging.INFO)
