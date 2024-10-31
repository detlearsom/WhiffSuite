import os
import sys
import json
import datetime
import data_manip
import ml
from cli import CLI
import inspect

def main(args):
    """
    Creates a new CLI object and starts parsing arguments.

    :param args: The provided arguments
    """
    try:
        cli = CLI()
        cli.parse_arguments(args)
    except Exception as _:
        exc_type, _, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

    if cli.args.metriclist:
        cli.processMetricListing()
        exit()
    if cli.args.snifflist:
        cli.processSniffTestListing()
        exit()
    if cli.args.list == True:
        print(cli.manipultor.processed_df[cli.manipulator.label_field].value_counts())
        exit()

    if cli.args.sniff != None:
        out = cli.chooseSniffTest()
        exit()

    if cli.args.metric != None:
        out = cli.chooseMetric()
        print(out)
        exit()
        now = datetime.datetime.now().strftime("%Y-%d-%B-%I-%M-%S")
        out_file = os.path.join(cli.args.results, cli.args.target, "metrics", "{}_{}.results".format(cli.args.metric, now))
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w+") as f:
            f.write(json.dumps(out))
    if cli.args.ids != None:
        ml.mlPipeline(cli.df, cli.metadata, cli.args.target, cli.args.ids, cli.args.results, cli.args.verbose)

if __name__ == "__main__":
    main(sys.argv[1:])
