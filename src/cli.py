import argparse
import inspect
import difflib
import metrics
import data_manip
import sniffs
import os
import warnings

class CLI(object):
    def __init__(self):
        """
        Creates a new CLI object to handle arguments
        """
        self.args = None
        self.manipulator = None
        self.parameter_args = None

    def parse_arguments(self, args):
        """
        Defines the allowed application arguments and invokes the evaluation of the arguments.

        :param args: The application arguments
        """
        parser = argparse.ArgumentParser(description="Generate Complexity Metrics on Network Data")

        # Required arguments
        parser.add_argument('--csv', type=str, help='Directory containing network flow statistics in CSV form')
        parser.add_argument("--results", type=str, help='Location of Results Folder')
        parser.add_argument("--metadata", type=str, help='Location of Metadata File')

        # Metadata arguments -- removing these because they're just too cumbersome. Should just rely on handwritten
        # metadata file
        #parser.add_argument("--drop", nargs='+', help='Fields to be dropped during data pre-processing')
        #parser.add_argument("--unique", nargs='+', help='Fields containing discrete values')
        #parser.add_argument("--label", type=str, help='Name of field containing target labels')
        #parser.add_argument("--benign", type=str, help='Name of benign label')
        #parser.add_argument("--dport", type=str, help='Name of destination port field')
        #parser.add_argument("--ports", nargs='+', help='List of bad ports to be ignored/considered background traffic')
        #parser.add_argument("--control", nargs='+', help='Fields that an attacker can control')
        #parser.add_argument("--dbytes", type=str, help="Name of Total Destination Bytes field")
        #parser.add_argument("--sbytes", type=str, help="Name of Total Source Bytes field")

        # Parameter Arguments
        parser.add_argument("--port", type=int, default=None, help="Target Port")

        # Other Arguments
        parser.add_argument("--list", action='store_true', help='List attack labels and exit')
        parser.add_argument("--target", type=str, default=None, help='Target label for classifiers')
        parser.add_argument("--manipulate", action="store_true", help='Apply manipulations stored in metadata file')
        parser.add_argument("--siamese", action='store_true', help='Calculate results from Siamese Network')
        parser.add_argument("--sniff", type=str, help='Apply hueristic sniff test for bad smells.')
        parser.add_argument("--metric", type=str, help='Apply metric across dataset or to target')
        parser.add_argument("--ids", type=str, help='Apply (basic) classifier to dataset')
        parser.add_argument("--verbose", action="store_true", help='Print verbose results')
        parser.add_argument("--metriclist", action="store_true", help="List metric options")
        parser.add_argument("--snifflist", action="store_true", help="List sniff test options")

        self.args = parser.parse_args()
        if (self.args.csv is None) or (self.args.results is None) or (self.args.metadata is None):
            raise ValueError("Must provide all of the following arguments: csv, results, metadata")
        if (self.args.target is None):
            warnings.warn("", RuntimeWarning)
        if self.args.results != None:
            if not os.path.exists(self.args.results):
                os.makedirs(self.args.results)
        self.manipulator = data_manip.Manipulator(dataset_path=self.args.csv, metadata_path=self.args.metadata, target_label=self.args.target, metadata_manip=self.args.manipulate)
        self.parameter_args = {"port": self.args.port, "results": self.args.results}

    def chooseMetric(self):
        metric_name = self.parseMetric(self.args.metric)
        label_field = self.manipulator.label_field
        results = {}
        for name, metric in inspect.getmembers(metrics):
            if inspect.isclass(metric):
                if metric_name == name:
                    if "Metric" in name:
                        met = metric(self.manipulator)
                        """
                        if "GiniImpurity" in name or "InfoGain" in name:
                            results[self.args.target] = met.apply_metric(self.df, self.metadata, self.args.target) 
                        else:
                            data = data_manip.reformatForDiv(self.df, self.metadata)
                            data1 = data[data[label_field] == self.args.target]
                            for col in data1.columns:
                                results[col] = met.apply_metric(data1[col])
                        """
                        results = met.apply_metric()
                    elif "Divergence" in name:
                        div = metric(self.manipulator)
                        data = data_manip.reformatForDiv(self.df, self.metadata)
                        data1 = data[data[label_field] == self.args.target]
                        data2 = data[data[label_field] == self.metadata['benign_label']]
                        for col in list(set(data1.columns) & set(data2.columns)):
                            results[col] = div.apply_metric(data1[col], data2[col])
        return results

    @staticmethod
    def processMetricListing():
        emph_start = '\033[1m'
        emph_end = '\033[0m'
        for name, metric in inspect.getmembers(metrics):
            if inspect.isclass(metric):
                if "Metric" in name or "Divergence" in name:
                    if name != "BaseMetric":
                        met = metric(manipulator=None)
                        print('[#] {}{}{}'.format(emph_start, met.metric_name, emph_end))
                        print('\t[!] {}Description:{} {}'.format(emph_start, emph_end, met.metric_description))
                        print('\t[!] {}Type:{} {}'.format(emph_start, emph_end, met.metric_type))

    @staticmethod
    def parseMetric(metric_name):
        available_metrics = []
        for name, metric in inspect.getmembers(metrics):
            if inspect.isclass(metric):
                if "Metric" in name or "Divergence" in name:
                    if name != "BaseMetric":
                        available_metrics.append(name)
        
        highest_sim = 0.0
        highest_sim_metric = None
        for metric in available_metrics:
            if metric_name == metric:
                return metric
            counter_check = metric.lower()
            similarity = difflib.SequenceMatcher(None, metric_name.lower(), counter_check).ratio()
            if similarity == 1.0:
                return metric
            if similarity > highest_sim:
                highest_sim = similarity
                highest_sim_metric = metric

        if highest_sim >= 0.6:
            print("Could not find attack with name " + metric_name + ". Closest match was " + highest_sim_metric + ".")
        else:
            print("Could not find attack with name " + metric_name + " or with similar name.")
            exit(1)

    def chooseSniffTest(self):
        sniff_name = self.parseSniffTest(self.args.sniff)
        results_dict = {}
        for name, sniff in inspect.getmembers(sniffs):
            if inspect.isclass(sniff):
                if sniff_name == name:
                    t = sniff(self.manipulator)
                    arg_dict = {}
                    for _arg in self.parameter_args:
                        if _arg in list(inspect.signature(t.pipeline).parameters.keys()):
                            arg_dict[_arg] = self.parameter_args[_arg]
                    results_dict[sniff_name] = t.pipeline(**arg_dict)
                    return results_dict


    @staticmethod
    def parseSniffTest(sniff_name):
        available_sniffs = []
        for name, sniff in inspect.getmembers(sniffs):
            if inspect.isclass(sniff):
                if "Sniff" in name:
                    if name != "BaseSniff":
                        available_sniffs.append(name)
        highest_sim = 0.0
        highest_sim_test = None
        for sniff in available_sniffs:
            if sniff_name == sniff:
                return sniff
            counter_check = sniff.lower()
            similarity = difflib.SequenceMatcher(None, sniff_name.lower(), counter_check).ratio()
            if similarity == 1.0:
                return sniff
            if similarity > highest_sim:
                highest_sim = similarity
                highest_sim_test = sniff

        if highest_sim >= 0.6:
            print("Could not find test with name " + sniff_name + ". Closest match was " + highest_sim_test + ".")
        else:
            print("Could not find test with name " + sniff_name + " or with similar name.")
            exit(1)

    @staticmethod
    def processSniffTestListing():
        emph_start = '\033[1m'
        emph_end = '\033[0m'
        for name, sniff in inspect.getmembers(sniffs):
            if inspect.isclass(sniff):
                if "Sniff" in name:
                    if name != "BaseSniff":
                        t = sniff(manipulator=None)
                        print('[#] {}{}{}'.format(emph_start, t.test_name, emph_end))
                        print('\t[!] {}Description:{} {}'.format(emph_start, emph_end, t.test_description))
                        print('\t[!] {}Type:{} {}'.format(emph_start, emph_end, t.test_type))
