import datetime
import os
import inspect
import numpy as np
from scipy import spatial
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing

import data_manip
import ml
import utils
import metrics
import pandas as pd

from abc import ABC, abstractmethod

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

class MetadataInformationMissingError(Exception):
    pass

class BaseSniff(ABC):
    """
    Base Class for our sniff tests.
    """
    def __init__(self, name: str, description: str, type: str, manipulator: data_manip.Manipulator = None, suppress_reformatting: bool = False):
        """
        Name = Name of sniff test
        Description = Description of how the sniff test works
        Type = Smell Category
        """
        self.test_name = name
        self.test_description = description
        self.test_type = type
        self.operations_log = []
        self.manipulator = manipulator
        self.suppress_reformatting = suppress_reformatting
        if self.manipulator is not None:
            self.df = self.apply_manipulator()
        else:
            self.df = None


    def apply_manipulator(self):
        if not self.manipulator.instantiated:
            warnings.warn("Manipulator Not Instatiated. Instantiating Now.")
            self.manipulator.instantiate_metadata()
        if not self.manipulator.maniped and not self.suppress_reformatting:
            warnings.warn("Dataset not manipulated. Applying basic reformatting now.")
            self.manipulator.reformatForML()
        return self.manipulator.processed_df

    def update_log(self, operation_name, operands="", notes=""):
        operation_log = {
            "operation": f"{operation_name}",
            "details": {
                "operands": operands,
                "notes": notes,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
        }

        self.operations_log.append(operation_log)

    def combine_df_labels(self):
        try:
            label_field = self.check_manipulator("label_field", necessary=True)
            labels = self.check_manipulator("labels", necessary=True)
        except MetadataInformationMissingError as e:
            self.update_log("combine_df_labels", operands="", notes="Failed")
            print(e)
            raise
        self.update_log("combine_df_labels", operands="", notes="Success")
        self.df[label_field] = labels
    

    def check_manipulator(self, attr, necessary=False):
        if getattr(self.manipulator, attr, None) is None:
            if necessary:
                raise MetadataInformationMissingError(
                    f"Required '{attr}' is not defined in the Manipulator (potentially missing from metadata)."
                )
            else:
                warnings.warn(f"Optional {attr} is not defined in the Manipulator (potentially missing from metadata)", RuntimeWarning)
                self.update_log("check_metadata", f"{attr}", "Attribute not found")
                return None
        else:
            _attr = getattr(self.manipulator, attr, None)
            self.update_log("check_metadata", f"{attr}", "Attribute found")
            return _attr


    @abstractmethod
    def pipeline(self):
        """
        We implement the pipeline method for all tests,
        running the test and returning output in Dict
        format.
        """
        raise NotImplementedError("Must implement pipeline method")

class CosineSniff(BaseSniff):
    def __init__(self, manipulator: data_manip.Manipulator):
        super(CosineSniff, self).__init__("Cosine Similarity Test",
                                         "Approximates cluster size of input flows, returning a result between 0 and 1.\
We also track the number of flows that are effectively identical (according to \
some cut-off value), returning a result between 0 and 1", 
                                          "Repetitiveness",
                                          manipulator = manipulator)

    def cosine_similarity(self, data, num_points=1000):
        """
        Calculate the cosine similarity between `num_points' randomly sampled pairs.
        Assumes data has been split up into clusters, indicated by 'Cluster' column.
        """
        separated_results_list = []
        results_dict = {}
        clusters = data['Cluster']
        cluster_list, cluster_value_list = utils.get_ordered_unique_values(clusters)

        for cluster_value in range(0, len(cluster_list)):
            result_list = []
            cluster_data = data[data['Cluster'] == cluster_value]
            for _ in range(0, num_points):
                samples = cluster_data.sample(n=2) # Randomly sample 2 points
                samples = samples.drop(['Cluster'], axis=1) # Drop Cluster column before calculating cosine similarities
                result = 1 - \
                    spatial.distance.cosine(samples.iloc[0], samples.iloc[1]) # Calculate cosine distance, and subtract from 1 to calculate cosine similarities
                    # Could add a check here to ensure there are no collisions between sample1 and sample2
                result_list.append(result)
            separated_results_list.append(result_list) # Keep the results of each cluster separated
            results_dict[cluster_value] = separated_results_list[(cluster_value-1)]
        return separated_results_list, cluster_value_list

    def cosSimResults(self, cos_sim, num_points=1000, cutoff=0.95):
        """
        Organise Cosine Similarity results by calculating mean scores
        as well as the number of flows that are above the cutoff score
        (and we consider to be effectively identical).
        """
        identikit = {}
        for idx, cs in enumerate(cos_sim):
            identikit[idx+1] = [sim for sim in cs if sim > cutoff]
        for key in identikit.keys():
            identikit[key] = len(identikit[key]) / num_points
        return np.mean(cos_sim, axis=1).tolist(), identikit

    def cosSimPipeline(self, df, num_points):
        """
        Run Cosign Similarity pipeline.
        1) Calculate number of clusters via elbow method
        2) Calculate Clusters
        3) Calculate Cosine Similarities for each Cluster, weighted
        4) Calculate Results
        """
        n_clusters = utils.calculateElbowValue(df)
        print(f"[*] {n_clusters} Clusters")
        if n_clusters == None:
            n_clusters = 1
        df, _ = ml.calculateClusters(df, n_clusters)
        cos_sim, weights = self.cosine_similarity(df, num_points)
        averages, cutoff = self.cosSimResults(cos_sim, num_points)
        return averages, cutoff, weights

    def pipeline(self, num_points: int = 1000):
        """
        Run Full Pipeline.
        """
        results_dict = {}
        cos = None
        cut = None
        try:
            self.manipulator.reformatForCosine()
            cos, cut, weights = self.cosSimPipeline(self.df, num_points)
            results_dict["Cosine Similarities"] = cos # Cosine Similarities
            results_dict["Cosine Weighted Average"] = np.average(cos, weights=[weight/sum(weights) for weight in weights]) # Get Weighted Average for each cluster
            results_dict["Largest Cluster Percentage"] = weights[0]/sum(weights) # Get the largest cluster size
            results_dict["Cutoff Percentages"] = cut # Percentage of Cutoff values
            results_dict["Cluster Sizes"] = weights # Size of each cluster
        except Exception as e:
            print(e)
            raise
        return results_dict


class PortSniff(BaseSniff):
    def __init__(self, manipulator: data_manip.Manipulator):
        super(PortSniff, self).__init__("Port Test", "Returns the percentage of flows associated with\
background traffic according to their destination port number.",
                                                    "Mislabelled",
                                                    manipulator=manipulator)

    def pipeline(self):
        '''
        Get the number of flows in malicious class whose destination port is connected to a port associated with background traffic. 

        Inputs:
        - manipulator (Manipulator): The manipulator instance that holds the manipulated data and metadata.
        - target (str): The target value for testing

        Returns: A dictionary with test results
        '''
        results_dict = {}
        try:
            target = self.check_manipulator("target_label", necessary=True) # self.manipulator.target_label
            dport = self.check_manipulator("dst_port", necessary=True) # self.manipulator.dst_port
            background_ports = self.check_manipulator("background_ports", necessary=True) # self.manipulator.background_ports
            label_field = self.check_manipulator("label_field", necessary=True) # self.manipulator.label_field
            self.combine_df_labels()
        except MetadataInformationMissingError as e:
            print(e)
        port_count = 0
        for port in background_ports:
            condition = f'(`{label_field}` == {target}) & (`{dport}` == {port})'
            port_count += self.df.query(condition).shape[0]
        null_count = self.df[(self.df[label_field] == target)][dport].isnull().sum()
        empty_count = self.df[(self.df[label_field] == target) & (self.df[dport] == "")].shape[0]

        condition = f'`{label_field}` == {target}'

        results_dict["Background Ports In Mal Class"] = int(port_count)
        results_dict["Null Ports in Mal Class"] = int(null_count)
        results_dict["Empty Ports in Mal Class"] = int(empty_count)
        results_dict["Background Ports In Mal Class Percentage"] = port_count / self.df.query(condition).shape[0]
        results_dict["Background, Null & Empty Ports in Mal Class"] = int(port_count) + int(null_count) + int(empty_count)
        results_dict["Background, Null & Empty Ports in Mal Class Percentage"] = (int(port_count) + int(null_count) + int(empty_count)) / self.df.query(condition).shape[0]

        return results_dict

class BackwardPacketsSniff(BaseSniff):
    """
    Backwards Packets Test
    The lack of communication from Server -> Host implies
    that the target system isn't properly responding to input
    """
    def __init__(self, manipulator: data_manip.Manipulator):
        super(BackwardPacketsSniff, self).__init__(
            "Backward Packets", "Returns the percentage of flows with no packets \
in the backwards (server to host) direction.",
            "Mislabelled",
            manipulator=manipulator)

    def pipeline(self):
        """
        Run Pipeline
        """
        results_dict = {}
        try:
            label_field = self.check_manipulator("label_field", necessary=True) # self.manipulator.label_field
            target = self.check_manipulator("target_label", necessary=True) # self.manipulator.target_label
            bwd = self.check_manipulator("backward_packets_field", necessary=True) # self.manipulator.backwards_packets_field
            self.combine_df_labels()
        except MetadataInformationMissingError as e:
            print(e)
            raise
        length = self.df[self.df[label_field] == target].shape[0]
        bwdLength = self.df[(self.df[label_field] == target) & (self.df[bwd] == 0)].shape[0]

        results_dict["Percentage of Flows with Zero Backwards Packets"] = (bwdLength / length)
        results_dict["Total Number of Flows with Zero Backwards Packets"] = length
        return results_dict


class NearestNeighboursSniff(BaseSniff):
    def __init__(self, manipulator: data_manip.Manipulator):
        super(NearestNeighboursSniff, self).__init__(
            "Nearest Neighbours Test", "Returns the percentage of flows that are\
mislabelled according to the Edited Nearest\
Neighbours Criteria.",
                                       "Mislabelled",
                                       manipulator=manipulator,
                                       suppress_reformatting=True)

    def mislabelNN(self, num_points: int = 1000000):
        neigh = NearestNeighbors(n_neighbors=4, radius=0.5) # Calculate Nearest Neighbours
        self.manipulator.getLabelledRows([0, 1])
        combined_df = self.manipulator.processed_df.reset_index(drop=True)
        attack_df = self.manipulator.getLabelledRows(1)
        if combined_df.shape[0] < num_points:
            num_points = combined_df.shape[0] # If they're aren't enough points, reset num_points
        combined_df = combined_df.sample(num_points) # Sample num_points points from combined DF
        neigh.fit(combined_df)
        knn = neigh.kneighbors(attack_df)
        return knn, combined_df, attack_df

    def pipeline(self, num_points=1000000):
        try:
            benign_label = self.check_manipulator("benign_label", necessary=True) # self.manipulator.benign_label
            label_field = self.check_manipulator("label_field", necessary=True) # self.manipulator.label_field
            target = self.check_manipulator("target_label", necessary=True)  # self.manipulator.target_label
        except MetadataInformationMissingError as e:
            print(e)
            raise
        results_dict = {}
        if target == benign_label:
            raise ValueError("Target label cannot equal benign label")
        self.manipulator.reformatForClustering()
        knn, benign_df, _ = self.mislabelNN(num_points)
        mislabel = 0
        for idx in knn[1]:
            try:
                counts = benign_df.iloc[idx][label_field].value_counts()[0]
            except Exception as e:
                counts = 0
            if counts >= 2:
                mislabel += 1
        results_dict["Mislabelled"] = mislabel
        results_dict["Percentage"] = (mislabel / knn[1].shape[0])
        return results_dict

class RollingImportancesSniff(BaseSniff):
    def __init__(self, manipulator: data_manip.Manipulator):
        super(RollingImportancesSniff, self).__init__("Rolling Importances Test", 
                                                     "Orders features according to mutual information\
and then measures efficacy of 5 features together\
for classification using random forest. Outputs JSON file \
with F1 score for each group of 5 features. (Unused, not particularly useful)",
                                                      "Simulation Artefacts",
                                                      manipulator=manipulator)

    def pipeline(self, n=5):
        cols = []
        results = []
        f1_scores = {}
        results_dict = {}

        info = metrics.InfoGainMetric(self.manipulator)
        importances = info.apply_metric()
        importDf = pd.DataFrame(importances, index=[0])
        for i in range(0, len(importDf.columns) - n):
            n_cols = importDf.columns[i:i+n]
            train_features, train_labels, test_features, test_labels, _, _ = self.manipulator.getTrainTestFeatures(n_cols)
            predictions, test_score, _ = ml.ids("Forest", train_features, train_labels, test_features, test_labels)
            f1 = ml.results(test_score, predictions, test_labels)
            results.append(test_score)
            cols.append(n_cols.tolist())
            f1_scores[n_cols[-1]] = f1
        results_dict[f"Rolling Importances: {n} columns"] = cols
        results_dict[f"Rolling Importances: {n} columns F1 Scores"] = f1_scores
        return results_dict

class DropRollingImportancesSniff(BaseSniff):
    def __init__(self, manipulator: data_manip.Manipulator):
        super(DropRollingImportancesSniff, self).__init__("Rolling Importances With Drop Test", 
                                                         "Rolling Importances with Drop. (Unused,\
not particularly useful)", 
                                                         "Simulation Artefacts",
                                                         manipulator=manipulator)

    def pipeline(self):
        cols = []
        current_features = []
        f1_scores = {}
        results_dict = {}

        info = metrics.InfoGainMetric(manipulator=self.manipulator)
        importances = info.apply_metric()
        importDf = pd.DataFrame(importances, index=[0])
        self.manipulator.downsampleRows(frac=0.2)

        threshold = 0.2
        old_f1 = 0.5
        current_features.append(importDf.columns[0])

        for i in range(1, len(importDf.columns) - 1):
            current_features.append(importDf.columns[i])
            train_features, train_labels, test_features, test_labels, _, _ = self.manipulator.getTrainTestFeatures(current_features)
            predictions, test_score, _ = ml.ids("Forest", train_features, train_labels, test_features, test_labels)
            new_f1 = ml.results(test_score, predictions, test_labels)
            cols.append(current_features)
            if new_f1 - old_f1 > threshold:
                current_features.remove(importDf.columns[i])
                score = new_f1 - old_f1 + 0.5
            else:     
                score = new_f1 - old_f1 + 0.5
                old_f1 = new_f1
            f1_scores[importDf.columns[i]] = score
        results_dict["Rolling Importances w/ Drop F1 scores"] = f1_scores
        return results_dict

class SingleFeatureEfficacySniff(BaseSniff):
    def __init__(self, manipulator: data_manip.Manipulator):
        description = ("Single Feature Efficacy Test - Orders features according to mutual information "
                       "and trains a random forest for classification on a single feature. "
                       "Outputs JSON file with F1 score for each feature")
        super(SingleFeatureEfficacySniff, self).__init__("Single Feature Efficacy Test", description, "Simulation Artefacts", manipulator)

    def pipeline(self):
        """
        Run pipeline.
        """
        f1_scores = {}
        results_dict = {}
        info = metrics.InfoGainMetric(manipulator=self.manipulator) # InfoGain Metric
        importances = info.apply_metric() # Get InfoGain for each column
        importDf = pd.DataFrame(importances, index=[0])
        
        for i in range(0, len(importDf.columns)): # Run Random Forest Classifier, trained on each column
            train_features, train_labels, test_features, test_labels, _, _ = self.manipulator.getTrainTestFeatures([importDf.columns[i]])
            predictions, test_score, _ = ml.ids("Forest", train_features, train_labels, test_features, test_labels) 
            f1 = ml.results(test_score, predictions, test_labels)
            f1_scores[importDf.columns[i]] = f1
        results_dict["Single Feature Efficacy F1 Scores"] = f1_scores
        return results_dict

    
class CorrelationSniff(BaseSniff):
    def __init__(self, manipulator: data_manip.Manipulator):
        super(CorrelationSniff, self).__init__("Correlation Test",
                                              "Calculate correlations between\
features (Unused, not useful)",
                                              "Other",
                                              manipulator=manipulator)

    def pipeline(self, rearrange=False, importances=None):
        """
        Run pipeline.
        """
        if rearrange:
            if importances is not None:
                df = utils.rearrange_importance(df=self.manipulator.processed_df, importances=importances)
            else:
                df = utils.rearrange_importance(df=self.manipulator.processed_df, metadata=self.manipulator.metadata, target=self.manipulator.target_label)
        else:
            df = self.manipulator.processed_df
        correlations = df.corr()
        return correlations


class SimpleAdversarialSniff(BaseSniff):
    def __init__(self, manipulator: data_manip.Manipulator):
        super(SimpleAdversarialSniff, self).__init__("Unguided Adversarial Test",
                                                    "Adv (Unused, although interesting)",
                                                    "Other",
                                                    manipulator=manipulator)

    def _importances(self, classifier):
        num_feat = min(10, classifier.n_features_in_)
        importances = classifier.feature_importances_
        std = np.std([tree.feature_importances_ for tree in classifier.estimators_],
            axis=0)
        indices = np.argsort(importances)[::-1][:num_feat]
        return importances, std, indices
    
    def _first_run(self, train_features, train_labels, test_features, test_labels):
        scaler = preprocessing.StandardScaler().fit(train_features)

        train_features = scaler.transform(train_features)
        test_features = scaler.transform(test_features)
        predictions, test_score, classifier = ml.ids("Logistic",
                                                    train_features=train_features,
                                                    train_labels=train_labels,
                                                    test_features=test_features,
                                                    test_labels=test_labels)
        ml.results(test_score, predictions, test_labels)
        index_dict = {}
        return classifier, index_dict

    def _run(self, classifier, test_features):
        predictions = classifier.predict(test_features)
        return predictions

    def _adv(self, indices, test_features, test_labels, scale):
        labels = np.where(test_labels == 1)[0]
        for index in indices:
           test_features[labels, index] = test_features[labels, index] * scale,
        test_features = test_features.to_numpy()
        return test_features
    
    def pipeline(self):
        train_features, train_labels, test_features, test_labels, _, _ = self.manipulator.getTrainSniffFeatures()
        classifier, indices = self._first_run(self.manipulator.processed_df, train_features, train_labels, test_features, test_labels)
        control_important = []
        for field in self.manipulator.control_fields:
            if field in indices.keys():
                control_important.append(indices[field])
        for scale in  [1.5, 2, 2.5, 3, 3.5]:
            modified_test_features = self._adv(control_important, test_features, test_labels, scale)
            scaler = preprocessing.StandardScaler().fit(train_features)
            modified_test_features = scaler.transform(modified_test_features)
            predictions = self._run(classifier, modified_test_features, test_labels)
            test_score = np.mean(test_labels == predictions)
            ml.results(test_score, predictions, test_labels)

class TimeBehaviourSniff(BaseSniff):
    def __init__(self, manipulator: data_manip.Manipulator):
        super(TimeBehaviourSniff, self).__init__("Time vs Behaviour Plots",
                                                "Plot time vs behaviour time series data",
                                                "Other",
                                                manipulator=manipulator)
    
    def plot_series(self, df, output_directory, filename):
        """
        We assume that the CSV is already sorted according to time, so we just need to plot it.
        """
        def _plot_series(df, file_path):
            import seaborn as sns
            import matplotlib.pyplot as plt
            fig = sns.lineplot(data=df, x="idx", y="value", hue="variable").get_figure()
            fig.savefig(file_path)
            plt.clf()
            # Check if the directory exists
        if not os.path.exists(output_directory):
            # If not, create it
            os.makedirs(output_directory)
        # Create the full file path
        file_path = os.path.join(output_directory, filename)
        return _plot_series(df, file_path)


    def pipeline(self, port=None, remove_outliers=False, results=None):

        results_dict = {}

        try:
            label_field = self.check_manipulator("label_field", necessary=True) #self.manipulator.label_field
            port_field = self.check_manipulator("dst_port", necessary=True) #self.manipulator.dst_port
            destination_bytes =  self.check_manipulator("destination_bytes_field", necessary=True) #self.manipulator.destination_bytes_field
            source_bytes = self.check_manipulator("source_bytes_field", necessary=True) #self.manipulator.source_bytes_field
            target = self.check_manipulator("target_label") #self.manipulator.target_label
        except MetadataInformationMissingError as e:
            print(e)
            raise

        if target is None and port is not None:
            trunc_df = self.df[self.df[port_field] == port][[destination_bytes, source_bytes]]
        elif target is None and port is not None:
            trunc_df = self.df[self.df[label_field] == target][[destination_bytes, source_bytes]]
        elif target is not None and port is not None:
            trunc_df = self.df[(self.df[label_field] == target) & (self.df[port_field] == port)][[destination_bytes, source_bytes]]
        else:
            trunc_df = self.df[[destination_bytes, source_bytes]]
        if remove_outliers:
            trunc_df = trunc_df[(np.abs(stats.zscore(trunc_df[source_bytes])) < 3)]
            trunc_df = trunc_df[(np.abs(stats.zscore(trunc_df[destination_bytes])) < 3)]

        ordered_source_df = trunc_df.sort_values(source_bytes)
        ordered_destination_df = trunc_df.sort_values(destination_bytes)

        trunc_df["idx"] = [i for i in range(0, trunc_df.shape[0])]
        ordered_source_df["idx"] = [i for i in range(0, ordered_source_df.shape[0])]
        ordered_destination_df["idx"] = [i for i in range(0, ordered_destination_df.shape[0])]

        trunc_df = pd.melt(trunc_df, id_vars=["idx"], value_vars=[destination_bytes, source_bytes])
        ordered_source_df = pd.melt(ordered_source_df, id_vars=["idx"], value_vars=[destination_bytes, source_bytes])
        ordered_destination_df = pd.melt(ordered_destination_df, id_vars=["idx"], value_vars=[destination_bytes, source_bytes])

        results_dict["Spearman Correlation between Source and Destination Bytes"] = self.df[[destination_bytes, source_bytes]].corr(method="spearman")
        results_dict["Pearson Correlation between Source and Destination Bytes"] = self.df[[destination_bytes, source_bytes]].corr(method="pearson")

        if results is not None:
            self.plot_series(trunc_df, results, "TimeSeries.pdf")
            self.plot_series(ordered_source_df, results, "SourceSeries.pdf")
            self.plot_series(ordered_destination_df, results, "DestinationSeries.pdf")
        return results_dict

class FlowRateSniff(BaseSniff):
    def __init__(self, manipulator: data_manip.Manipulator):
        super(FlowRateSniff, self).__init__("Flow Rate Test",
                                           "Calculate rate of flows over time",
                                           "Other",
                                           manipulator=manipulator)

    def pipeline(self, port=None, rate=60, results=None):
        results_dict = {}
        try:
            dst_port = self.check_manipulator("dst_port", necessary=True) #self.manipulator.dst_port
            destination_bytes = self.check_manipulator("destination_bytes_field", necessary=True) #self.manipulator.destination_bytes_field
            source_bytes = self.check_manipulator("source_bytes_field", necessary=True) #self.manipulator.source_bytes_field
            time_field = self.check_manipulator("timestamp_field", necessary=True) #self.manipulator.timestamp_field
            time_format = self.check_manipulator("timestamp_format", necessary=True) #self.manipulator.timestamp_format
        except MetadataInformationMissingError as e:
            print(e)
            raise

        if port != None:
            self.df = self.df[self.df[dst_port] == port][[time_field, source_bytes, destination_bytes]]
        self.df[time_field] = pd.to_datetime(self.df[time_field], format=time_format)
        self.df = self.df.sort_values(by=time_field, ascending=True)
        self.df = self.df.reset_index()

        low_time = self.df[time_field].iloc[0]
        high_time = self.df[time_field].iloc[self.df.shape[0]-1]

        count_list = []
        src_bytes_mean_list = []
        dst_bytes_mean_list = []
        src_bytes_std_list = []
        dst_bytes_std_list = []

        while low_time < high_time:
            new_time = low_time + datetime.timedelta(seconds=rate)
            time_mask = ((self.df[time_field] >= low_time) & (self.df[time_field] <= new_time))
            between_flows = self.df.loc[time_mask] 
            between_count = between_flows.shape[0]

            count_list.append(between_count)
            src_bytes_mean_list.append(between_flows[source_bytes].mean())
            dst_bytes_mean_list.append(between_flows[destination_bytes].mean())
            src_bytes_std_list.append(between_flows[source_bytes].std())
            dst_bytes_std_list.append(between_flows[destination_bytes].std())

            low_time = new_time

        results_dict["Summary Mean Flow Count"] = sum(count_list) / len(count_list)
        results_dict["Summary Mean Src Bytes"] = sum(src_bytes_mean_list) / len(src_bytes_mean_list)
        results_dict["Summary Mean Dst Bytes"] = sum(dst_bytes_mean_list) / len(dst_bytes_mean_list)
        results_dict["Summary Std Src Bytes"] = sum(src_bytes_std_list) / len(src_bytes_std_list)
        results_dict["Summary Std Dst Bytes"] = sum(dst_bytes_std_list) / len(dst_bytes_std_list)
        if results is None:
            results_dict["Provide results directory for full results"] = ""
        else:
            results_df = pd.DataFrame({"Flow Count": count_list, "Source Bytes Mean": src_bytes_mean_list,
                                       "Destination Bytes Mean": dst_bytes_mean_list,
                                       "Source Bytes Std": src_bytes_std_list,
                                       "Destination Bytes Std": dst_bytes_std_list})
            utils.save_dataframe(results_df, results, "flowRateSniffTest_results.csv")
        return results_dict

class RollingRatioSniff(BaseSniff):
    def __init__(self, manipulator: data_manip.Manipulator):
        super(RollingRatioSniff, self).__init__("Rolling Ratio Test",
                                    "Calculate unique items in a column in a rolling manner",
                                    "Low Data Diversity",
                                    manipulator=manipulator)

    def pipeline(self, column="Total Fwd Pkts", rate=60, results=None):
        results_dict = {}
        gini_list = []

        try:
            time_field = self.check_manipulator("timestamp_field", necessary=True) #self.manipulator.timestamp_field
            time_format = self.check_manipulator("timestamp_format", necessary=True) #self.manipulator.timestamp_format
        except MetadataInformationMissingError as e:
            print(e)
            raise
        self.df[time_field] = pd.to_datetime(self.df[time_field], format=time_format)
        self.df = self.df.sort_values(by=time_field, ascending=True)
        self.df = self.df.reset_index()

        low_time = self.df[time_field].iloc[0]
        high_time = self.df[time_field].iloc[self.df.shape[0]-1]

        while low_time < high_time:
            new_time = low_time + datetime.timedelta(seconds=rate)
            time_mask = ((self.df[time_field] >= low_time) & (self.df[time_field] <= new_time))
            between_flows = self.df.loc[time_mask] 
            gini = metrics.GiniCoefficientMetric(data_manip.Manipulator(original_df=between_flows))
            gini_list.append(gini.apply_metric(column))
            low_time = new_time
            if results:
                utils.save_dataframe(between_flows[column].value_counts(), 
                                     results, 
                                     f"rollingRatioSniff_results_{low_time.strftime(time_format)}.csv",
                                     index=True)
        results_dict[f"Rolling Gini Impurity of Column {column} with rate {rate}s"] = gini_list
        if results is None:
            results_dict["Provide results directory for full results"] = ""
        return results_dict

class RollingMetricSniff(BaseSniff):
    def __init__(self, manipulator: data_manip.Manipulator):
        super(RollingMetricSniff, self).__init__("Rolling Metric Test",
                                            "Calculate metric results for a column in a rolling manner",
                                            "Other",
                                            manipulator=manipulator)

    def pipeline(self, metric=metrics.EntropyMetric, column="Dst Port", rate=60, results=None):
        results_dict = {}
        metric_list = []
        try:
            time_field = self.check_manipulator("timestamp_field", necessary=True) #self.manipulator.timestamp_field
            time_format = self.check_manipulator("timestamp_format", necessary=True) #self.manipulator.timestamp_format
        except MetadataInformationMissingError as e:
            print(e)
            raise
        self.df[time_field] = pd.to_datetime(self.df[time_field], format=time_format)
        self.df = self.df.sort_values(by=time_field, ascending=True)

        low_time = self.df[time_field].iloc[0]
        high_time = self.df[time_field].iloc[self.df.shape[0]-1]
        while low_time < high_time:
            new_time = low_time + datetime.timedelta(seconds=rate)
            time_mask = ((self.df[time_field] >= low_time) & (self.df[time_field] <= new_time))
            between_flows = self.df.loc[time_mask] 
            m = metric(data_manip.Manipulator(original_df=between_flows))
            arg_dict = {}
            for _arg in {**locals()}.keys():
                if _arg in list(inspect.signature(m.apply_metric).parameters.keys()):
                    arg_dict[_arg] = {**locals()}[_arg]
            metric_list.append(m.apply_metric(**arg_dict))
            low_time = new_time
        if results:
            utils.save_dataframe(pd.DataFrame({f"{m.metric_name} rolling results": metric_list}), results, f"{m.metric_name.replace(' ', '')}RollingResults.csv")
        results_dict[f"{m.metric_name} rolling results mean"] = np.mean(metric_list)
        results_dict[f"{m.metric_name} rolling results std dev"] = np.std(metric_list)
        if results is None:
            results_dict["Provide results directory for full results"] = ""
        



class CompleteSniff(BaseSniff):
    def __init__(self, manipulator: data_manip.Manipulator):
        super(CompleteSniff, self).__init__("Complete", "Run all tests", "Other")
    
    def pipeline(self, df, metadata, target):
        cosine = CosineSniff()
        ports = PortSniff()
        nn = NearestNeighboursSniff()
        sing = SingleFeatureEfficacySniff()

        results_dict = {}

        try:
            print("[!] Running Cosine Sniff")
            results_dict["CosineTest"] = cosine.pipeline(df, metadata, target, 500)
        except:
            results_dict["CosineTest"] = "Error!"
        try:
            print("[!] Running Mislabelled Ports Test")
            results_dict["MislabelPortsTest"] = ports.pipeline(df, metadata, target)
        except:
            results_dict["MislabelPortsTest"] = "Error!"
        try:
            print("[!] Running Mislabelled Test")
            results_dict["MislabelTest"] = nn.pipeline(df, metadata, target, 500)
        except:
            results_dict["MislabelTest"] = "Error!"
        try:
            print("[!] Running Importances Test")
            results_dict["SingleFeatureEfficacy"] = sing.pipeline(df, metadata, target)
        except:
            results_dict["SingleFeatureEfficacy"] = "Error!"

        return results_dict

#m = data_manip.Manipulator("/home/rob/Documents/PhD/WhiffSuite/tests/csvs", metadata_path="/home/rob/Documents/PhD/WhiffSuite/tests/metadata/our_metadata.json", target_label="Attack", metadata_manip=False)

#t = CosineSniff(m)
#t.pipeline()
#print(m.operations_log)