import pandas as pd
import numpy as np
import os, glob
import json
import warnings
import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import OrderedDict

supported_keys = ["label_field",
                  "dst_port",
                  "drop_fields",
                  "unique_fields", 
                  "string_fields",
                  "numeric_fields", 
                  "benign_label", 
                  "comment", 
                  "background_ports",
                  "backward_packets_field",
                  "source_bytes_field"
                  "destination_bytes_field",
                  "timestamp_field",
                  "timestamp_format",
                  "operations"]

critical_keys = ["label_field"]

class Manipulator:
    def __init__(self, dataset_path=None, metadata_path=None, target_label=None,  original_df=None, metadata_manip=False):
        self.processed_df = None
        self.labels = None
        self.target_label = target_label
        self.operations_log = []
        self.instantiated = False
        self.maniped = False
        print("[*] Loading Metadata")
        self.metadata = self.load_metadata(metadata_path) if metadata_path else None
        print("[*] Metadata Loaded")
        if self.metadata is not None:
            print("[*] Instantiating Metadata")
            self.instantiate_metadata()
            print("[*] Metadata Instantiated")        
        print("[*] Loading Files")
        self.original_df = self.readDirec(dataset_path) if dataset_path else original_df
        print("[*] Files Loaded")
        self.processed_df = self.original_df
        if self.metadata is not None:
            print("[*] Beginning Initial Manipulation")
            self.initial_manip()
            print("[*] Initial Manipulation Finished")
            if metadata_manip:
                self.apply_metadata()
        print(self.operations_log)


    def readFile(self, _file: str):
        df = pd.read_csv(_file)
        #### Just for ICSX ####
        #df = df.drop(df[((df['totalSourceBytes'] % 64) == 0) & ((df['totalDestinationBytes'] % 64) == 0)].index)
        #df = df.drop(df[(df["Tag"] != "Attack") & (df["totalDestinationBytes"] == 163972)].index)
        return df

    def readDirec(self, _path: str):
        dfs = []    
        for filename in glob.glob(os.path.join(_path, '*.csv')):
            print(f"[*] Loading file {filename}")
            dfs.append(self.readFile(filename))
        print(f"[*] Concatenating {len(dfs)} files")
        return pd.concat(dfs, ignore_index=True)


    def load_metadata(self, _file: str):
        '''
        Load our metadata
        '''
        with open(_file, 'r') as md:
            return json.load(md, object_pairs_hook=OrderedDict)
    
    def update_log(self, operation_name, operands=""):
        operation_log = {
            "operation": f"{operation_name}",
            "details": {
                "operands": operands,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z"),
                "shape": self.processed_df.shape
            }
        }
        self.operations_log.append(operation_log)

    def instantiate_metadata(self):
        def check_metadata(key: str):
            to_return = self.metadata[key] if key in self.metadata else None
            if to_return == None:
                if key in critical_keys:
                    warnings.warn(f"Warning: Critical metadata \"{key}\" not in metadata file. This will almost certainly cause a runtime error later.", RuntimeWarning)
                else:
                    warnings.warn(f"Warning: {key} not in metadata file", RuntimeWarning)
                return None
            return to_return
        self.label_field = check_metadata("label_field")
        self.src_ip = check_metadata("source_ip_field")
        self.dst_ip = check_metadata("destination_ip_field")
        self.dst_port = check_metadata("dst_port")
        self.timestamp_field = check_metadata("timestamp_field")
        self.timestamp_format = check_metadata("timestamp_format")
        self.string_fields = check_metadata("string_fields")
        self.numeric_fields = check_metadata("numeric_fields")
        self.drop_cols = check_metadata("drop_fields")
        self.unique_cols = check_metadata("unique_fields")
        self.benign_label = check_metadata("benign_label")
        self.background_ports = check_metadata("background_ports")
        self.backward_packets_field = check_metadata("backward_packets_field")
        self.source_bytes_field = check_metadata("source_bytes_field")
        self.destination_bytes_field = check_metadata("destination_bytes_field")
        self.control_fields = check_metadata("control_fields")
        self.operations = check_metadata("operations")
        self.instantiated = True

    def initial_manip(self):
        self.numeric_columns([self.dst_port])
        self.string_columns([self.label_field])
        self.drop_duplicates()

    def drop_columns(self, cols_to_drop):
        '''
        Drop specified columns
        '''
        self.processed_df = self.processed_df.drop(cols_to_drop, axis=1)
        self.update_log("drop_columns", cols_to_drop)
    
    def drop_duplicates(self):
        '''
        Drop duplicates
        '''
        self.processed_df = self.processed_df.drop_duplicates()
        self.update_log("drop_duplicates")

    def numeric_columns(self, numeric_cols):
        '''
        Often, the columns on NIDS datasets are not numeric when they should be.
        This method coerces the specified columns to be numeric.
        '''
        for col in numeric_cols:
            self.processed_df[col] = pd.to_numeric(self.processed_df[col], errors="coerce")
            self.processed_df = self.processed_df.dropna(subset=[col])
        self.update_log("numeric_columns", numeric_cols)

    def string_columns(self, string_cols):
        '''
        Often, the columns on NIDS datasets are not strings when they should be,
        or contain extraneous spaces.
        This method coerces the specified columns to be better-formatted strings.
        '''
        for col in string_cols:
            self.processed_df.loc[0:,col] = self.processed_df[col].astype(str)
            self.processed_df.loc[0:, col] = self.processed_df[col].str.strip()
        self.update_log("string_columns", string_cols)

    def categorical_columns(self, cat_cols):
        '''
        Convert columns to categorical values.
        '''
        for col in cat_cols:
            uniques = self.processed_df[col].unique()
            col_mapping = dict(zip(uniques, range(len(uniques))))
            self.processed_df = self.processed_df.replace({col: col_mapping})
        self.update_log("categorical_columns", cat_cols)

    def getLabelledRows(self, label):
        if self.labels is not None:
            self.processed_df[self.label_field] = self.labels
            if type(label) == str:
                self.processed_df = self.processed_df[self.processed_df[self.label_field] == label]
            elif type(label) == list:
                self.processed_df = self.processed_df[self.processed_df[self.label_field].isin(label)]
            self.labels = self.processed_df[self.label_field]
            self.processed_df.drop([self.label_field], axis=1)
        else:
            if type(label) == str:
                self.processed_df = self.processed_df[self.processed_df[self.label_field] == label]
            elif type(label) == list:
                self.processed_df = self.processed_df[self.processed_df[self.label_field].isin(label)]
        self.update_log("getLabelledRows", label)
    
    def dropInfsAndNaNs(self):
        self.processed_df = self.processed_df.replace([np.inf, -np.inf], np.nan)
        self.processed_df = self.processed_df.dropna()
        self.update_log("dropInfsAndNaNs")

    def dropRowsBasedOnCondition(self, conditions_list, and_cond=True):
        '''
        Drop rows based on certain conditions
        '''
        query_string = ''

        operators = {'gt': '>', 'lt': '<', 'eq': '==', 'neq': '!=', 'lte': '<=', 'gte': '>='}

        for condition in conditions_list:
            for column, operation_dict in condition.items():
                operation, value = list(operation_dict.items())[0]
                if query_string != '':
                    if and_cond:
                        query_string += ' & '
                    else:
                        # If and_cond is not True, then it is an 'or' condition
                        query_string += ' | '
                query_string += f'(`{column}` {operators[operation]} {value})'
        self.processed_df = self.processed_df.query(query_string)
        self.update_log("dropRowsBasedOnCondition", query_string)

    def splitAndDropLabels(self):
        self.labels = self.processed_df[self.label_field]
        self.processed_df = self.processed_df.drop(self.label_field, axis=1)
        self.update_log("splitAndDropLabels")

    def convertStrLabelsToInt(self):
        if self.benign_label == None:
            self.labels.loc[(self.labels != self.target_label)] = 'Other'
            label_mapping = {"Other": 0,
                        self.target_label: 1}
        else:
            label_index = self.labels.loc[((self.labels != self.benign_label) & (self.labels != self.target_label))].index
            self.labels = self.labels.drop(label_index)
            self.processed_df = self.processed_df[~self.processed_df.index.isin(label_index)]
            label_mapping = {self.benign_label: 0,
                             self.target_label: 1}
        self.labels = self.labels.replace(label_mapping)
        # As the labels have been changed, reset the benign and target labels
        self.benign_label = label_mapping[self.benign_label]
        self.target_label = label_mapping[self.target_label]
        self.update_log("convertStrLabelsToInt")

    def dropFixedValueColumns(self):
        col_log = []
        for col in self.processed_df.columns:
            if self.processed_df[col].isnull().all():
                self.processed_df = self.processed_df.drop([col], axis=1)
                col_log.append(col)
        self.update_log("dropFixedValueColumns", col_log)

    def downsampleRows(self, frac: int = None, n: int = None):
        if not (bool(frac is None) ^ bool(n is None)):
            raise RuntimeError("downsampleRows: Please supply only one of frac or n")
        elif (n is None):
            if self.labels is not None:
                self.processed_df["temp_labels"] = self.labels
                self.processed_df = self.processed_df.sample(frac=frac)
                self.labels = self.processed_df["temp_labels"]
                self.processed_df = self.processed_df.drop(["temp_labels"], axis=1)
                self.update_log("downsampleRows", f"frac: {frac}")
            else:
                self.processed_df = self.processed_df.sample(frac=frac)
                self.update_log("downsampleRows", f"frac: {frac}")
        elif (frac is None):
            if self.labels is not None:
                self.processed_df["temp_labels"] = self.labels
                self.processed_df = self.processed_df.sample(n=n)
                self.labels = self.processed_df["temp_labels"]
                self.processed_df = self.processed_df.drop(["temp_labels"], axis=1)
            else:
                self.processed_df = self.processed_df.sample(n=n)
                self.update_log("downsampleRows", f"n: {n}")


    def minMaxScale(self):
        scaler = MinMaxScaler()
        columns =self.processed_df.columns
        self.processed_df[columns] = scaler.fit_transform(self.processed_df[columns])
        self.update_log("minMaxScale")

    def processLabels(self):
        self.splitAndDropLabels()
        self.convertStrLabelsToInt()

    def reformatForML(self):
        self.drop_columns(self.drop_cols)
        self.categorical_columns(self.unique_cols)
        self.processLabels()
        self.maniped = True

    def reformatForDivergence(self):
        self.dropInfsAndNaNs()
        self.drop_columns(self.drop_cols)
        self.categorical_columns(self.unique_cols)
        self.processLabels()
        self.maniped = True

    def reformatForClustering(self):
        minority = self.processed_df[self.processed_df[self.label_field] == self.target_label]
        majority = self.processed_df[self.processed_df[self.label_field] == self.benign_label]
        self.processed_df = pd.concat([minority, majority])
        self.processLabels()
        self.drop_columns(self.drop_cols)
        self.categorical_columns(self.unique_cols)
        self.minMaxScale()
        self.maniped = True

    def reformatForCosine(self):
        self.drop_columns(self.drop_cols)
        self.categorical_columns(self.unique_cols)
        self.getLabelledRows(self.target_label)
        self.processLabels()
        self.minMaxScale()
        self.dropInfsAndNaNs()
        self.maniped = True

    def getTrainTestFeatures(self, columns : list[str] = None):
        indices = np.arange(self.processed_df.shape[0])
        if columns is None:
            train_features, test_features, train_labels, test_labels, train_indices, test_indices = train_test_split(self.processed_df, self.labels, indices, test_size=0.2, random_state=100)
            self.update_log("getTrainTestFeatures")
            return train_features, train_labels, test_features, test_labels, train_indices, test_indices
        else:
            train_features, test_features, train_labels, test_labels, train_indices, test_indices = train_test_split(self.processed_df[columns], self.labels, indices, test_size=0.2, random_state=100)
            self.update_log("getTrainTestFeatures", columns)
            return train_features, train_labels, test_features, test_labels, train_indices, test_indices

    def apply_metadata(self):
        '''
        Apply manipulations according to the metadata.
        '''
        if self.operations is None:
            raise RuntimeError('No operations found in metadata file.')
        for operation in self.operations:
            op_name = operation['operation']
            params = operation['params']
            if hasattr(self, op_name):
                if params is None:
                    getattr(self, op_name)
                else:
                    getattr(self, op_name)(**params)
            else:
                warnings.warn(f"Warning: {op_name} not a valid manipulation command.")
                self.operations_log.append(f"Operation {op_name} not recognized")
        self.maniped = True


def saveMetadata(_file: str, drop_fields: list, unique_fields:list, label_field: str, benign_label: str, dst_port:str, ignore_ports: list):
    metadata = {}
    metadata['drop_fields'] = drop_fields
    metadata['unique_fields'] = unique_fields
    metadata['label_field'] = label_field
    metadata['benign_label'] = benign_label
    metadata['dst_port'] = dst_port
    metadata['ignore_ports'] = ignore_ports

    with open(_file, 'w') as md:
        json.dump(metadata, md)



def _reformatForML(df: pd.DataFrame, drop_fields: list, unique_fields: list, label_field: str, benign_label:str, keep_nan:bool, target: str):
    """
    Reformat DataFrame for Machine Learning algorithms
    """
    print("[*] Reformating for ML")
    print("[*] Dropping Drop Fields")
    df = df.drop(drop_fields, axis=1)
    print("[*] Dropping and replacing NaNs/Infs")
    if keep_nan:
        df = df.replace(np.inf, np.nan)
        df = df.replace(np.nan, 0)
    else:
        df = df.replace(np.inf, np.nan)
        df = df.dropna()
    print("[*] Saving and dropping Label Field")
    labels = df[label_field]
    df = df.drop(label_field, axis=1)
    if benign_label == None:
        labels.loc[(labels != target)] = 'Other'
        label_mapping = {"Other": 0,
                        target: 1}
    else:
        print("[*] Building label mapping")
        label_index = labels.loc[((labels != benign_label) & (labels != target))].index
        labels = labels.drop(label_index)
        df = df[~df.index.isin(label_index)]
        label_mapping = {benign_label: 0,
                        target: 1}
    print("[*] Instantiating label mapping")
    labels = labels.replace(label_mapping)
    print("[*] Dealing with unique fields")
    if unique_fields is not None:
        for field in unique_fields:
            unique_field = df[field].unique()
            field_mapping = dict(zip(unique_field, range(len(unique_field))))
            df = df.replace({field: field_mapping})
    return df, labels


def reformatForML(df: pd.DataFrame, metadata: dict, target: str):
    """
    Reformat DataFrame for Machine Learning algorithms
    """
    keep_nan=False
    if "keep_nan" in metadata:
        keep_nan = True
    df, labels = _reformatForML(df, metadata["drop_fields"], metadata["unique_fields"], metadata["label_field"], metadata["benign_label"], keep_nan, target)
    return df, labels

def reformatForClustering(df, metadata, targets):
    print("[*] Reformating for Clustering")
    drop_labels = metadata["drop_fields"]
    label_field = metadata["label_field"]
    unique_fields = metadata["unique_fields"]
    print("[*] Calculating Majority and Minority Classes")
    df_minority = df[df[metadata["label_field"]] == targets[1]]
    df_majority = df[df[metadata["label_field"]] == metadata["benign_label"]]
    del df
    df = pd.concat([df_minority, df_majority])
    if len(targets) == 1:
        print("[*] Building label mapping")
        df[label_field].loc[(df[label_field] != targets[0])] = 'Other'
        label_mapping = {"Other": "0",
                        targets[0]: "1"}
        print("[*] Instantiating label mapping")
        df[label_field] = df[label_field].replace(label_mapping)
        targets.append("0")
    print("[*] Dropping Drop Fields")
    df = df.loc[df[label_field].isin(targets), :]
    df = df.drop(drop_labels, axis=1)
    print("[*] Dropping and replacing NaNs/Infs")
    if "keep_nan" in metadata: 
        df = df.replace(np.inf, np.nan)
        df = df.replace(np.nan, 0)
    else:
        df = df.replace(np.inf, np.nan)
        df = df.dropna()
    if unique_fields is not None:
        for field in unique_fields:
            print("Enumerating field: {}".format(field))
            unique_field = df[field].unique()
            field_mapping = dict(zip(unique_field, range(len(unique_field))))
            df = df.replace({field: field_mapping})
    print("[*] MinMaScaling")
    scaler = MinMaxScaler()
    scaler.fit(df.loc[:,df.columns != label_field])
    df.loc[:,df.columns != label_field] = scaler.transform(df.loc[:,df.columns != label_field])
    print("[*] Saving and dropping Label Field")
    labels = df[label_field]
    df = df.drop(label_field, axis=1)
    print("[*] Preprocessed!")
    return df, labels

def reformatForDiv(df=None, metadata=None):
    """
    Reformat DataFrame for divergence metrics.
    """
    data = df.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    data  = data.drop(metadata["drop_fields"], axis=1)
    if metadata['unique_fields'] is not None:
        for field in metadata['unique_fields']:
            unique_field = data[field].unique()
            field_mapping = dict(zip(unique_field, range(len(unique_field))))
            data = data.replace({field: field_mapping})
    return data

def reformatForCosine(df, metadata, target, sample=0.5):
    print("[*] Reformating for Clustering")
    print("[*] DF shape before dropping NaNs: ", df.shape)
    print(f"[*] Sampling frac={sample}")
    data = df.sample(frac=sample)
    print("[*] Dropping Drop Fields")
    data = data.drop(metadata["drop_fields"], axis=1)
    print("[*] Dealing with unique fields")
    if metadata['unique_fields'] is not None:
        for field in metadata['unique_fields']:
            print(f"[*] Looking at field {field}")
            unique_field = data[field].unique()
            field_mapping = dict(zip(unique_field, range(len(unique_field))))
            data = data.replace({field: field_mapping})
    print("[*] Dropping Label Field")
    data = data[data[metadata["label_field"]] == target]
    data = data.drop(metadata["label_field"], axis=1)
    print("[*] MinMax Scaling")
    data = (data - data.min())/(data.max() - data.min())
    print("[*] Dropping Fixed Value Columns")
    for col in data.columns:
        if data[col].isnull().all():
            data = data.drop([col], axis=1)
    print("[*] Dropping and replacing NaNs/Infs")
    if "keep_nan" in metadata: 
        data = data.replace(np.inf, np.nan)
        data = data.replace(np.nan, 0)
    else:
        data = data.replace(np.inf, np.nan)
        data = data.dropna()
    print("[*] Finished Reformating for Cosine Similarity")
    return data

def getTrainTestFeatures(df: pd.DataFrame, labels: pd.Series):
    indices = np.arange(df.shape[0])
    train_features, test_features, train_labels, test_labels, train_indices, test_indices = train_test_split(df, labels, indices, test_size=0.2, random_state=100)

    train_labels = utils.removeInfinite(train_labels, train_features)
    train_features = utils.removeInfinite(train_features, train_features)

    test_labels = utils.removeInfinite(test_labels, test_features)
    test_features = utils.removeInfinite(test_features, test_features)

    return train_features, train_labels, test_features, test_labels, train_indices, test_indices

