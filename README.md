## Basic Information

This is a tool designed for auditing the issues with network datasets, based on flow statistics. **This tool can be used to run a series of tests on network data, detailed below. We have provided a subset of CIC IDS 18 to run these tests, in zipped format, in the `data/` directory.** The tool assumes that the provided dataset file is a CSV of comma separated flow statistics alongside a label field. Many of the tests are comparative, requiring some sensible baseline to evaluate the complexity of a given subset of the dataset. In the case of network intrusion data, the label field may be the attack labels and the baseline traffic may be the benign data.

## Requirements

Start a fresh Python3 virtual environment:

`python -m venv ~/[ENV_NAME]`

Activate the virtual environment:

`source ~/[ENV_NAME]/bin/activate`

Then simply install the requirements found in the `requirements.txt` file:

`pip install -r requirements.txt`

## Data

We've included a small sample of *CIC IDS 2018* to run our tests on. Other datasets need to be downloaded from their respective sources.

We've run this tool on the following datasets:

- [UNSW NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- [CIC IDS 2017](https://www.unb.ca/cic/datasets/ids-2017.html)
- [CSE-CIC IDS 2018](https://www.unb.ca/cic/datasets/ids-2018.html)
- [ISCX 2012](https://www.unb.ca/cic/datasets/ids.html)
- [TON IoT](https://research.unsw.edu.au/projects/toniot-datasets)
- [Bot IoT](https://research.unsw.edu.au/projects/bot-iot-dataset)
- [CTU 13](https://www.stratosphereips.org/datasets-ctu13)
- [ODDS](https://odds.cs.stonybrook.edu/)


## Example Commands

(NB: see 'Data' section above. These commands will only work if the necessary data is downloaded and provided correctly. We've included a small sample of *CIC IDS 2018* to run our tests on, but this must be extracted from the provided zip file in the `data/` directory!)

- **Metric**: `python3 src/whiff.py --metadata metadata/cic2018/metadata.json --results results/CIC18_trunc/ --target Bot  --folder --csv data/CIC18_trunc/ --metric KLDivergence`
- **Test**: `python3 ./src/whiff.py --metadata metadata/cic2018/metadata.json --results results/CIC18_trunc/ --target FTP-BruteForce --folder --csv data/CIC18_trunc/  --test CosineTest`

A list of valid options for the `--metric` and `--sniff` flags can be found in the *Metrics* and *Tests* sections below, or you can run WhiffSuite with the `--metriclist` or `--snifflist` flags. 

Our heuristics tests can be batch run for each dataset via the files in the `scripts/` folder.

- **Batch Tests**: `./scripts/CIC18_trunc_calculations.sh`

## Other

We provide other files necessary to recreate our experimentation/metric calculation can be found in `nb/`. More details can be found in the README in `nb/`

## Metadata Format

Our metadata assumes that the dataset file is comma-separated CSV with labelled columns. Metadata format is JSON. The recognised keys are as follows:

* `label_field`: string - name of label column
* `benign_label`: string - name of benign class
* `dst_port`: string - name of the destination port field
* `drop_fields`: list of strings - drop the named columns
* `unique_fields`: list of strings - enumerate unique items in named columns
* `string_fields`: list of strings - convert the named columns to strings, alongside some simple preprocessing
* `numeric_fields`: list of strings - convert the named columns to floats, only necessary for columns which (incorrectly) have mixed dtypes
* `background_ports`: list of ints - consider malicious traffic from the named ports to be background traffic (accidently labelled as malicious)
* `backwards_packets_field`: string - name of the field which contains the total number of backwards packets of a flow
* `source_bytes_field`: string - name of the field which contains the total number of bytes originating from the source IP
* `destination_bytes_field`: string - name of the field which contains the total number of bytes originating from the destination IP
* `timestamp_field`: string - name of field containing flow timestamps
* `timestamp_format`: string - timestamp format string used to parse timestamp_field
* `comment`: string - unused, just useful to comment on the metadata file.


Note that WhiffSuite provides some standard commands for manipulating datasets which rely on the above metadata. However, it does minimal checking that the metadata file is coherent. For instance, running the Port Sniff Test requires that the `dst_port` column is defined. However, if this column is also included in `drop_fields`, there is no guarantee that the test will work (as the port field will have been dropped before the Sniff test is applied). To provide more control over dataset manipulation, commands can be provided via the `operation` key in the metadata file.

* `operations`: list of dictionaries - operation is defined by the `operation` key, parameters for the operation are defined by the `params` key. All methods of the Manipulator class are valid operations. Example operation: `[{"operation": "drop_columns", "params": {"cols_to_drop": ["Col1", "Col2"]}, {"operation": "minMaxScale"}}]`



## Tests

* Cosine Test: Approximate cluster size and ratio of identical flows. Outputs one result for each. Both in range $[0,1]$
* Port Test: Percentage of background flows to benign flows, based off of ignore_ports ports. Outputs single result in range $[0,1]$
* Nearest Neighbours Test: Percentage of flows that are misclassified according to the Edited Nearest Neighbours criteria. Outputs single result in range $[0,1]$
* Single Feature Efficacy Test: Considers only a single feature to train our random forest. Outputs JSON file with F1 score for each feature.
* Rolling Importances Test: Order features according to mutual information, measure efficacy of 5 features together for classification using random forest. Outputs JSON file with F1 score for each group of 5 features - **Unused/Untested**
* Backwards Packets Test: Percentage of flows with no backwards packets to total flows. Outputs single result in range $[0,1]$ - **Unused/Untested**
* Siamese Network: Measure rate of F1 gain of a few-shot learning siamese network - **Unused/Untested**

## Metrics

* Gini Impurity
* Info Gain
* Entropy
* Normalised Entropy

## Distances

* KL Divergence
* JS Divergence
* KS Test
* KS Test (via KDE Estimate)
* EM Distance


## TODO

* Improve the manner in which the results are displayed --- maybe spit numbers + visuals into HTML file?
* Add graph metrics
* Improve manner that results are handed off to visuals
* Improve visuals
* Add metadata files for existing datasets
* Add testing
* Add siamese network test
