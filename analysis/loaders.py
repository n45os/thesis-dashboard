import glob
import json
import os

def load_data(root_folder):
    data_list = []
    for subdir, dirs, files in os.walk(root_folder):
        for file in glob.glob(subdir + "/*.json"):
            with open(file, "r") as f:
                data = json.load(f)
                round_data = data.get("history", {}).get("metrics_centralized", {})
                config_data = data.get("config", {})
                # there is a new key in the data, "timing". This is not present in the old data
                if "timing" in data:
                    timing_data = data.get("timing", {})
                else:
                    timing_data = {}
                data_list.append((round_data, config_data, timing_data))
    return data_list

def retrieve_benign_data_for_config(config_data, all_data):
    # Filter for benign data with matching config except for adversaries_fraction
    benign_data = []
    for round_data, config, timing_data in all_data:
        if config.get("dataset", {}).get("name") == config_data.get("dataset", {}).get("name") and \
           config.get("strategy", {}).get("name") == config_data.get("strategy", {}).get("name") and \
           config.get("model", {}).get("name") == config_data.get("model", {}).get("name") and \
           config.get("adversaries_fraction", 1) == 0.0:
            benign_data.append((round_data, config, timing_data))
    return benign_data

def retrieve_adversarial_data_for_config(config_data, all_data):
    # Filter for adversarial data with matching config except for adversaries_fraction
    adversarial_data = []
    for round_data, config, timing_data in all_data:
        if config.get("dataset", {}).get("name") == config_data.get("dataset", {}).get("name") and \
           config.get("strategy", {}).get("name") == config_data.get("strategy", {}).get("name") and \
           config.get("model", {}).get("name") == config_data.get("model", {}).get("name") and \
           config.get("adversaries_fraction", 0) != 0.0:
            adversarial_data.append((round_data, config, timing_data))
    return adversarial_data

def only_benign_data(data_list):
    benign_data = []
    for round_data, config, timing_data in data_list:
        if config.get("adversaries_fraction", 0) == 0:
            benign_data.append((round_data, config, timing_data))
    return benign_data

def only_adversarial_data(data_list):    
    adversarial_data = []
    for round_data, config, timing_data in data_list:
        if config.get("adversaries_fraction", 0) != 0:
            adversarial_data.append((round_data, config, timing_data))
    return adversarial_data

def count_benign_data(data_list):
    return len(only_benign_data(data_list))

def count_adversarial_data(data_list):
    return len(only_adversarial_data(data_list))

def get_existing_adversaries_fractions(data_list):
    adversaries_fractions = set()
    for _, config, _ in data_list:
        adversaries_fractions.add(config.get("adversaries_fraction", 0))
    # remove 0.0 from the set
    adversaries_fractions.remove(0.0)
    return sorted(list(adversaries_fractions))


