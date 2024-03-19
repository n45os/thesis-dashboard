import json
from analysis.loaders import load_data, only_adversarial_data, only_benign_data



class BenignSimulationResult:
    def __init__(self, dataset, model, strategy, data = None):
        self.dataset = dataset
        self.model = model
        self.strategy = strategy

        self.stats: Stat = None
        if data is not None:
            self.load_stats(data)
            
        print(self.stats)

    def load_stats(self, data):
        ben_dat = get_benign_data_of(data, self.model, self.dataset, self.strategy)
        if ben_dat is None or len(ben_dat) == 0:
            print(
                f"No benign data found for {self.model}, {self.dataset}, {self.strategy}"
            )
            return
        if self.stats is None:
            self.stats = Stat(ben_dat[0])


class AttackSimulationResult:
    def __init__(
        self,
        dataset,
        model,
        strategy,
        attack,
        perturbation_type,
        attack_fraction,
        data = None,
        benign_res: BenignSimulationResult = None,
    ):
        self.attack = attack
        self.attack_fraction = attack_fraction
        self.perturbation_type = perturbation_type
        self.dataset = dataset
        self.model = model
        self.strategy = strategy

        self.benign_result: BenignSimulationResult = benign_res
        self.stats: Stat = None
        if data is not None:
            self.load_stats(data)

    def load_stats(self, data):
        adv_dat = get_advers_data_of(
            data,
            self.attack,
            self.attack_fraction,
            self.perturbation_type,
            self.model,
            self.dataset,
            self.strategy,
        )
        if self.benign_result is None:
            ben_dat = get_benign_data_of(data, self.model, self.dataset, self.strategy)
            if len(ben_dat) == 0:
                print(
                    f"No benign data found for {self.model}, {self.dataset}, {self.strategy}"
                )
            else:
                self.benign_result = BenignSimulationResult(
                    self.dataset, self.model, self.strategy, ben_dat
                )
        else:
            print("Benign result already loaded")
        
        if adv_dat is None or len(adv_dat) == 0:
            print(
                f"No adversarial data found for {self.attack}, {self.attack_fraction}, {self.model}, {self.dataset}, {self.strategy}"
            )
            return
        if self.stats is None:
            self.stats = Stat(adv_dat[0])
        else:
            print("Stats already loaded")
        
        if self.benign_result.stats is not None:
            self.stats.add_impact(
                self.benign_result.stats.end_accuracy, self.stats.end_accuracy
            )


class Stat:
    def __init__(self, data):
        round_data, config, timing = data
        # round_impact is not the same as the impact. the impact is the difference between the benign and adversarial final accuracy
        self.end_accuracy: float = (
            round_data["accuracy"][-1][1] if "accuracy" in round_data else 0
        )
        if "impact" in round_data or "round_impact" in round_data:
            if "impact" in round_data:
                round_impacts: list = round_data["impact"]
            elif "round_impact" in round_data:
                round_impacts: list = round_data["round_impact"]
            self.average_round_impact: float = sum(
                [im for _, im in round_impacts]
            ) / len(round_impacts)
            print(self.average_round_impact)
            self.impact = -1
        else:
            self.impact = None
            self.average_round_impact = 0
        
        print(self.end_accuracy)

    def add_stat(self, round_data):
        self.end_accuracy = (
            round_data["accuracy"][-1][1] if "accuracy" in round_data else 0
        )
        if "impact" in round_data or "round_impact" in round_data:
            if "impact" in round_data:
                self.round_impacts = round_data["impact"]
            elif "round_impact" in round_data:
                self.round_impacts = round_data["round_impact"]
        self.average_round_impact = sum(self.round_impacts) / len(self.round_impacts)

    def add_impact(self, benign_accuracy, attack_accuracy):
        self.impact = benign_accuracy - attack_accuracy

    def get_dict(self):
        if self.impact is None or self.impact == -1:
            return {
                "end_accuracy": self.end_accuracy,
                "average_round_impact": "Not available",
                "impact": "Not available",
            }
        else:
            return {
                "end_accuracy": self.end_accuracy,
                "average_round_impact": self.average_round_impact,
                "impact": self.impact,
            }

    def get_csv(self):
        if self.impact is None or self.impact == -1:
            return f"{self.end_accuracy}, Not available, Not available"
        else:
            return f"{self.end_accuracy}, {self.average_round_impact}, {self.impact}"


class Summary:
    def __init__(self, root_folder, save_path):
        self.benign_results: list[BenignSimulationResult] = []
        self.attack_results: list[AttackSimulationResult] = []

        self.path = root_folder
        self.save_path = save_path

        data = load_data(self.path)

        self.get_all_results(data)

        self.add_impacts()

    def get_all_strategies(self, data):
        # get the data an retrun a list of the name of all strategies in the data
        strategies = []
        for round_data, config, timing in data:
            strat = config.get("strategy", {}).get("name")
            if strat not in strategies:
                strategies.append(strat)

        return strategies

    def get_all_dataset_model_pairs(self, data):
        pairs = []
        for round_data, config, timing in data:
            mdl = config.get("model", {}).get("name")
            dtst = config.get("dataset", {}).get("name")
            if (mdl, dtst) not in pairs:
                pairs.append((mdl, dtst))

        return pairs

    def get_all_attacks_fraction_pairs(self, data):
        pairs = []
        for round_data, config, timing in data:
            att_frac = config.get("adversaries_fraction", {})
            att_name = config.get("adversaries", {}).get("advers_fn")
            att_pertr = config.get("adversaries", {}).get("perturbation_type")
            if (att_name, att_frac, att_pertr) not in pairs:
                pairs.append((att_name, att_frac, att_pertr))

        return pairs
    
    def get_benign_of(self, mdl, dtst, strat):
        for res in self.benign_results:
            if res.dataset == dtst and res.model == mdl and res.strategy == strat:
                return res
        return None

    def get_all_results(self, data):
        ben_res = None

        for mdl, dtst in self.get_all_dataset_model_pairs(data):
            for strat in self.get_all_strategies(data):
                # check if the benign result is already in the list
                if len(self.benign_results) > 0:
                    for res in self.benign_results:
                        if (
                            res.dataset == dtst
                            and res.model == mdl
                            and res.strategy == strat
                        ):
                            break
                    else:
                        ben_res = BenignSimulationResult(dtst, mdl, strat, data)
                        # ben_res.load_stats(data)
                        self.benign_results.append(ben_res)

                else:
                    ben_res = BenignSimulationResult(dtst, mdl, strat, data=data)
                    # ben_res.load_stats(data)
                    self.benign_results.append(ben_res)


        for att, frac, pertr_type in self.get_all_attacks_fraction_pairs(data):
            for mdl, dtst in self.get_all_dataset_model_pairs(data):
                for strat in self.get_all_strategies(data):
                 
                        # # check if the attack result is already in the list
                        # if len(self.attack_results) > 0:
                        #     # # avoid adding a duplicate
                        #     # for res in self.attack_results:
                        #     #     if (
                        #     #         res.dataset == dtst
                        #     #         and res.model == mdl
                        #     #         and res.strategy == strat
                        #     #         and res.attack == att
                        #     #         and res.attack_fraction == frac
                        #     #     ):
                        #     #         break
                        # else:
                    ben_res = self.get_benign_of(mdl, dtst, strat)
    
                    att_res = AttackSimulationResult(
                        dtst, mdl, strat, att, pertr_type, frac, data, ben_res
                    )
                    # att_res.load_stats(data)
                    self.attack_results.append(att_res)

    def add_impacts(self):
        for res in self.attack_results:
            if res.stats is not None:
                if res.benign_result.stats is not None:
                    res.stats.add_impact(
                        res.benign_result.stats.end_accuracy, res.stats.end_accuracy
                    )

    def save_attack_results(self):
        # save the attack results in json and csv format

        results = []
        # create the folder
        import os

        # add a uuid to the filename
        import uuid
        idd = str(uuid.uuid4())[:8]

        os.makedirs(f"{self.save_path}/organized_results", exist_ok=True)

        # make csv header
        with open(f"{self.save_path}/organized_results/summary-{idd}.csv", "w") as f:
            f.write(
                "dataset, model, strategy, attack, perturbation_type, attack_fraction, end_accuracy, average_round_impact, impact\n"
            )

        for res in self.attack_results:
            # make a python dict from the results
            if res.stats is not None:
                results.append(
                    {
                        "dataset": res.dataset,
                        "model": res.model,
                        "strategy": res.strategy,
                        "attack": res.attack,
                        "attack_fraction": res.attack_fraction,
                        "stats": res.stats.get_dict(),
                    }
                )

                # add a line to the csv file
                with open(f"{self.save_path}/organized_results/summary-{idd}.csv", "a") as f:
                    f.write(
                        f"{res.dataset}, {res.model}, {res.strategy}, {res.attack}, {res.perturbation_type}, {res.attack_fraction}, {res.stats.get_csv()}\n"
                    )
            else:
                results.append(
                    {
                        "dataset": res.dataset,
                        "model": res.model,
                        "strategy": res.strategy,
                        "attack": res.attack,
                        "attack_fraction": res.attack_fraction,
                        "stats": "No stats available",
                    }
                )
                with open(f"{self.save_path}/organized_results/summary-{idd}.csv", "a") as f:
                    f.write(
                        f"{res.dataset}, {res.model}, {res.strategy}, {res.attack},Not available, {res.attack_fraction}, Not available, Not available, Not available\n"
                    )

        with open(f"{self.save_path}/organized_results/summary-{idd}.json", "w") as f:
            json.dump(results, f, indent=2)

        # create a csv where it has dataset, model, strategy, attack1_frac1, attack2_frac1 ... , attack1_frac2, attack2_frac2 ...
        # for each result, add the impact. attack1_frac1, attack2_frac1 will be unique and not repeated

       # Create the CSV header
        header_elements = ["dataset", "model", "strategy"]
        unique_attacks = set()
        for res in self.attack_results:
            if res.stats is not None:
                # fraction is not 0
                if res.attack_fraction != 0:
                    attack_key = f"{res.attack}_{res.attack_fraction}"
                    if res.attack == "minmax" or res.attack == "minsum" or res.attack == "tailored":
                        attack_key = f"{res.attack}_{res.attack_fraction}_{res.perturbation_type}"
                    unique_attacks.add(attack_key)

        header_elements.extend(sorted(unique_attacks))  # Sort the attacks for consistent column order
        header = ", ".join(header_elements)

        # Write the header to the file
        with open(f"{self.save_path}/organized_results/summary_impacts-{idd}.csv", "w") as f:
            f.write(header + "\n")
            
        # Step 1: Aggregate the results
        results_dict = {}
        for res in self.attack_results:
            if res.stats is not None and res.attack_fraction != 0:
                key = (res.dataset, res.model, res.strategy)
                if key not in results_dict:
                    results_dict[key] = {attack: "Not available" for attack in sorted(unique_attacks)}
                if res.attack == "minmax" or res.attack == "minsum" or res.attack == "tailored":
                        attack_key = f"{res.attack}_{res.attack_fraction}_{res.perturbation_type}"
                else:
                    attack_key = f"{res.attack}_{res.attack_fraction}"
                results_dict[key][attack_key] = str(res.stats.impact) if res.stats.impact is not None and res.stats.impact != -1 else "Not available"

        # Step 2: Write the aggregated results to the file
        with open(f"{self.save_path}/organized_results/summary_impacts-{idd}.csv", "a") as f:
            for key, impacts in results_dict.items():
                line_elements = list(key) + [impacts[attack] for attack in sorted(unique_attacks)]
                line = ", ".join(line_elements)
                f.write(line + "\n")


def get_advers_data_of(all_data, attack, attack_fraction,perturbation_type , model, dataset, strategy):
    adversarial_data = []
    for round_data, config, timing in all_data:
        mdl = config.get("model", {}).get("name")
        dtst = config.get("dataset", {}).get("name")
        strat = config.get("strategy", {}).get("name")
        att_frac = config.get("adversaries_fraction", {})
        pertr_type = config.get("adversaries", {}).get("perturbation_type")
        att_name = config.get("adversaries", {}).get("advers_fn")
        if (
            mdl == model
            and dtst == dataset
            and strat == strategy
            and att_frac == attack_fraction
            and pertr_type == perturbation_type
            and att_name == attack
        ):
            adversarial_data.append((round_data, config, timing))
    if len(adversarial_data) > 1:
        print(
            f"More than one adversarial data found for {attack}, {attack_fraction}, {perturbation_type}, {model}, {dataset}, {strategy}"
        )
    if len(adversarial_data) == 0:
        print(
            f"No adversarial data found for {attack}, {attack_fraction}, {perturbation_type}, {model}, {dataset}, {strategy}"
        )
        return None
    return adversarial_data


def get_benign_data_of(all_data, model, dataset, strategy):
    benign_data = []
    for round_data, config, timing in all_data:
        adv_frac = config.get("adversaries_fraction", 0)
        mdl = config.get("model", {}).get("name")
        dtst = config.get("dataset", {}).get("name")
        strat = config.get("strategy", {}).get("name")
        if adv_frac == 0 and mdl == model and dtst == dataset and strat == strategy:
            benign_data.append((round_data, config, timing))
    if len(benign_data) > 1:
        print(f"More than one benign data found for {model}, {dataset}, {strategy}")
    if len(benign_data) == 0:
        print(f"No benign data found for {model}, {dataset}, {strategy}")
        return None
    return benign_data


def extract_final_data(round_data):
    final_accuracy = round_data["accuracy"][-1][1] if "accuracy" in round_data else 0
    final_loss = round_data["loss"][-1][1] if "loss" in round_data else 0
    return final_accuracy, final_loss


def calculate_impact(all_data):
    model_dataset = [
        ("femnist", "my_simple_net"),
        ("cifar10", "resnet"),
        ("cifar100", "resnet"),
        ("mnist", "my_simple_net"),
    ]
    strategies = [
        "fed_avg",
        "multikrum",
        "krum",
        "fed_median",
        "bulyan",
        "trimmed_mean",
        "dnc",
        "cc",
    ]
    attacks = ["lie", "minmax", "minsum", "tailored", "pga"]
    attack_fraction = [0.25, 0.4]

    impacts = Impacts()
    for m_d in model_dataset:
        for strat in strategies:
            benign_data = get_benign_data_of(all_data, m_d, strat)
            if len(benign_data) == 0:
                continue
            ben_accuracy, ben_loss = extract_final_data(benign_data[0][0])
            for attack in attacks:
                for a_f in attack_fraction:
                    this_imp = impacts.get_impact(attack, a_f)
                    if this_imp is None:
                        print("No impact like this")
                        this_imp = Impact(attack, a_f)
                        impacts.add_impact(this_imp)
                    adversarial_data = get_advers_data_of(
                        all_data, attack, a_f, m_d, strat
                    )
                    if len(adversarial_data) == 0:
                        continue
                    adv_accuracy, adv_loss = extract_final_data(adversarial_data[0][0])
                    this_imp.add_entry(
                        m_d,
                        ben_accuracy,
                        ben_loss,
                        adv_accuracy,
                        adv_loss,
                    )
    # save in json format
    impacts.save_in_json()
    return impacts




def extract_impact(all_data):
    benign_data = only_benign_data(all_data)
    adversarial_data = only_adversarial_data(all_data)


# data = load_data("results/r30_c100_f60_b10_ev10_lr0.01_m0.9_le3 4")


# summary = Summary(data)
# summary.save_attack_results()
