import pandas as pd
import numpy as np
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import yaml

with open(os.path.join(".", 'config', 'config.yml'), 'r', encoding='utf-8') as conf_file:
    config = yaml.load(conf_file, Loader=yaml.SafeLoader)


def print_report(metrics=["tf_idf", "tfidf_lsi", "word2vec_cosine", "word2vec_wmd", "doc2vec"]):
    general_report = pd.DataFrame(columns=metrics)
    all_report = pd.DataFrame()
    for metric in metrics:
        detailed_report, mean_accuracy, positions = evaluate(metric)
        general_report.loc["mean_accuracy", metric] = mean_accuracy
        general_report.loc["mean_position", metric] = np.mean(positions)
        all_report = all_report.append(detailed_report, ignore_index=True)
    pivot_report = general_report.transpose().reset_index().rename(columns={"index": "metrics"})
    pivot_report["mean_accuracy"] = pivot_report["mean_accuracy"].apply(lambda x: round(x, 2))
    figure = sns.barplot(x="metrics", y="mean_accuracy", data=pivot_report)
    return all_report, figure


def evaluate(metric, folder, benchmark="position"):
    """

    :param metric:
    :param benchmark: "top3", "position"
    :return:
    """
    # evaluate
    # load configuration file
    ground_truth_path = config["FileLocation"]["GROUND_TRUTH"]
    metric_path = folder
    metric_folder = metric
    accuracies = []
    detailed_report = pd.DataFrame()
    positions = []
    acc_pos = 0
    acc_instances = 0
    for path, _, files in os.walk(ground_truth_path):
        for file in files:
            file_name = file.split(".xlsx")[0]
            # print("Evaluate for: {}  Metric: {}".format(file_name, metric))
            ground_truth = pd.read_excel(os.path.join(path, file))
            if os.path.exists(os.path.join(metric_path, metric_folder, file_name + ".json")):
                with open(os.path.join(metric_path, metric_folder, file_name + ".json")) as json_file:
                    result = json.load(json_file)
                    # print("result:", result)
            else:
                continue
            ground_truth_dict = {ground_truth.Index[i]: ground_truth.EU_Top1_idx[i] for i in range(len(ground_truth))}
            sum_instances = 0
            pos = 0
            unmatched = 0
            for item_dict in result["Paragraphs"]:  # this is a list
                tw_para = item_dict["Index"]
                sim_list = item_dict["Similar Paragraphs"]  # list of dictionary
                if benchmark == "top3":
                    sim_list = sim_list[:3]
                candidate_list = [sim_par["Index"] for sim_par in sim_list]
                if ground_truth_dict[tw_para] is not np.nan:
                    if ground_truth_dict[tw_para] in candidate_list[:3]:
                        # print("found in rank {}".format(candidate_list.index(ground_truth_dict[tw_para])+1))
                        pos += 1
                        acc_pos += 1
                    if ground_truth_dict[tw_para] in candidate_list:
                        position = sim_list[candidate_list.index(ground_truth_dict[tw_para])]["Similarity Rank"]
                        # print(position)
                        positions.append(position + 1)
                    sum_instances += 1
                    acc_instances +=1
                else:
                    unmatched += 1
            accuracy = pos / sum_instances
            detailed_report = detailed_report.append({"doc": "Ground Truth {}".format(file_name),
                                                      "sum of instances": round(sum_instances),
                                                      "unmatched paragraphs": round(unmatched),
                                                      metric + "_accuracy": round(accuracy, 3),
                                                      metric + "_mean_positions": np.mean(positions)}, ignore_index=True)

            accuracies.append(accuracy)
            # print("unmatched paragraph: ", unmatched)
            # print("total instances: ", sum_instances)
            # print("accuracy: {}".format(pos / sum_instances))
    mean_accuracy = np.mean(accuracies)
    acc_accuracy = acc_pos / acc_instances
    # print(detailed_report)
    # print("mean_accuracy for metrics: {}".format(metric), mean_accuracy)
    return detailed_report, mean_accuracy, positions, acc_accuracy


if config["Reporting"]:
    all_report, barplot = print_report()
    barplot.figure.savefig(os.path.join(config["Report_loc"], "result.png"))
    plt.show()


def gen_report(mode="accuracy", metrics=["tfidf", "tfidf_lsi", "word2vec_cosine", "word2vec_wmd", "doc2vec"]):
    """ mode: "accuracy" / "mean_positions"
    """
    summary = None
    for metric in metrics:
        detailed_report, mean_accuracy, positions, acc_accuracy = evaluate(metric)
        if summary is None:
            summary = detailed_report[["doc", metric + "_" + mode]]
        else:
            summary = pd.concat([summary, detailed_report[metric + "_" + mode]], axis=1)
    return summary
