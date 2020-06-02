import os
import pandas as pd


class CorpusGenerator:
    """generates data from selected location and output dataframes.
        task: "SIM" (for similarity) ; "RELATION" (for Relation Extraction)
        data_location: absolute location of data;
        country_list: ["TW", "EU"],
        generate_only_for: dict. Takes a dict of index list of documents, country as key.
                                 For example: {"EU": ["1","2"],"TW":["03-1","04"]}
        filtered: dict. Excluding documents to be included in corpus. Format is the same as generate_only_for.
        Note: either generate_only_for & filtered should be None
        """

    def __init__(self, country_list, task, data_location="default", generate_only_for=None, filtered=None):

        if data_location == "default":
            self.data_location = os.path.join("..", "..","data", "processed_data")
        else:
            self.data_location = data_location
        self.task = task
        self.country_list = country_list
        # define location_dict here to store folder name for each country:
        self.loc_dict = {"TW": "TW",
                         "EU": "EU_validated"}
        assert any([x is None for x in [generate_only_for, filtered]])
        if generate_only_for is not None:
            self.generate_only_for = generate_only_for
            print("generate only for:", self.generate_only_for)
        else:
            self.generate_only_for = None
        if filtered is not None:
            self.filtered = filtered
        else:
            self.filtered = None

    def get_index_from_filename(self, country, filename):
        if country == "TW":
            index = filename.split(" ")[0]
        elif country == "EU":
            index = filename.split("No._")[-1].split(".csv")[0]
        return index

    def read_from_location(self):
        dataframe_all = pd.DataFrame(columns=["Index", "Text"])
        for country in self.country_list:
            country_dataframe = pd.DataFrame(columns=["Index", "Text"])
            file_location = os.path.join(self.data_location, self.loc_dict[country])
            all_files = []
            target_files = []
            for dirpath, dirname, files in os.walk(file_location):
                all_files = files
            if (self.generate_only_for is None) and (self.filtered is None):
                # default mode: extract all files from the folder
                target_files = all_files
                print("Including all data")
            elif (self.filtered is not None) and (self.generate_only_for is None):
                # exclude the files from all_files
                if country == "TW":
                    target_files = list(filter(lambda x: x.split(" ")[0] not in self.filtered[country], all_files))
                elif country == "EU":
                    target_files = list(
                        filter(lambda x: x.split("No._")[-1].split(".csv")[0] not in self.filtered[country], all_files))
            elif (self.filtered is None) and (self.generate_only_for is not None):
                # generate only for listed
                if country == "TW":
                    target_files = list(filter(lambda x: x.split(" ")[0] in self.generate_only_for[country], all_files))
                elif country == "EU":
                    # print(self.generate_only_for[country])
                    target_files = ["UN_Regulation_No._{}.csv".format(i) for i in self.generate_only_for[country]]
            # print("target_file:",country, target_files)
            for document in target_files:
                sgl_dataframe = pd.read_csv(os.path.join(file_location, document))
                if "paragraph" in list(sgl_dataframe.columns):
                    sgl_dataframe = sgl_dataframe.rename(columns={"paragraph": "Index"})
                sgl_dataframe.loc[:, "Index"] = "{}_{}_".format(country, self.get_index_from_filename(country,
                                                                                                      document)) + sgl_dataframe.loc[
                                                                                                                   :, "Index"]
                country_dataframe = country_dataframe.append(sgl_dataframe.loc[:, ["Index", "Text"]],
                                                             ignore_index=True)
            dataframe_all = dataframe_all.append(country_dataframe, ignore_index=True)
        dataframe_all = dataframe_all.dropna(subset=["Text"])
        # clean the data

        return dataframe_all
