import re
import PyPDF2 as pypdf
import pandas as pd


class DataPreprocessorEU:

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.document_name = pdf_path.split("/")[-1]
        print(
            "Preprocessing for document {}".format(self.document_name))
        pdf = open(self.pdf_path, mode="rb")
        self.document = pypdf.PdfFileReader(pdf)
        assert self.document is not None
        print("total_page of document: ", self.document.numPages)

    def get_page(self, page_num):
        page = None
        if self.document is not None:
            page = self.document.getPage(page_num - 1)
            page = page.extractText().replace("\n", "")
            page = " ".join(page.split())
        else:
            print("document not found")
        return page

    def header_footer_removal(self, page_num):
        page = self.get_page(page_num)
        header = "E\/ECE(?:\/\w*[.]*\d*)*\s{}".format(page_num)
        header_withoutpage = "E\/ECE(?:\/\w*[.]*\d*)*"
        if len(re.findall(header, page)) != 0:
            # print("found header in page {}".format(page_num),re.findall(header,page))
            removed = re.sub(header, "", page).strip()
            if len(re.findall(header_withoutpage, page)) != 0:
                removed = re.sub(header_withoutpage, "", removed).strip()
            # print("print header!")
        elif len(re.findall(header_withoutpage, page)) != 0:
            removed = re.sub(header_withoutpage, "", page).strip()
            # print("found header!")
        else:
            removed = page
        # footer = " \d{1,2} [A-Z].*"
        # if len(re.findall(footer, removed)) != 0:
        # print("found footer in page {}".format(page_num), re.findall(footer, removed))
        # removed = re.sub(footer, "", removed)
        return removed

    def is_contentpage(self, page_num):
        page = self.get_page(page_num)
        is_contentpage = False
        if len(re.findall("\.{3,}", page)) >= 3:
            is_contentpage = True
        return is_contentpage

    def is_annex(self, page_num):
        page = self.header_footer_removal(page_num)
        is_annex = False
        if re.match(r"Annex\s\d+\s", page[:50]):
            is_annex = True
        return is_annex

    def extract_titles(self, page_num):
        if self.is_contentpage(page_num) is True:
            page = self.get_page(page_num)
            main_titles = [title for title in re.findall(r"(\d+[.]\s+[A-Za-z\s]+)+\s+[.]+", page) if
                           "Foreword" not in title]
            main_titles = [re.sub(r"\s+", " ", title) for title in main_titles]
            return main_titles

    def special_tokenize(self, text):
        # paragraph_mention = "(?:in|of|and|or|to|see|See|by)*\s*(?:paragraph|Paragraph|para\.|paragraphs)\s(?:\d+[.](?:\d+.)*)(?:(?:(?:\sand\s)*(?:,\s)*(?:\d+(?:[.]\d+)+.))*)*"
        paragraph_mention = "(?:in|of|and|or|to|see|See|by\s)*(?:(?:paragraph|Paragraph|para[.]|paragraphs| and|[,]| or)\s((?:\d+[.])+))"
        # for mention in re.findall(paragraph_mention, text):
        #    if mention.split()[0] in ["in", "of", "and", "or", "to", "see", "See", "by"]:
        #        prefix = mention.split()[0]
        #    else:
        #        prefix = ""
        #    text = re.sub(mention.strip(),r"{} <paragraph: {}>".format(prefix,",".join([para for para in re.findall(r"\d+[.](?:\d+.)*",mention)])),text)
        text = re.sub(paragraph_mention, r" Paragraph-\1", text)
        annex = r"(?:Annex|annex|Appendix)\s(\d+)"
        text = re.sub(annex, r"Annex-\1", text)
        revision = r"Revision\s(\d+)"
        text = re.sub(revision, r"Revision-\1", text)
        reg = r"Regulation\sNo\.\s(\d+)"
        text = re.sub(reg, r"Regulation-No._\1", text)
        law = r"(Supplement|Table|clause|Figure)\s(\d+)"
        text = re.sub(law, r"\1_\2", text)
        number = r"No\.\s(\d+)"
        text = re.sub(number, r"No.\1", text)
        date = r"([0-9]{1,})\s(January|February|March|April|May|June|July|August|September|October|November|December)\s([0-9]{4})"
        text = re.sub(date, r"\1-\2-\3", text)
        split = r'\s((?:[1-9][0-9]{0,2}[.])+)\s'
        text = re.sub(split, r"<PARAGRAPH \1>", text)
        text = re.sub(" +", " ", text)
        return text

    def all_content(self):
        # start from the page after index and before the start of annex
        start_index = 0
        end_index = self.document.numPages
        all_content = ""
        for index in range(self.document.numPages):
            page_num = index + 1
            try:
                if self.is_annex(page_num) is True:
                    end_index = page_num - 1
                    print("main page: -P.{}".format(end_index))
                    break
            except:
                break
        print(end_index)
        for index in range(5):
            try:
                if self.is_contentpage(index + 1) is True:
                    start_index = index + 1
            except:
                break
        print("starting from page: ", start_index + 1)

        for index in range(start_index, end_index):
            try:
                removed = self.header_footer_removal(index + 1)
                all_content = all_content + " " + removed
                special_tokenized = self.special_tokenize(all_content)
            except:
                special_tokenized = ""
        return special_tokenized

    def split_paragraphs(self):
        all_content = self.all_content()
        print(all_content)
        contentlist = re.split(r"(<PARAGRAPH .*?>)", all_content)[1:]
        df = pd.DataFrame(columns=["paragraph", "Text"])
        df.loc[:, "paragraph"] = [contentlist[i] for i in range(0, len(contentlist), 2)]
        df.loc[:, "Text"] = [contentlist[i] for i in range(1, len(contentlist) + 1, 2)]
        return df


class DataPreprocessorTW:

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.document_name = pdf_path.split("\\")[-1]
        print(self.document_name)
        self.docu_index = self.document_name.split()[0]
        if self.docu_index[0] == "0":
            self.docu_index = self.docu_index[1:]
        if "-0" in self.docu_index:
            self.docu_index = self.docu_index.split("-")[0]
        print("docuindex", self.docu_index)
        print(
            "Preprocessing for document {}".format(self.document_name))
        pdf = open(self.pdf_path, mode="rb")
        self.document = pypdf.PdfFileReader(pdf)
        assert self.document is not None
        print("total_page of document: ", self.document.numPages)
        self.pages = []

    def get_page(self, page_num):
        page = None
        if self.document is not None:
            page = self.document.getPage(page_num - 1)
            page = page.extractText().replace("\n", "")
            page = " ".join(page.split())
        else:
            print("document not found")
        return page

    def clean_header(self):
        pages = []
        for page in range(self.document.numPages):
            # print(data.get_page(page+1))
            self.pages.append(self.get_page(page + 1))

        if self.document.numPages == 1:
            self.pages = [self.pages[0].split("1-")[-1]]

        else:
            flag = 1
            while len(set([page[:flag] for page in self.pages])) == 1:
                flag = flag + 1
            self.pages = [page[flag - 1:] for page in self.pages]
        return self.pages

    def match_trim_pagenum(self):
        cleaned_pages = []
        for page in self.pages:
            if re.match(r"{}-".format(str(self.pages.index(page) + 1)), page):
                page = page[len(str(self.pages.index(page) + 1)) + 1:]
                cleaned_pages.append(page)
            else:
                cleaned_pages = self.pages
        print("cleanedpage:", cleaned_pages)
        return cleaned_pages

    def split(self):
        self.clean_header()
        cleaned_pages = self.match_trim_pagenum()
        all = " ".join(cleaned_pages)
        mentions = "(?:in|of|and|or|to|see|See|by)*\s*(?:paragraph|Paragraph|para[.]|paragraphs| and|[,]| or| either| with)\s({}(?:\.\d+)*)".format(
            self.docu_index)
        all = re.sub(mentions, " Paragraph-\1", all)
        print(self.docu_index)
        split = r"\s({}(?:\.\d+)*) ".format(self.docu_index)
        splitted = re.split(split, all)
        print("splitted", splitted)
        if splitted[0] in [" ", ""]:
            splitted = splitted[1:]
        dataframe = pd.DataFrame(columns=["Index", "Text"])
        dataframe.loc[:, "Index"] = [splitted[i] for i in range(0, len(splitted), 2)]
        dataframe.loc[:, "Text"] = [splitted[i] for i in range(1, len(splitted) + 1, 2)]
        return dataframe
