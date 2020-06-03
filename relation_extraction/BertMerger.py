import spacy
from spacy.tokens import Token
from spacy.matcher import Matcher


class BadTERMMerger(object):
    def __init__(self, nlp):
        # Register a new token extension to flag bad HTML
        Token.set_extension("bad_term", default=False, force=True)
        self.matcher = Matcher(nlp.vocab)
        self.matcher.add(
            "BAD_TERM",
            None,
            [{"LOWER": "paragraph"}, {"LIKE_NUM": True}],
            [{"LOWER": "paragraphs"}, {"LIKE_NUM": True}],
            [{"LOWER": "annex"}, {"LIKE_NUM": True}],
            [{"LOWER": "no"}, {"LIKE_NUM": True}],
            [{"LOWER": "1958"}, {"LOWER": "agreement"}]
        )

    def __call__(self, doc):
        # This method is invoked when the component is called on a Doc
        matches = self.matcher(doc)
        spans = []  # Collect the matched spans here
        for match_id, start, end in matches:
            spans.append(doc[start:end])
        with doc.retokenize() as retokenizer:
            for span in spans:
                retokenizer.merge(span)
                for token in span:
                    token._.bad_term = True  # Mark token as bad HTML
        return doc
