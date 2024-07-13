import json
import re
import math
import pickle
from hazm.lemmatizer import Lemmatizer


# This class represents all information we want to save for every term_data in the index,
# including its frequency and its positional postings list.
class TermData:
    def __init__(self):
        self.frequency = 0  # frequency of the term_data in all documents
        self.postings = {}  # key: doc_id, value: a Posting object

    def __str__(self):
        result = f"frequency: {self.frequency}, postings:\n"
        for k, v in self.postings.items():
            result += f"index: {k}, posting: {str(v)}\n"
        return result


# This class represents a posting, that is the occurrence of a term_data in a specific document.
# It includes the frequency of the term_data in that one document, and its positions.
class Posting:
    def __init__(self):
        self.frequency = 0  # frequency of the term_data in the specified document
        self.positions = []  # list of positions
        self.tf_idf = 0  # tf_idf of the term_data in the specified doc will be saved here

    def __str__(self):
        return f"frequency: {self.frequency}, positions: {self.positions}"


def regex_replace(patterns, text):
    for pattern, repl in patterns:
        text = re.sub(pattern, repl, text)
    return text


def correct_spacing(text):
    punc_after, punc_before = r"\.:!،؛؟»\]\)\}", r"«\[\(\{"
    affix_spacing_patterns = [
        (r"([^ ]ه) ی ", r"\1‌ی "),  # fix ی space
        (r"(^| )(ن?می) ", r"\1\2‌"),  # put zwnj after می, نمی
        # put zwnj before تر, تری, ترین, گر, گری, ها, های
        (
            r"(?<=[^\n\d "
            + punc_after
            + punc_before
            + "]{2}) (تر(ین?)?|گری?|های?)(?=[ \n"
            + punc_after
            + punc_before
            + "]|$)",
            r"‌\1",
        ),
        # join ام, ایم, اش, اند, ای, اید, ات
        (
            r"([^ ]ه) (ا(م|یم|ش|ند|ی|ید|ت))(?=[ \n" + punc_after + "]|$)",
            r"\1‌\2",
        ),
        # شنبهها => شنبه‌ها
        ("(ه)(ها)", r"\1‌\2"),
    ]
    text = regex_replace(affix_spacing_patterns, text)
    return text


def replace_unicodes(text):
    replacements = [
        ("﷽", "بسم الله الرحمن الرحیم"),
        ("﷼", "ریال"),
        ("(ﷰ|ﷹ)", "صلی"),
        ("ﷲ", "الله"),
        ("ﷳ", "اکبر"),
        ("ﷴ", "محمد"),
        ("ﷵ", "صلعم"),
        ("ﷶ", "رسول"),
        ("ﷷ", "علیه"),
        ("ﷸ", "وسلم"),
        ("ﻵ|ﻶ|ﻷ|ﻸ|ﻹ|ﻺ|ﻻ|ﻼ", "لا"),
        ("آ", "ا"),
        ("ك", "ک"),
        ("ي", "ی"),
    ]

    text = regex_replace(replacements, text)
    return text


def remove_punctuations(text):
    diacritics_patterns = [
        # سه یونیکد اول: تنوین ها / سه یونیکد بعدی: اعراب / یونیکد بعدی: تشدید / یونیکد بعدی: ساکن / باقی: علائم نگارشی
        ("[\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652()<>؛،. »«}{؟!:\"+=/|%*'-]", ""),
    ]
    text = regex_replace(diacritics_patterns, text)
    return text


def translate_numbers(text):
    english_numbers = "0123456789%٠١٢٣٤٥٦٧٨٩"
    persian_numbers = "۰۱۲۳۴۵۶۷۸۹٪۰۱۲۳۴۵۶۷۸۹"

    translations = {ord(a): b for a, b in zip(english_numbers, persian_numbers)}
    return text.translate(translations)


def normalize(text):
    text = replace_unicodes(text)
    text = remove_punctuations(text)
    text = correct_spacing(text)
    text = translate_numbers(text)
    return text


def process_verbs(tokens):
    before_verbs = {
        "خواهم",
        "خواهی",
        "خواهد",
        "خواهیم",
        "خواهید",
        "خواهند",
        "نخواهم",
        "نخواهی",
        "نخواهد",
        "نخواهیم",
        "نخواهید",
        "نخواهند",
    }

    if len(tokens) == 1:
        return tokens

    result = [""]
    for token in reversed(tokens):
        if token in before_verbs:
            result[-1] = token + "_" + result[-1]
        else:
            result.append(token)
    return list(reversed(result[1:]))


def tokenize(text):
    pattern = re.compile(r'([؟!?]+|[\d.:]+|[:.،؛»\])}"«\[({/\\])')
    text = pattern.sub(r" \1 ", text.replace("\n", " ").replace("\t", " "))
    tokens = [word for word in text.split(" ") if word]
    return tokens


def tf_idf(term_data, doc_id):
    term_data_frequency_in_doc = positional_index[term_data].postings[doc_id].frequency
    n = len(content_dataset)
    df = len(positional_index[term_data].postings.keys())
    return (1 + math.log(term_data_frequency_in_doc, 10)) * math.log(n / df)


def save_data(file_name, file_data):
    db = {file_name: file_data}
    db_file = open(file_name, 'ab')
    pickle.dump(db, db_file)
    db_file.close()


if __name__ == "__main__":
    with open('IR_data_news_12k.json', 'r', encoding='utf-8') as file:
        data = file.read()

    decoded_data = json.loads(data)

    content_dataset, url_dataset, title_dataset = {}, {}, {}
    for index, data in decoded_data.items():
        content_dataset[index] = data['content']
        url_dataset[index] = data['url']
        title_dataset[index] = data['title']

    lemmatizer = Lemmatizer()
    positional_index = {}
    for doc_id, content in content_dataset.items():
        content = normalize(content)
        tokens = tokenize(content)
        tokens = process_verbs(tokens)
        for position, token in enumerate(tokens):
            token = lemmatizer.lemmatize(token)
            if token not in positional_index.keys():
                positional_index[token] = TermData()

            if doc_id not in positional_index[token].postings.keys():
                positional_index[token].postings[doc_id] = Posting()

            # we first update the term_data frequency
            positional_index[token].frequency += 1

            # then we update its posting
            positional_index[token].postings[doc_id].frequency += 1
            positional_index[token].postings[doc_id].positions.append(position)

    positional_index = dict(sorted(positional_index.items(), key=lambda item: item[1].frequency, reverse=True))
    for k, v in dict(list(positional_index.items())[:50]).items():
        print(f"{k}: {v.frequency}")
    positional_index = dict(list(positional_index.items())[50:])

    doc_vectors = {}
    for k, v in positional_index.items():
        for doc_id in v.postings.keys():
            doc_tf_idf = tf_idf(k, doc_id)
            v.postings[doc_id].tf_idf += doc_tf_idf
            if doc_id in doc_vectors.keys():
                doc_vectors[doc_id] += doc_tf_idf ** 2
            else:
                doc_vectors[doc_id] = doc_tf_idf ** 2

    for doc_id, sum_of_squares in doc_vectors.items():
        normalized_length = math.sqrt(sum_of_squares) if sum_of_squares > 0 else 1
        for term_data in positional_index.values():
            if doc_id in term_data.postings.keys():
                term_data.postings[doc_id].tf_idf /= normalized_length

    champion_list = {}
    for k, v in positional_index.items():
        postings = v.postings
        champion_postings = dict(sorted(postings.items(), key=lambda item: item[1].tf_idf, reverse=True))
        champion_list[k] = TermData()
        champion_list[k].frequency = v.frequency
        champion_list[k].postings = dict(list(champion_postings.items())[:20])

    save_data("positional_index_file", positional_index)
    save_data("champion_list_file", champion_list)
