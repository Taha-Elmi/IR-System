import pickle
import json
from hazm.lemmatizer import Lemmatizer
from index_creator import TermData, Posting, normalize, tokenize, process_verbs
from time import time


def load_data(file_name):
    db_file = open(file_name, 'rb')
    db = pickle.load(db_file)
    db_file.close()
    return db[file_name]


def get_scores(tokens, use_champion_list=True):
    doc_scores = {}
    if use_champion_list:
        index = load_data("champion_list_file")
    else:
        index = load_data("positional_index_file")

    for query_token in tokens:
        posting_list = index[query_token]
        for k, v in posting_list.postings.items():
            if k in doc_scores.keys():
                doc_scores[k] += v.tf_idf
            else:
                doc_scores[k] = v.tf_idf
    return doc_scores


def process_query(query, k=10, use_champion_list=True):
    lemmatizer = Lemmatizer()

    query = normalize(query)
    tokens = tokenize(query)
    tokens = process_verbs(tokens)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    doc_scores = get_scores(tokens, use_champion_list=use_champion_list)
    sorted_doc_scores = dict(sorted(doc_scores.items(), key=lambda item: item[1], reverse=True))
    return dict(list(sorted_doc_scores.items())[:k])


def show_results(results):
    counter = 1
    for doc_id, score in results.items():
        print(f"{counter}- {url_dataset[doc_id]}")
        print(title_dataset[doc_id])
        print()
        counter += 1


if __name__ == "__main__":
    with open('IR_data_news_12k.json', 'r', encoding='utf-8') as file:
        data = file.read()

    decoded_data = json.loads(data)

    url_dataset, title_dataset = {}, {}
    for index, data in decoded_data.items():
        url_dataset[index] = data['url']
        title_dataset[index] = data['title']

    while True:
        query = input("Enter query to search, or enter 'exit' to quit ")
        if query == "exit":
            break
        else:
            try:
                results = process_query(query, use_champion_list=False)
                show_results(results)
            except:
                print("no doc found.")

    # try:
    #     query = input()
    #
    #     t1 = time()
    #
    #     results = process_query(query, k=5, use_champion_list=True)
    #     show_results(results)
    #
    #     t2 = time()
    #     print("duration:", t2 - t1)
    # except:
    #     print("no doc found.")
