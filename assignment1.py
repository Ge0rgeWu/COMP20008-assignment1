# This is the file you will need to edit in order to complete assignment 1
# You may create additional functions, but all code must be contained within this file


# Some starting imports are provided, these will be accessible by all functions.
# You may need to import additional items
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import csv
import os
import re

# You should use these two variable to refer the location of the JSON data file and the folder containing the news articles.
# Under no circumstances should you hardcode a path to the folder on your computer (e.g. C:\Chris\Assignment\data\data.json) as this path will not exist on any machine but yours.
datafilepath = 'data/data.json'
articlespath = 'data/football'


def task1():
    with open(datafilepath, 'r') as loadf:
        data = json.load(loadf)
        team_codes = data['teams_codes']
        team_codes = sorted(team_codes)
        # print(team_codes)
    return team_codes


def task2():
    with open(datafilepath, 'r') as loadf:
        data = json.load(loadf)

        team_scores = data['clubs']
        csv_data = []
        for i in range(len(team_scores)):
            csv_data.append(
                (team_scores[i]['club_code'], team_scores[i]['goals_scored'], team_scores[i]['goals_conceded']))
        csv_data = sorted(csv_data, key=lambda t: t[0])
        head = ['team_code', 'goals_scored_by_team', 'goals_scored_against_team']
        with open('task2.csv', 'w', encoding='utf-8', newline='') as f:
            csvf = csv.writer(f)
            csvf.writerow(head)
            csvf.writerows(csv_data)
        # print(csv)
        # print(len(team_scores))
        # print((team_scores[1]['club_code']), team_scores[1]['goals_scored'], team_scores[1]['goals_conceded'] )
        # team_codes = sorted(team_codes)
    return


def task3():
    csv_data = []
    for filename in os.listdir(articlespath):
        data_path = os.path.join(articlespath, filename)
        # print(data_path)
        data_txt = open(data_path).read()
        # expression error '[0-9]*\-[0-9]*'
        data_tmp = re.findall('[0-9]+-[0-9]+', data_txt)
        # if filename == '014.txt':
        #     print(data_tmp)
        if len(data_tmp) == 0:
            csv_data.append((filename, 0))
        else:
            max_score = 0
            for tmp in data_tmp:
                score = (tmp.split('-'))
                assert len(score) == 2
                if int(score[0]) <= 99 and int(score[1]) <= 99 and int(score[0]) + int(score[1]) > max_score:
                    max_score = int(score[0]) + int(score[1])
                    # if filename == '014.txt':
                    #     print(max_score)
            csv_data.append((filename, max_score))
    head = ['filename', 'total_goals']
    csv_data = sorted(csv_data, key=lambda t: t[0])
    with open('task3.csv', 'w', encoding='utf-8', newline='') as f:
        csvf = csv.writer(f)
        csvf.writerow(head)
        csvf.writerows(csv_data)
        # print(score)
        # print(filename, data_tmp)
        # data = np.loadtxt(data_path, dtype=str)
        # print(data_txt)
    return


def task4():
    total_data = []
    for filename in os.listdir(articlespath):
        data_path = os.path.join(articlespath, filename)
        # print(data_path)
        data_txt = open(data_path).read()
        # expression error '[0-9]*\-[0-9]*'
        data_tmp = re.findall('[0-9]+-[0-9]+', data_txt)
        # no match
        if len(data_tmp) == 0:
            total_data.append((filename, 0))
        else:
            # judge the max
            max_score = 0
            for tmp in data_tmp:
                score = (tmp.split('-'))
                assert len(score) == 2
                # delete useless values
                if int(score[0]) <= 99 and int(score[1]) <= 99 and int(score[0]) + int(score[1]) > max_score:
                    max_score = int(score[0]) + int(score[1])
                    # if filename == '014.txt':
                    #     print(max_score)
            total_data.append((filename, max_score))
    # sort by names
    total_data = sorted(total_data, key=lambda t: t[0])
    total_scores = [data[1] for data in total_data]
    # print(total_scores)
    plt.title("total_scores")
    plt.boxplot(total_scores)
    plt.savefig('task4.png')
    return


def task5():
    csv_data = []
    with open(datafilepath, 'r') as loadf:
        data = json.load(loadf)
        club_names = data['participating_clubs']
        club_count = [0] * len(club_names)
        # print(club_count)
        # print(club_names)
        for filename in os.listdir(articlespath):
            data_path = os.path.join(articlespath, filename)
            # print(data_path)
            data_txt = open(data_path).read()
            for i in range(len(club_names)):
                # calculate
                if data_txt.find(club_names[i]) != -1:
                    club_count[i] = club_count[i] + 1
        for i in range(len(club_names)):
            csv_data.append((club_names[i], club_count[i]))
        head = ['club_name', 'number_of_mentions']
        csv_data = sorted(csv_data, key=lambda t: t[0])
        with open('task5.csv', 'w', encoding='utf-8', newline='') as f:
            csvf = csv.writer(f)
            csvf.writerow(head)
            csvf.writerows(csv_data)
        fig = plt.figure(figsize=(12, 8))
        plt.barh(club_names, club_count)
        plt.title('club count')
        # plt.xticks(rotation=-15)
        plt.savefig('task5.png')
    return


def task6():
    with open(datafilepath, 'r') as loadf:
        data = json.load(loadf)
        club_names = data['participating_clubs']
        club_single_count = [0] * len(club_names)
        club_two_count = np.array([[0] * 20] * len(club_names))
        sim = np.array([[0.0] * 20] * 20)
        # print(sim[0][1])
        # print(club_count)
        # print(club_names)
        for filename in os.listdir(articlespath):
            data_path = os.path.join(articlespath, filename)
            # print(data_path)
            data_txt = open(data_path).read()
            for i in range(len(club_names)):
                if data_txt.find(club_names[i]) != -1:
                    club_single_count[i] = club_single_count[i] + 1
                    for j in range(i + 1, len(club_names)):
                        if data_txt.find(club_names[j]) != -1:
                            club_two_count[i, j] = club_two_count[i, j] + 1
        # calculate the similarity
        for i in range(len(club_names)):
            for j in range(i + 1, len(club_names)):
                # print(i, j)
                sim[i, j] = float(club_two_count[i, j]) / float(club_single_count[i] + club_single_count[j] + 1e-8)
                sim[j, i] = sim[i, j]
        # print(sim)
        fig = plt.figure(figsize=(14, 12))
        df = pd.DataFrame(sim, club_names, club_names)
        ax = sns.heatmap(data=df)
        plt.xlabel('club names')
        plt.ylabel('club names')
        plt.savefig('task6.png')

    return


def task7():
    with open(datafilepath, 'r') as loadf:
        data = json.load(loadf)
        team_scores = data['clubs']
        scores = []
        for i in range(len(team_scores)):
            scores.append(team_scores[i]['goals_scored'])

        club_names = data['participating_clubs']
        club_count = [0] * len(club_names)

        for filename in os.listdir(articlespath):
            data_path = os.path.join(articlespath, filename)
            data_txt = open(data_path).read()
            for i in range(len(club_names)):
                if data_txt.find(club_names[i]) != -1:
                    club_count[i] = club_count[i] + 1
        plt.xlabel('scores')
        plt.ylabel('club count')
        plt.scatter(scores, club_count)
        plt.savefig('task7.png')
    return


def task8(filename):
    data_txt = open(filename).read()
    text = re.sub('[^a-zA-Z]+', ' ', data_txt)
    text = text.lower()
    text_list = text.split()
    return_text = text_list.copy()

    nltk = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
            "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they",
            "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
            "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
            "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at",
            "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above",
            "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", 'each', "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
            "can", "will", "just", "don", "should", "now"]
    for t in text_list:
        # print(t)
        if t in nltk:
            return_text.remove(t)
        elif len(t) == 1:
            return_text.remove(t)

    return return_text


def task9():
    filelists = os.listdir(articlespath)
    filelists.sort()
    # print(filelists)
    data_ = []
    data_result = []
    for i in range(len(filelists)):
        data_path = os.path.join(articlespath, filelists[i])
        data_.append(task8(data_path))
    for data in data_:
        tmp = ' '
        tmp = tmp.join(data)
        data_result.append(tmp)
    # print(data_result)
    data_tfidf = TfidfVectorizer().fit_transform(data_result)
    simi = []
    # print(cosine_similarity(data_tfidf[17], data_tfidf[26]))
    for i in range(len(data_result)):
        simi.append(cosine_similarity(data_tfidf[i], data_tfidf).flatten())

    simi_np = np.array(simi)
    for i in range(simi_np.shape[0]):
        simi_np[i, 0:i + 1] = 0
    # print(simi_np.shape)
    simi_np = simi_np.flatten()
    # print(simi_np[simi_np.argsort()[:-10:-1]])
    # print(simi_np.argsort()[:-10:-1])
    tmp = simi_np.argsort()[:-11:-1]
    csv_result = []
    for t in tmp:
        csv_result.append((filelists[t // 265], filelists[t % 265], simi[t // 265][t % 265]))
        # print(t // 265 + 1, t % 265 + 1, simi[t // 265][t % 265])
    # print(csv_result)
    head = ['article1', 'article2', 'similarity']
    with open('task9.csv', 'w', encoding='utf-8', newline='') as f:
        csvf = csv.writer(f)
        csvf.writerow(head)
        csvf.writerows(csv_result)
    # for i in range(sim.shape[0]):
    #     sim[i, i] = 0
    # print(sim)
    # print('cos', i, cosine_similarity(data_tfidf[i], data_tfidf))
    # print('cos', cosine_similarity(data_tfidf[0], data_tfidf))
    # for i in range(len(data_result)):
    #     for j in range(i+1, len(data_result)):
    #         vectorizer = TfidfVectorizer()
    #         d_i = vectorizer.fit_transform(data_result[i])
    #         print(d_i)
    #         d_j = vectorizer.transform(data_result[j])
    #         print(vectorizer.get_feature_names())
    #         cos_ = cosine_similarity(d_i, d_j)
    #         print(cos_.shape)
    # for i in range(len(filelists)):
    #     for j in range(i+1, len((filelists))):
    #         data_path_i = os.path.join(articlespath, filelists[i])
    #         data_path_j = os.path.join(articlespath, filelists[j])
    #         data_result_i = task8(data_path_i)
    #         data_result_j = task8(data_path_j)
    #
    #         vectorizer = TfidfVectorizer()
    #         d_i = vectorizer.fit_transform(data_result_i)
    #         print(d_i)
    #         d_j = vectorizer.transform(data_result_j)
    #         print(vectorizer.get_feature_names())
    # cos_ = cosine_similarity(d_i, d_j)
    # print(cos_.shape)

    # for filename in os.listdir(articlespath):
    #     data_path = os.path.join(articlespath, filename)
    #     data_result = task8(data_path)
    #     vectorizer = TfidfVectorizer()
    #     d = vectorizer.fit_transform(data_result)
    #     print(d)
    return
