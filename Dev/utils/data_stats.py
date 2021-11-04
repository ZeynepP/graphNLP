import json
import random
import re  # import the regular expressions library; will be used to strip punctuation
from collections import defaultdict

import nltk
from nltk.corpus import stopwords  # import stopwords from nltk corpus

def get_tokens(raw, encoding='utf8'):
    '''get the nltk tokens from a text'''
    tokens = nltk.word_tokenize(raw, language="french")  # tokenize the raw UTF-8 text
    return tokens


def get_nltk_text(raw: object, encoding: object = 'utf8') -> object:
    '''create an nltk text using the passed argument (raw) after filtering out the commas'''
    # turn the raw text into an nltk text object
    no_commas = re.sub(r'[.|,|\']', ' ', raw)  # filter out all the commas, periods, and appostrophes using regex
    tokens = nltk.word_tokenize(no_commas, language="french")  # generate a list of tokens from the raw text
    text = nltk.Text(tokens, encoding)  # create a nltk text from those tokens
    return text


def get_stopswords(type="veronis"):
    '''returns the veronis stopwords in unicode, or if any other value is passed, it returns the default nltk french stopwords'''

    raw_stopword_list = ["Ap.", "Apr.", "GHz", "MHz", "USD", "a", "afin", "ah", "ai", "aie", "aient", "aies", "ait",
                             "alors", "après", "as", "attendu", "au", "au-delà", "au-devant", "aucun", "aucune",
                             "audit", "auprès", "auquel", "aura", "aurai", "auraient", "aurais", "aurait", "auras",
                             "aurez", "auriez", "aurions", "aurons", "auront", "aussi", "autour", "autre", "autres",
                             "autrui", "aux", "auxdites", "auxdits", "auxquelles", "auxquels", "avaient", "avais",
                             "avait", "avant", "avec", "avez", "aviez", "avions", "avons", "ayant", "ayez", "ayons",
                             "b", "bah", "banco", "ben", "bien", "bé", "c", "c'", "c'est", "c'était", "car", "ce",
                             "ceci", "cela", "celle", "celle-ci", "celle-là", "celles", "celles-ci", "celles-là",
                             "celui", "celui-ci", "celui-là", "celà", "cent", "cents", "cependant", "certain",
                             "certaine", "certaines", "certains", "ces", "cet", "cette", "ceux", "ceux-ci", "ceux-là",
                             "cf.", "cg", "cgr", "chacun", "chacune", "chaque", "chez", "ci", "cinq", "cinquante",
                             "cinquante-cinq", "cinquante-deux", "cinquante-et-un", "cinquante-huit", "cinquante-neuf",
                             "cinquante-quatre", "cinquante-sept", "cinquante-six", "cinquante-trois", "cl", "cm",
                             "cm²", "comme", "contre", "d", "d'", "d'après", "d'un", "d'une", "dans", "de", "depuis",
                             "derrière", "des", "desdites", "desdits", "desquelles", "desquels", "deux", "devant",
                             "devers", "dg", "différentes", "différents", "divers", "diverses", "dix", "dix-huit",
                             "dix-neuf", "dix-sept", "dl", "dm", "donc", "dont", "douze", "du", "dudit", "duquel",
                             "durant", "dès", "déjà", "e", "eh", "elle", "elles", "en", "en-dehors", "encore", "enfin",
                             "entre", "envers", "es", "est", "et", "eu", "eue", "eues", "euh", "eurent", "eus", "eusse",
                             "eussent", "eusses", "eussiez", "eussions", "eut", "eux", "eûmes", "eût", "eûtes", "f",
                             "fait", "fi", "flac", "fors", "furent", "fus", "fusse", "fussent", "fusses", "fussiez",
                             "fussions", "fut", "fûmes", "fût", "fûtes", "g", "gr", "h", "ha", "han", "hein", "hem",
                             "heu", "hg", "hl", "hm", "hm³", "holà", "hop", "hormis", "hors", "huit", "hum", "hé", "i",
                             "ici", "il", "ils", "j", "j'", "j'ai", "j'avais", "j'étais", "jamais", "je", "jusqu'",
                             "jusqu'au", "jusqu'aux", "jusqu'à", "jusque", "k", "kg", "km", "km²", "l", "l'", "l'autre",
                             "l'on", "l'un", "l'une", "la", "laquelle", "le", "lequel", "les", "lesquelles", "lesquels",
                             "leur", "leurs", "lez", "lors", "lorsqu'", "lorsque", "lui", "lès", "m", "m'", "ma",
                             "maint", "mainte", "maintes", "maints", "mais", "malgré", "me", "mes", "mg", "mgr", "mil",
                             "mille", "milliards", "millions", "ml", "mm", "mm²", "moi", "moins", "mon", "moyennant",
                             "mt", "m²", "m³", "même", "mêmes", "n", "n'avait", "n'y", "ne", "neuf", "ni", "non",
                             "nonante", "nonobstant", "nos", "notre", "nous", "nul", "nulle", "nº", "néanmoins", "o",
                             "octante", "oh", "on", "ont", "onze", "or", "ou", "outre", "où", "p", "par", "par-delà",
                             "parbleu", "parce", "parmi", "pas", "passé", "pendant", "personne", "peu", "plus",
                             "plus_d'un", "plus_d'une", "plusieurs", "pour", "pourquoi", "pourtant", "pourvu", "près",
                             "puisqu'", "puisque", "q", "qu", "qu'", "qu'elle", "qu'elles", "qu'il", "qu'ils", "qu'on",
                             "quand", "quant", "quarante", "quarante-cinq", "quarante-deux", "quarante-et-un",
                             "quarante-huit", "quarante-neuf", "quarante-quatre", "quarante-sept", "quarante-six",
                             "quarante-trois", "quatorze", "quatre", "quatre-vingt", "quatre-vingt-cinq",
                             "quatre-vingt-deux", "quatre-vingt-dix", "quatre-vingt-dix-huit", "quatre-vingt-dix-neuf",
                             "quatre-vingt-dix-sept", "quatre-vingt-douze", "quatre-vingt-huit", "quatre-vingt-neuf",
                             "quatre-vingt-onze", "quatre-vingt-quatorze", "quatre-vingt-quatre", "quatre-vingt-quinze",
                             "quatre-vingt-seize", "quatre-vingt-sept", "quatre-vingt-six", "quatre-vingt-treize",
                             "quatre-vingt-trois", "quatre-vingt-un", "quatre-vingt-une", "quatre-vingts", "que",
                             "quel", "quelle", "quelles", "quelqu'", "quelqu'un", "quelqu'une", "quelque", "quelques",
                             "quelques-unes", "quelques-uns", "quels", "qui", "quiconque", "quinze", "quoi", "quoiqu'",
                             "quoique", "r", "revoici", "revoilà", "rien", "s", "s'", "sa", "sans", "sauf", "se",
                             "seize", "selon", "sept", "septante", "sera", "serai", "seraient", "serais", "serait",
                             "seras", "serez", "seriez", "serions", "serons", "seront", "ses", "si", "sinon", "six",
                             "soi", "soient", "sois", "soit", "soixante", "soixante-cinq", "soixante-deux",
                             "soixante-dix", "soixante-dix-huit", "soixante-dix-neuf", "soixante-dix-sept",
                             "soixante-douze", "soixante-et-onze", "soixante-et-un", "soixante-et-une", "soixante-huit",
                             "soixante-neuf", "soixante-quatorze", "soixante-quatre", "soixante-quinze",
                             "soixante-seize", "soixante-sept", "soixante-six", "soixante-treize", "soixante-trois",
                             "sommes", "son", "sont", "sous", "soyez", "soyons", "suis", "suite", "sur", "sus", "t",
                             "t'", "ta", "tacatac", "tandis", "te", "tel", "telle", "telles", "tels", "tes", "toi",
                             "ton", "toujours", "tous", "tout", "toute", "toutefois", "toutes", "treize", "trente",
                             "trente-cinq", "trente-deux", "trente-et-un", "trente-huit", "trente-neuf",
                             "trente-quatre", "trente-sept", "trente-six", "trente-trois", "trois", "très", "tu", "u",
                             "un", "une", "unes", "uns", "v", "vers", "via", "vingt", "vingt-cinq", "vingt-deux",
                             "vingt-huit", "vingt-neuf", "vingt-quatre", "vingt-sept", "vingt-six", "vingt-trois",
                             "vis-à-vis", "voici", "voilà", "vos", "votre", "vous", "w", "x", "y", "z", "zéro", "à",
                             "ç'", "ça", "ès", "étaient", "étais", "était", "étant", "étiez", "étions", "été", "étée",
                             "étées", "étés", "êtes", "être", "ô"]

    # get French stopwords from the nltk kit
    raw_stopword_list += stopwords.words('french')  # create a list of all French stopwords

    return raw_stopword_list


def filter_stopwords(text, stopword_list):
    '''normalizes the words by turning them all lowercase and then filters out the stopwords'''
    words = [w.lower() for w in text]  # normalize the words in the text, making them all lowercase
    # filtering stopwords
    filtered_words = []  # declare an empty list to hold our filtered words
    for word in words:  # iterate over all words from the text
        if word not in stopword_list and word.isalpha() and len(
                word) > 1:  # only add words that are not in the French stopwords list, are alphabetic, and are more than 1 character
            filtered_words.append(word)  # add word to filter_words list if it meets the above conditions

    return filtered_words



# <codecell>

def get_class_dist(path):
    counter = defaultdict(int)
    multiple = 0
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            try:
                topics = obj["topics"]
                if len(topics) > 0:
                    for t in topics:
                        counter[t["label"]] += 1

            except :
                print("Empty topics for ", obj["id"])
    print(counter)
    return counter


def get_data_byclass(path):
    data_dict = dict()
    with open(path) as f:
        for line in f:
            obj = json.loads(line)

            topics = obj["topics"]
            if len(topics) > 0:
                for t in topics:
                    if not t["label"] in data_dict :
                        data_dict[t["label"]] = [obj]
                    else:
                        data_dict[t["label"]].append(obj)

    return data_dict

# will split data to test and train
def create_raw(path, out, percentage):
    french_stopwords = get_stopswords()

    data_dict = get_data_byclass(path)
    for topic in data_dict:
        l = data_dict[topic]
        random.shuffle(l)
        size = len(l)
        train_size = int(size * percentage)
        print(topic, train_size)
        with open(out+"test.txt", 'a+') as f:
            for obj in l[:train_size]:
                text = get_nltk_text(obj["document"])
                filtered_words = filter_stopwords(text, french_stopwords)
                doc = ' '.join(filtered_words)#nlp(' '.join(filtered_words))

                f.write(doc.text)
                f.write("\t")
                f.write(topic )
                f.write("\n")

        with open(out+"train.txt", 'a+') as f:
            for obj in l[train_size:]:
                text = get_nltk_text(obj["document"])
                filtered_words = filter_stopwords(text, french_stopwords)
                doc = ' '.join(filtered_words)#nlp(' '.join(filtered_words))

                f.write(doc.text)
                f.write("\t")
                f.write(topic)
                f.write("\n")
import csv
def to_csv_check(path, out):

    with open(path, encoding="utf-8") as f:
        to_csv = []
        csvfile = open(out, 'a')
        writer = csv.writer(csvfile, delimiter='\t')
        for line in f:
            obj = json.loads(line)
            doc_time = obj["docTime"]
            title = obj["title"]
            end_time = obj["endTime"]
            topics = obj["topics"]
            if len(topics) > 0:
                for t in topics:
                    ina_label = t["label"]

            channel = obj["media"]

            writer.writerow([channel, ina_label, doc_time, end_time, title])
        csvfile.close()


def to_text_label(path, out):

    with open(path, encoding="utf-8") as f:
        with open(out, 'a') as fout:
            for line in f:
                obj = json.loads(line)
                doc_id = str(obj["id"])
                doc = obj["document"].replace("\t"," ")
                topics = obj["topics"]
                if len(topics) > 0:
                    for t in topics:
                        ina_label = t["label"]


                fout.write("\t".join([doc_id, ina_label, doc]))
                fout.write("\n")




path = "./data/ina-clean/spacy_subjects2019.json"
# create_raw(path, "./data/ina-clean/raw/",0.20)
to_text_label(path,  "./data/ina-clean/jt.txt")

#{'Société': 3282, 'Justice': 804, 'Faits divers': 1283, 'Environnement': 729, 'Politique France': 1848, 'Economie': 1406, 'International': 2209, 'Catastrophes': 1800, 'Culture-loisirs': 467, 'Histoire-hommages': 494, 'Santé': 565, 'Education': 304, 'Sport': 546, 'Sciences et techniques': 147})