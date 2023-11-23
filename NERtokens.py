from cltk.tag import ner
import time as t
import glob


def create_ner(text):
    # make list of NER for each row
    ner_list = ner.tag_ner("grc", text)
    # Filter elements with 'Entity'
    entity_list = [element[0] for element in ner_list if len(
        element) == 2 and element[1] == 'Entity']
    return entity_list


def write_ALLner(flat_list, folder):
    # sort alphabetically
    word_count_ordered = sorted(set(flat_list))

    with open(f'NER_tokens{folder}.txt', 'w', newline='') as file:
        for el in word_count_ordered:
            file.write(el + '\n')

    return


def NER_list(folder):

    print('making list of NER...')
    start_time = t.time()
    ListNER = []
    for file in glob.glob(folder + '/*.txt'):
        f = open(file, 'r').read()
        ListNER.append(create_ner(f))

    end_time = t.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time} seconds")
    # exec time: 33 min ca
    flat_list = [item for sublist in ListNER for item in sublist]
    print("length: ", len(flat_list))
    print("lenght SET: ", len(set(flat_list)))
    # create CSV doc
    write_ALLner(flat_list, folder)

    return


if __name__ == "__main__":
    f = 'Rcorpus'
    NER_list(f)
