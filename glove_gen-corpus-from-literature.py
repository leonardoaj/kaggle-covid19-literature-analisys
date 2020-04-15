from functions import get_filename_list, read_json

with open("literature_corpus.txt", "w") as corpus:
    for count, filename in enumerate(get_filename_list()):
        print(count, filename)

        paragraphs = read_json(filename, as_list=True)

        for paragraph in paragraphs:
            if paragraph:
                corpus.write(paragraph)

