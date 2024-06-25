import json
import os


folder_path = "../data/dev/"
documents = []

number_of_documents = 0
number_of_lines = 0
number_of_words = 0

# loop through all the files in the folder
for file in os.listdir(folder_path):
    if not file.endswith(".dev"):
        continue

    with open(folder_path + file, "r") as f:
        # read the file
        document = []
        for line in f:
            line = line.strip()
            line = ' '.join(line.split())
            if line == "" and len(document) > 0:
                documents.append('\n'.join(document))
                document = []
                number_of_documents += 1
            else:
                document.append(line)
                number_of_lines += 1
                number_of_words += len(line.split())
        
        if len(document) > 0:
            documents.append('\n'.join(document))
            number_of_documents += 1


print(f"Number of documents: {number_of_documents:,}")
print(f"Number of lines: {number_of_lines:,}")
print(f"Number of words: {number_of_words:,}")


# save the documents
with open("../data/dev.jsonl", "w") as f:
    for document in documents:
        f.write(json.dumps(document, ensure_ascii=False) + "\n")
