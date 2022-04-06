def read_tsv(path):
<<<<<<< HEAD
    with open(path, 'r') as file:
=======
    with open(path, 'r', encoding='utf-8') as file:
>>>>>>> origin/main
        data = []
        all_words = []
        all_tags = []

        words = []
        tags = []

        for idx, line in enumerate(file):
            if line == '\n':
                data.append([words, tags])
                all_words.extend(words)
                all_tags.extend(tags)
                words = []
                tags = []
                continue

            try:
                word, tag = line.strip().split('\t')
            except ValueError:
                raise Exception('Not enough data in line number %d.' % (idx + 1))
            words.append(word)
            tags.append(tag)

        if len(words) > 0 and len(tags) > 0:
            data.append([words, tags])
            all_words.extend(words)
            all_tags.extend(tags)

        return data, all_words, all_tags
