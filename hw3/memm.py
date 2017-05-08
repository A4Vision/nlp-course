from data import *
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import time

def extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag):
    """
        Receives: a word's local information
        Rerutns: The word's features.
    """
    features = {}
    features['word'] = curr_word
    ### YOUR CODE HERE
    features['prev_tag'] = prev_tag
    features['word, prev_tag'] = curr_word + ' ' + prev_tag
    features['bigram'] = prev_word + ', ' + curr_word
    features['trigram'] = prevprev_word + ' ' + prev_word + ' ' + curr_word
    features['prev_bigram'] = prevprev_word + ' ' + prev_word
    features['next_bigram'] = curr_word + ' ' + next_word
    features['next_trigram'] = prev_word + ' ' + curr_word + ' ' + next_word
    features['prev_word, prevprev_tag'] = prev_word + ' ' + prevprev_tag
    features['next_word, prev_tag'] = next_word + ' ' + prev_tag
    features['tag_bigram'] = prev_tag + ' ' + prevprev_tag
    # for i in xrange(5):
    #     features['prefix #' + str(i) + ', tag'] = curr_word[:i] + ' ' + prev_tag
    #     features['suffix #' + str(i) + ', tag'] = curr_word[:i] + ' ' + prev_tag
    ### END YOUR CODE
    return features

def extract_features(sentence, i):
    curr_word = sentence[i][0]
    prev_token = sentence[i - 1] if i > 0 else ('<s>', '*')
    prevprev_token = sentence[i - 2] if i > 1 else ('<s>', '*')
    next_token = sentence[i + 1] if i < (len(sentence) - 1) else ('</s>', 'STOP')
    return extract_features_base(curr_word, next_token[0], prev_token[0], prevprev_token[0], prev_token[1], prevprev_token[1])

def vectorize_features(vec, features):
    """
        Receives: feature dictionary
        Rerutns: feature vector

        Note: use this function only if you chose to use the sklearn solver!
        This function prepares the feature vector for the sklearn solver,
        use it for tags prediction.
    """
    example = [features]
    return vec.transform(example)

def create_examples(sents):
    print "building examples"
    examples = []
    labels = []
    num_of_sents = 0
    for sent in sents:
        num_of_sents += 1
        for i in xrange(len(sent)):
            features = extract_features(sent, i)
            examples.append(features)
            labels.append(tagset[sent[i][1]])
    return examples, labels
    print "done"

def memm_greeedy(sent, logreg, vec):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Rerutns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    ### YOUR CODE HERE
    sent_with_tags = map(lambda i: [sent[i], ''], range(len(sent)))
    for i in xrange(len(sent)):
        features = extract_features(sent_with_tags, i)
        vec_features = vectorize_features(vec, features)
        predicted_tags[i] = index_to_tag_dict[logreg.predict(vec_features).item()]
        sent_with_tags[i][1] = predicted_tags[i]

    ### END YOUR CODE
    return predicted_tags

def memm_viterbi(sent, logreg, vec):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Rerutns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    ### YOUR CODE HERE
    # for i in xrange(len(sent)):
    #     print sent[i]
    #     features = extract_features(sent, i)
    #     vec_features = vectorize_features(vec, features)
    #     print logreg.predict(vec_features)
    # backpointers = [[-1] * len(vec.get_feature_names())] * len(sent)
    # scores = [[0] * len(vec.get_feature_names())] * len(sent)
    position = 0
    ### END YOUR CODE
    return predicted_tags

def memm_eval(test_data, logreg, vec):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm & greedy hmm
    """
    acc_viterbi, acc_greedy = 0.0, 0.0
    ### YOUR CODE HERE
    count_total_greedy = 0
    count_good_greedy = 0
    count_total_viterbi = 0
    count_good_viterbi = 0
    for sentence in test_data:
        words = [word for word, tag in sentence]
        greedy_tags = memm_greeedy(words, logreg, vec)
        print sentence
        print greedy_tags
        viterbi_tags = memm_viterbi(words, logreg, vec)
        for (word, tag), memm_tag in zip(sentence, greedy_tags):
            if tag == memm_tag:
                count_good_greedy += 1
            count_total_greedy += 1
        for (word, tag), memm_tag in zip(sentence, viterbi_tags):
            # if tag == index_to_tag_dict[memm_tag.item()]:
            #     count_good_viterbi += 1
            count_total_viterbi += 1
    assert count_total_viterbi == count_total_greedy
    acc_greedy = str(float(count_good_greedy) / count_total_greedy)
    acc_viterbi = str(float(count_good_viterbi) / count_total_viterbi)
    ### END YOUR CODE
    return acc_viterbi, acc_greedy

if __name__ == "__main__":
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")[:2000]
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")

    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    #The log-linear model training.
    #NOTE: this part of the code is just a suggestion! You can change it as you wish!
    curr_tag_index = 0
    tagset = {}
    for train_sent in train_sents:
        for token in train_sent:
            tag = token[1]
            if tag not in tagset:
                tagset[tag] = curr_tag_index
                curr_tag_index += 1
    index_to_tag_dict = invert_dict(tagset)
    vec = DictVectorizer()
    print "Create train examples"
    train_examples, train_labels = create_examples(train_sents)
    num_train_examples = len(train_examples)
    print "#example: " + str(num_train_examples)
    print "Done"

    print "Create dev examples"
    dev_examples, dev_labels = create_examples(dev_sents)
    num_dev_examples = len(dev_examples)
    print "#example: " + str(num_dev_examples)
    print "Done"

    all_examples = train_examples
    all_examples.extend(dev_examples)

    print "Vectorize examples"
    all_examples_vectorized = vec.fit_transform(all_examples)
    train_examples_vectorized = all_examples_vectorized[:num_train_examples]
    dev_examples_vectorized = all_examples_vectorized[num_train_examples:]
    print "Done"

    logreg = linear_model.LogisticRegression(
        multi_class='multinomial', max_iter=128, solver='lbfgs', C=100000, verbose=1)
    print "Fitting..."
    start = time.time()
    logreg.fit(train_examples_vectorized, train_labels)
    end = time.time()
    print "done, " + str(end - start) + " sec"
    #End of log linear model training

    acc_viterbi, acc_greedy = memm_eval(dev_sents, logreg, vec)
    print "dev: acc memm greedy: " + acc_greedy
    print "dev: acc memm viterbi: " + acc_viterbi
    if os.path.exists('Penn_Treebank/test.gold.conll'):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        acc_viterbi, acc_greedy = memm_eval(test_sents, logreg, vec)
        print "test: acc memmm greedy: " + acc_greedy
        print "test: acc memmm viterbi: " + acc_viterbi