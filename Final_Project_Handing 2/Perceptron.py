"""
The features that have been chosen are numerous, here are some of the new ones:
* The prefix and the suffix of the first and of the second word
* The first letter of the first word and the first letter of the second word
* Whether the length of the first and of the second word is < 3 or not
* The previous tags concatenated with the first word 
* and so on...
"""
from collections import defaultdict


class StructuredPerceptron(object):

    def __init__(self):
        """
        initialize model parameters
        """
        self.tags = set()
        self.feature_weights = defaultdict(lambda: defaultdict(float)) #feature_name -> tags -> weight
        self.weight_totals = defaultdict(lambda: defaultdict(float)) #feature_name -> tags -> weight
        self.timestamps = defaultdict(lambda: defaultdict(float)) #feature_name -> tags -> weight

        self.tag_dict = defaultdict(set) #word -> {tags}

        self.START = "__START__"
        self.END = "__END__"
        
        
    def normalize(self, word):
        """
        lowercase word, and replace numbers, user names, and URLs
        """

        return tuple([w.strip().lower() for w in word])

    
    def evaluate(self, data_instances):
        correct = 0
        total = 0
        for (words, tags) in data_instances:
            preds = self.predict(words)
            matches = sum(map(lambda x: int(x[0]==x[1]), zip(preds, tags)))
            correct += matches
            total += len(tags)
        return correct/total
        
    
    def fit(self, X_train, y_train, X_test, y_test, iterations=10, learning_rate=0.25, verbose=False):      
        # initialize tag dictionary for each word tuples and get tag set
        instances = [(X_train[i], y_train[i]) for i in range(len(X_train))]
        for (words, tags) in instances:
            self.tags.update(set(tags))

            for word, tag in zip(words, tags):
                self.tag_dict[self.normalize(word)].add(tag)
        
        dev_instances = [(X_test[i], y_test[i]) for i in range(len(X_test))]
        # iterate over data
        for iteration in range(1, iterations+1):
            correct = 0
            total = 0
            if verbose:
                print('Iteration {}'.format(iteration+1), file=sys.stderr, flush=True)
                print("*" * 15, file=sys.stderr, flush=True)

            random.shuffle(instances)
            for i, (words, tags) in enumerate(instances):
                if i > 0:
                    if i%1000==0:
                        print('%s'%i, file=sys.stderr, flush=True)
                    elif i%20==0:
                        print('.', file=sys.stderr, flush=True, end='')

                prediction = self.predict(words) #get prediction for every word (a list of prediction)
                global_gold_features, global_prediction_features = self.get_global_features(words, prediction, tags) #get the most frequent features for each prediction 
                #and get the most frequent feature for each tag 
                
                                    
                # update weight vector:
                # 1. move closer to true tag
                for tag, fids in global_gold_features.items(): # for every true tag and counter object
                    for fid, count in fids.items():            # for every features of that tag and the relative count
                        nr_iters_at_this_weight = iteration - self.timestamps[fid][tag]
                        self.weight_totals[fid][tag] += nr_iters_at_this_weight * self.feature_weights[fid][tag]
                        self.timestamps[fid][tag] = iteration
                        self.feature_weights[fid][tag] += learning_rate * count   #increase the importance of that tag for that specific feature

                # 2. move further from wrong tag
                for tag, fids in global_prediction_features.items():
                    for fid, count in fids.items():
                        nr_iters_at_this_weight = iteration - self.timestamps[fid][tag]
                        self.weight_totals[fid][tag] += nr_iters_at_this_weight * self.feature_weights[fid][tag]
                        self.timestamps[fid][tag] = iteration
                        self.feature_weights[fid][tag] -= learning_rate * count
                        
                # compute training accuracy for this iteration
                correct += sum([int(predicted_tag == true_tag) for predicted_tag, true_tag in zip(prediction, tags)])
                total += len(tags)

                # output examples
                if verbose and i%1000==0:
                    print("current word accuracy:{:.2f}".format(correct/total))
                    print(list(zip(words, 
                                   [self.normalize(word) for word in words], 
                                   tags, 
                                   prediction)), file=sys.stderr, flush=True)
            
            print('\t{} features'.format(len(self.feature_weights)), file=sys.stderr, flush=True)
            print('\tTraining accuracy: {:.2f}\n'.format(correct/total), file=sys.stderr, flush=True)
            print('\tDevelopment accuracy: {:.2f}\n'.format(self.evaluate(dev_instances)), file=sys.stderr, flush=True)
         
        # average weights
        for feature, tags in self.feature_weights.items():
            for tag in tags:
                total = self.weight_totals[feature][tag]
                total += (iterations - self.timestamps[feature][tag]) * self.feature_weights[feature][tag]
                averaged = round(total / float(iterations), 3)
                self.feature_weights[feature][tag] = averaged


    def get_features(self, word, previous_tag2, previous_tag, words, i):
        #word is now a list of tuples
        first_word = word[0]
        second_word = word[1]

        features = {
                    'PREFIX_1={}'.format(first_word[:3]),
                    'SUFFIX_1={}'.format(first_word[-3:]),
                    'PREFIX_2={}'.format(second_word[:3]),
                    'SUFFIX_2={}'.format(second_word[-3:]),
                    'LEN<=3_1={}'.format(len(first_word)<=3),
                    'LEN<=3_2={}'.format(len(second_word)<=3),
                    'FIRST_LETTER_1={}'.format(first_word[0]),
                    'FIRST_LETTER_2={}'.format(second_word[0]),
                    'WORD_1={}'.format(first_word),
                    'WORD_2={}'.format(second_word),
                    'NORM_WORD_1={}'.format(words[i][0]),
                    'NORM_WORD_2={}'.format(words[i][1]),
                    'PREV_WORD_1={}'.format(words[i-1][0]),
                    'PREV_WORD_2={}'.format(words[i-1][1]),
                    'PREV_WORD_PREFIX_1={}'.format(words[i-1][0][:3]),
                    'PREV_WORD_PREFIX_2={}'.format(words[i-1][1][:3]),
                    'PREV_WORD_SUFFIX_1={}'.format(words[i-1][0][-3:]),
                    'PREV_WORD+WORD={}+{}+{}'.format(words[i-1][1], words[i][0], words[i][1]),
                    'PREV_TAG={}'.format(previous_tag),                 # previous tag
                    'PREV_TAG2={}'.format(previous_tag2),                 # two-previous tag
                    'PREV_TAG_BIGRAM={}+{}'.format(previous_tag2, previous_tag),  # tag bigram
                    'PREV_TAG+WORD={}+{}'.format(previous_tag, first_word),            # word-tag combination
                    'PREV_TAG+BIGRAM={}+{}+{}'.format(previous_tag, first_word, second_word),            # word-tag combination
                    'PREV_TAG+PREFIX_1={}_{}'.format(previous_tag, first_word[:3]),        # prefix and tag
                    'PREV_TAG+SUFFIX_1={}_{}'.format(previous_tag, first_word[-3:]),        # suffix and tag
                    'PREV_TAG+PREFIX_2={}_{}'.format(previous_tag, second_word[:3]),        # prefix and tag
                    'PREV_TAG+SUFFIX_2={}_{}'.format(previous_tag, second_word[-3:]),        # suffix and tag
                    'WORD+TAG_BIGRAM_1={}+{}+{}'.format(first_word, previous_tag2, previous_tag),
                    'WORD+TAG_BIGRAM_2={}+{}+{}'.format(second_word, previous_tag2, previous_tag),
                    'PREV_WORD_SUFFIX_2={}'.format(words[i-1][1][-3:]),
                    'SUFFIX+2TAGS_1={}+{}+{}'.format(first_word[-3:], previous_tag2, previous_tag),
                    'PREFIX+2TAGS_1={}+{}+{}'.format(first_word[:3], previous_tag2, previous_tag),
                    'SUFFIX+2TAGS_2={}+{}+{}'.format(second_word[-3:], previous_tag2, previous_tag),
                    'PREFIX+2TAGS_2={}+{}+{}'.format(second_word[:3], previous_tag2, previous_tag),
                    'BIAS'
            }
        return features
    
    
    def get_global_features(self, words, predicted_tags, true_tags):
        '''
        sum up local features
        '''
        context = [self.START] + [self.normalize(word) for word in words] + [self.END, self.END]

        global_gold_features = defaultdict(lambda: Counter())
        global_prediction_features = defaultdict(lambda: Counter())

        prev_predicted_tag = self.START
        prev_predicted_tag2 = self.START
        
        for j, (word, predicted_tag, true_tag) in enumerate(zip(words, predicted_tags, true_tags)):      # for every (word, prediction, tag)

            prediction_features = self.get_features(word, prev_predicted_tag2, prev_predicted_tag, context, j+1)  # get the features of that word

            # update feature correlation with true and predicted tag
            global_prediction_features[predicted_tag].update(prediction_features) #construct a dictionary where each tag is associated with a counter of the frequent features associated with the word 
            global_gold_features[true_tag].update(prediction_features)


            prev_predicted_tag2 = prev_predicted_tag
            prev_predicted_tag = predicted_tag

        return global_gold_features, global_prediction_features
            
    
    def get_scores(self, features):
        """
        predict scores for each tag given features
        """
        scores = defaultdict(float)
        # add up the scores for each tag
        for feature in features:
            if feature not in self.feature_weights: #features weights is a dictionary that contains the score of all tags for a given feature
                continue
            weights = self.feature_weights[feature] #weights 
            for tag, weight in weights.items():
                scores[tag] += weight


        # return tag scores
        if not scores:
            # if there are no scores (e.g., first iteration),
            # simply return the first tag with score 1
            scores[list(self.tags)[0]] = 1
        
        return scores


    def predict(self, words):
        context = [self.START] + [self.normalize(word) for word in words] + [self.END, self.END]
                
        prev_predicted_tag = self.START
        prev_predicted_tag2 = self.START

        out = []
        # for evert word, collect the features using the context, and the previous tags, and you get the tag with the highest score and collect all the prediction in a list
        for j, word in enumerate(words):
            predicted_tag = list(self.tag_dict[context[j+1]])[0] if len(self.tag_dict[context[j+1]]) == 1 else None
            if not predicted_tag:
                # get the predicted features. NB: use j+1, since context is longer than words
                prediction_features = self.get_features(word, prev_predicted_tag2, prev_predicted_tag, context, j+1)
                scores = self.get_scores(prediction_features)

                # predict the current tag
                predicted_tag = max(scores, key=scores.get)

            prev_predicted_tag2 = prev_predicted_tag
            prev_predicted_tag = predicted_tag

            out.append(predicted_tag)

        return out