# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 08:53:41 2017

@author: jmanuel.navarro
"""
from collections import Counter
import numpy as np

#CURATE DATASET
g = open ('reviews.txt','r')
lines=g.readlines()
# Delete fial character
reviews = list(map(lambda x:x[:-1],lines))
g.close()

g = open ('labels.txt','r')
lines=g.readlines()
labels = list(map(lambda x:x[:-1],lines))
g.close()


def pretty_print_review_and_label(i):
    print(labels[i] + "\t:\t" + reviews[i][:80] + "...")
    
pretty_print_review_and_label(0)


#COUNT WORDS OF POSITIVE AND NEGATOVE REVIEWS AND ASSIGN A RATIO
positive_counts = Counter()
negative_counts = Counter()
total_counts = Counter()

#Loop over all words
for review, label in zip(reviews,labels):
#    print(label,'//',review,'\n')
    review_splitted = review.split(' ')
    total_counts.update(review_splitted)
    if ((label == 'positive') or (label == 'POSITIVE')):
        positive_counts.update(review_splitted)
    else:
        negative_counts.update(review_splitted)
        
# Examine the counts of the most common words in positive reviews
#print(positive_counts.most_common())

# Examine the counts of the most common words in negative reviews
#print(negative_counts.most_common())
        
        
# Create Counter object to store positive/negative ratios
# As usual words are those more often, what we really need is to find those
# words which are more often in positive reviews than in negative reviews
pos_neg_ratios = Counter()
# We only take into account words appearing more than "threshold" times
threshold = 100
for term, count in list(total_counts.most_common()):
    if count >= threshold:
        ratio = positive_counts[term] / (negative_counts[term] + 1)
        # High value -> positive // low value = negative // close to 1 -> irrelevant
        pos_neg_ratios[term] = ratio
    
print('amazing: ', pos_neg_ratios["amazing"])
print('terrible: ', pos_neg_ratios["terrible"])
print('the: ', pos_neg_ratios["the"])

# Convert ratios to log, to normalize them
for term,counter in list(pos_neg_ratios.most_common()):
    pos_neg_ratios[term]=np.log(counter)

print('amazing: ', pos_neg_ratios["amazing"])
print('terrible: ', pos_neg_ratios["terrible"])
print('the: ', pos_neg_ratios["the"])

#CONVERT REVIEW INTO A NUMERICAL INPUT
vocab = set(total_counts.keys())
print(len(vocab))

layer_0 = np.zeros((1,len(vocab)))

word2index={}
for index,word in enumerate(vocab):
    word2index[word]=index

#print(word2index)

#PROCESS EVERY REVIEW TO CONVERT IT INTO NUMERICAL VALUES
def update_input_layer(review):
    
    global layer_0
    # clear out previous state by resetting the layer to be all 0s
    layer_0 *= 0
    
    #Generate an (1,n) array for layer 0, where n is the index of
    #each word in word2index. We will update the array whith the times the word
    #appears in review
    review_splitted = review.split(' ')
    for word in review_splitted:
        layer_0[0][word2index[word]] += 1
        
#CONVERT LABELS INTO POSITIVE OR NEGATIVE
def get_target_for_label(label):
    """Convert a label to `0` or `1`.
    Args:
        label(string) - Either "POSITIVE" or "NEGATIVE".
    Returns:
        `0` or `1`.
    """
    # TODO: Your code here
    if ((label == 'positive') or (label == 'POSITIVE')):
        return '1'
    else:
        return '0'
    
    
    
update_input_layer(reviews[3])
print(layer_0)

print(labels[0])
print(get_target_for_label(labels[0]))
print(labels[1])
print(get_target_for_label(labels[1]))


import time
import sys

# Encapsulate our neural network in a class
class SentimentNetwork:
    def __init__(self, reviews, labels, hidden_nodes = 10, learning_rate = 0.1):
        """Create a SentimenNetwork with the given settings
        Args:
            reviews(list) - List of reviews used for training
            labels(list) - List of POSITIVE/NEGATIVE labels associated with the given reviews
            hidden_nodes(int) - Number of nodes to create in the hidden layer
            learning_rate(float) - Learning rate to use while training
        
        """
        # Assign a seed to our random number generator to ensure we get
        # reproducable results during development 
        np.random.seed(1)

        # process the reviews and their associated labels so that everything
        # is ready for training
        self.pre_process_data(reviews, labels)
        
        # Build the network to have the number of hidden nodes and the learning rate that
        # were passed into this initializer. Make the same number of input nodes as
        # there are vocabulary words and create a single output node.
        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels):
        
        review_vocab = set()
        # TODO: populate review_vocab with all of the words in the given reviews
        #       Remember to split reviews into individual words 
        #       using "split(' ')" instead of "split()".
        for review in reviews:
            for word in review.split(' '):
                review_vocab.add(word)
        # Convert the vocabulary set to a list so we can access words via indices
        self.review_vocab = list(review_vocab)
        
        label_vocab = set()
        # TODO: populate label_vocab with all of the words in the given labels.
        #       There is no need to split the labels because each one is a single word.
        for label in labels:
            label_vocab.add(label)
        # Convert the label vocabulary set to a list so we can access labels via indices
        self.label_vocab = list(label_vocab)
        
        # Store the sizes of the review and label vocabularies.
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
        
        # Create a dictionary of words in the vocabulary mapped to index positions
        self.word2index = {}
        # TODO: populate self.word2index with indices for all the words in self.review_vocab
        #       like you saw earlier in the notebook
        
        # Create a dictionary of labels mapped to index positions
        self.label2index = {}
        # TODO: do the same thing you did for self.word2index and self.review_vocab, 
        #       but for self.label2index and self.label_vocab instead
         
        
    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Store the number of nodes in input, hidden, and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Store the learning rate
        self.learning_rate = learning_rate

        # Initialize weights
        
        # TODO: initialize self.weights_0_1 as a matrix of zeros. These are the weights between
        #       the input layer and the hidden layer.
        self.weights_0_1 = None
        
        # TODO: initialize self.weights_1_2 as a matrix of random values. 
        #       These are the weights between the hidden layer and the output layer.
        self.weights_1_2 = None
        
        # TODO: Create the input layer, a two-dimensional matrix with shape 
        #       1 x input_nodes, with all values initialized to zero
        self.layer_0 = np.zeros((1,input_nodes))
    
        
    def update_input_layer(self,review):
        # TODO: You can copy most of the code you wrote for update_input_layer 
        #       earlier in this notebook. 
        #
        #       However, MAKE SURE YOU CHANGE ALL VARIABLES TO REFERENCE
        #       THE VERSIONS STORED IN THIS OBJECT, NOT THE GLOBAL OBJECTS.
        #       For example, replace "layer_0 *= 0" with "self.layer_0 *= 0"
        pass
                
    def get_target_for_label(self,label):
        # TODO: Copy the code you wrote for get_target_for_label 
        #       earlier in this notebook. 
        pass
        
    def sigmoid(self,x):
        # TODO: Return the result of calculating the sigmoid activation function
        #       shown in the lectures
        pass
    
    def sigmoid_output_2_derivative(self,output):
        # TODO: Return the derivative of the sigmoid activation function, 
        #       where "output" is the original output from the sigmoid fucntion 
        pass

    def train(self, training_reviews, training_labels):
        
        # make sure out we have a matching number of reviews and labels
        assert(len(training_reviews) == len(training_labels))
        
        # Keep track of correct predictions to display accuracy during training 
        correct_so_far = 0
        
        # Remember when we started for printing time statistics
        start = time.time()

        # loop through all the given reviews and run a forward and backward pass,
        # updating weights for every item
        for i in range(len(training_reviews)):
            
            # TODO: Get the next review and its correct label
            
            # TODO: Implement the forward pass through the network. 
            #       That means use the given review to update the input layer, 
            #       then calculate values for the hidden layer,
            #       and finally calculate the output layer.
            # 
            #       Do not use an activation function for the hidden layer,
            #       but use the sigmoid activation function for the output layer.
            
            # TODO: Implement the back propagation pass here. 
            #       That means calculate the error for the forward pass's prediction
            #       and update the weights in the network according to their
            #       contributions toward the error, as calculated via the
            #       gradient descent and back propagation algorithms you 
            #       learned in class.
            
            # TODO: Keep track of correct predictions. To determine if the prediction was
            #       correct, check that the absolute value of the output error 
            #       is less than 0.5. If so, add one to the correct_so_far count.
            
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the training process. 

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")
    
    def test(self, testing_reviews, testing_labels):
        """
        Attempts to predict the labels for the given testing_reviews,
        and uses the test_labels to calculate the accuracy of those predictions.
        """
        
        # keep track of how many correct predictions we make
        correct = 0

        # we'll time how many predictions per second we make
        start = time.time()

        # Loop through each of the given reviews and call run to predict
        # its label. 
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1
            
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the prediction process. 

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i+1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
    def run(self, review):
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review.
        """
        # TODO: Run a forward pass through the network, like you did in the
        #       "train" function. That means use the given review to 
        #       update the input layer, then calculate values for the hidden layer,
        #       and finally calculate the output layer.
        #
        #       Note: The review passed into this function for prediction 
        #             might come from anywhere, so you should convert it 
        #             to lower case prior to using it.
        
        # TODO: The output layer should now contain a prediction. 
        #       Return `POSITIVE` for predictions greater-than-or-equal-to `0.5`, 
        #       and `NEGATIVE` otherwise.
        pass

#TRAIN A SENTIMENT NETWORK THAT WILL BE TRAINED WILL ALL BUT THE LAST
#1000 REVIEWS (WHICH WILL BE USED FOR TESTING)
mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.1)