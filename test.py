import sys #used for argv[]

import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt


#from custom_dataset_from_file import CustomDatasetFromFile
# Gensim
import gensim
##import gensim.corpora as corpora
##from gensim.utils import simple_preprocess
##from gensim.models import CoherenceModel
##from gensim.models import Word2Vec
import gensim.downloader as api

import nltk
from nltk.corpus import stopwords
import spacy
import copy


#My code
import json
from torch.utils.data.dataset import Dataset  # For custom datasets

from itertools import combinations
import operator



#import torch
from torch import nn
from tqdm.auto import tqdm
#from torchvision import transforms
#from torchvision.datasets import MNIST # Training dataset
#from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import timeit



#torch.manual_seed(0) # Set for testing purposes, please do not change!

#import numpy as np
#Load pretrained_model
#pretrained_model = api.load('glove-twitter-50') 
#pretrained_model2 = api.load('glove-twitter-50') 

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.fc1 = nn.Linear(50, 40)
        self.fc2 = nn.Linear(40, 30)
        self.fc3 = nn.Linear(30, 20)
        self.fc4 = nn.Linear(20, 15)
        self.fc5 = nn.Linear(15,10)
        
        #self.fc5_mu = nn.Linear(15,10)
        #self.fc5_sig = nn.Linear(15,10)
        
        self.fc6_mu = nn.Linear(10, 5)
        self.fc6_sig = nn.Linear(10, 5)
        
        self.fc7 = nn.Linear(5, 10)
        self.fc8 = nn.Linear(10, 15)
        self.fc9 = nn.Linear(15, 20)
        self.fc10 = nn.Linear(20, 30)
        self.fc11 = nn.Linear(30, 40)
        self.fc12 = nn.Linear(40, 50)

    def encode(self,x):
        
        z1 = self.fc1(x)  #50D --> 40D
        a1 = F.leaky_relu(z1, 0.1)#, inplace = True)
        
        z2 = self.fc2(a1) #40D --> 30D
        a2 = F.leaky_relu(z2, 0.1)#, inplace = True)
        
        z3 = self.fc3(a2) #30D --> 20D
        a3 = F.leaky_relu(z3, 0.1)#, inplace = True)
        
        
        z4 = self.fc4(a3) #20D --> 15D
        a4 = F.leaky_relu(z4, 0.1)#, inplace = True)
        
        z5 = self.fc5(a4) #15D --> 10D
        a5 = F.leaky_relu(z5, 0.1)#, inplace = True)
        
        z6_mu = self.fc6_mu(a5)       #10D --> 5D
        z6_logvar = self.fc6_sig(a5)  #No activation function at the final layer of the encoder
        
        
        return z6_mu, z6_logvar
  
    def decode(self,z):
        
        
        z1 = self.fc7(z)  #5D --> 10D
        a1 = F.leaky_relu(z1, 0.1)#, inplace = True)
        
        z2 = self.fc8(a1) #10D --> 15D
        a2 = F.leaky_relu(z2, 0.1)#, inplace = True)
        
        z3 = self.fc9(a2) #15D --> 20D
        a3 = F.leaky_relu(z3, 0.1)#, inplace = True)
        
        z4 = self.fc10(a3) #20D --> 30D
        a4 = F.leaky_relu(z4, 0.1)#, inplace = True)
        
        z5 = self.fc11(a4) #30D --> 40D
        a5 = F.leaky_relu(z5, 0.1)#, inplace = True)
        
        z6 = self.fc12(a5) #40D --> 50D, no activation function at the final layer
        
        return z6
        #a3 = F.relu(self.fc3(z))
        #return torch.sigmoid(self.fc4(a3))
  
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
  
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, BETA = 1.0):
    
    BCE = F.mse_loss(recon_x, x, reduction='sum') #Default value: size_average=None, reduce=None, reduction='mean')    
    
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE, BETA * KLD

# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_generator_block
def get_generator_block(input_dim, output_dim):
    '''
    Function for returning a block of the generator's neural network
    given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a generator neural network layer, with a linear transformation 
          followed by a batch normalization and then a relu activation
    '''
    return nn.Sequential(
        # Hint: Replace all of the "None" with the appropriate dimensions.
        # The documentation may be useful if you're less familiar with PyTorch:
        # https://pytorch.org/docs/stable/nn.html.
        #### START CODE HERE ####
        nn.Linear(input_dim, output_dim),
        #nn.BatchNorm1d(output_dim),
        #nn.ReLU(inplace=True),
        nn.LeakyReLU(0.1), #inplace=True),
        #### END CODE HERE ####
    )

# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: Generator
class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
          (MNIST images are 28 x 28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, z_dim=10, im_dim=50, hidden_dim=10):
        super(Generator, self).__init__()
        # Build the neural network
        self.gen = nn.Sequential(
            get_generator_block(z_dim, 15),                         #10 --> 15
            get_generator_block(15, hidden_dim + 10),               #15 --> 20
            get_generator_block(hidden_dim + 10, hidden_dim + 20),  #20 --> 30
            get_generator_block(hidden_dim + 20, hidden_dim + 30),  #30 --> 40
            #get_generator_block(hidden_dim + 10, hidden_dim + 15), 
            # There is a dropdown with hints if you need them! 
            #### START CODE HERE ####
            nn.Linear(hidden_dim + 30, im_dim),                     #40 --> 50
            #nn.Sigmoid()            
            #### END CODE HERE ####
        )
    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return self.gen(noise)
    
    # Needed for grading
    def get_gen(self):
        '''
        Returns:
            the sequential model
        '''
        return self.gen

# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_noise
def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim),
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    '''
    # NOTE: To use this on GPU with device='cuda', make sure to pass the device 
    # argument to the function you use to generate the noise.
    #### START CODE HERE ####
    return torch.randn((n_samples, z_dim), device=device)
    #### END CODE HERE ####

# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_discriminator_block
def get_discriminator_block(input_dim, output_dim):
    '''
    Discriminator Block
    Function for returning a neural network of the discriminator given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a discriminator neural network layer, with a linear transformation 
          followed by an nn.LeakyReLU activation with negative slope of 0.2 
          (https://pytorch.org/docs/master/generated/torch.nn.LeakyReLU.html)
    '''
    return nn.Sequential(
        #### START CODE HERE ####
        nn.Linear(input_dim, output_dim),
        #nn.LeakyReLU(negative_slope=0.1, inplace=True)
        nn.LeakyReLU(negative_slope=0.1)
        #### END CODE HERE ####
    )


# UNQ_C5 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: Discriminator
class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
            (MNIST images are 28x28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_dim=50, hidden_dim=10):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim + 30),
            get_discriminator_block(hidden_dim + 30, hidden_dim + 20),
            get_discriminator_block(hidden_dim + 20, hidden_dim + 10),


            get_discriminator_block(hidden_dim + 10, 15),
            get_discriminator_block(hidden_dim + 5, hidden_dim),
            get_discriminator_block(hidden_dim, 5),

            # Hint: You want to transform the final output into a single value,
            #       so add one more linear map.
            #### START CODE HERE ####
            nn.Linear(5, 1)
            #### END CODE HERE ####
        )

    def forward(self, image):
        '''
        Function for completing a forward pass of the discriminator: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_dim)
        '''
        return self.disc(image)
    
    # Needed for grading
    def get_disc(self):
        '''
        Returns:
            the sequential model
        '''
        return self.disc
    
# UNQ_C6 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_disc_loss
def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        real: a batch of real images
        num_images: the number of images the generator should produce, 
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        disc_loss: a torch scalar loss value for the current batch
    '''
    #     These are the steps you will need to complete:
    #       1) Create noise vectors and generate a batch (num_images) of fake images. 
    #            Make sure to pass the device argument to the noise.
    #       2) Get the discriminator's prediction of the fake image 
    #            and calculate the loss. Don't forget to detach the generator!
    #            (Remember the loss function you set earlier -- criterion. You need a 
    #            'ground truth' tensor in order to calculate the loss. 
    #            For example, a ground truth tensor for a fake image is all zeros.)
    #       3) Get the discriminator's prediction of the real image and calculate the loss.
    #       4) Calculate the discriminator's loss by averaging the real and fake loss
    #            and set it to disc_loss.
    #     Note: Please do not use concatenation in your solution. The tests are being updated to 
    #           support this, but for now, average the two losses as described in step (4).
    #     *Important*: You should NOT write your own loss function here - use criterion(pred, true)!
    #### START CODE HERE ####
    noise = get_noise(num_images, z_dim, device)
    fake_loss = criterion(disc(gen(noise).detach()), torch.zeros(num_images, 1).to(device))
    real_loss = criterion(disc(real), torch.ones(num_images, 1).to(device))
    disc_loss = (fake_loss + real_loss) / 2
    #### END CODE HERE ####
    return disc_loss



# UNQ_C7 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_gen_loss
def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        num_images: the number of images the generator should produce, 
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        gen_loss: a torch scalar loss value for the current batch
    '''
    #     These are the steps you will need to complete:
    #       1) Create noise vectors and generate a batch of fake images. 
    #           Remember to pass the device argument to the get_noise function.
    #       2) Get the discriminator's prediction of the fake image.
    #       3) Calculate the generator's loss. Remember the generator wants
    #          the discriminator to think that its fake images are real
    #     *Important*: You should NOT write your own loss function here - use criterion(pred, true)!

    #### START CODE HERE ####
    noise = get_noise(num_images, z_dim, device)
    gen_loss = criterion(disc(gen(noise)), torch.ones(num_images, 1).to(device))
    #### END CODE HERE ####
    return gen_loss

class CustomDatasetFromFile(Dataset):
    def __init__(self, user_corpus, user_hashtags, pretrained_model, test_ratio = 0.02, argmentation = 0):
        """
        A dataset example where the class is embedded in the file names
        This data example also does not use any torch transforms

        Args:
            folder_path (string): path to image folder
        """
        print("Init running")
        self.user_hashtags = user_hashtags
        
        #Split user corpus into train and test corpus based on the ratio
        self.length = int(test_ratio * len(user_corpus))
        self.train_corpus = user_corpus[self.length : ]
        self.test_corpus = user_corpus[ : self.length]
        
        print("len(train_test_ratio) = ", self.length)
        print("len(user_corpus) = ", len(user_corpus))
        print("len(train_corpus) = ", len(self.train_corpus))
        print("len(test_corpus) = ", len(self.test_corpus))
        
        #From corpus to token_list
        #if isTrain:
        self.train_user_token_lst = self.corpus_to_token_list(self.train_corpus, pretrained_model)
        self.test_user_token_lst = self.corpus_to_token_list(self.test_corpus, pretrained_model)

        print("len(train_user_token_lst) = ", len(self.train_user_token_lst))
        print("len(test_user_token_lst) = ", len(self.test_user_token_lst))
        
        #Data argumentation
        if argmentation > 0:
            self.train_user_token_lst = self.data_augmentation(self.train_user_token_lst, argmentation)
            print("After argumentation len(train_user_token_lst) = ", len(self.train_user_token_lst))


        # Calculate len
        self.data_len = len(self.train_user_token_lst)
        #self.data_len = len(self.image_list)

    def __getitem__(self, index):
         
        #Read word by word
        #Each element of user_token_lst is a tuple of two parts: ('word', [vector in array format])
        label = self.train_user_token_lst[index][0] #extract the first part of the tuple for 'word'
        vector = self.train_user_token_lst[index][1]#extract the second part of the tuple for vector (array) of word
                
        # Transform word vector into array type
        vector_as_np = np.asarray(vector)

        # Transform image to tensor, change data type into float
        vector_as_tensor = torch.from_numpy(vector_as_np).float()
        
        return (vector_as_tensor, label)

    def __len__(self):
        return self.data_len
    
    def get_user_token_lst(self):        
        return self.train_user_token_lst, self.test_user_token_lst

    def get_train_test_corpus(self):
        return self.train_corpus, self.test_corpus
    
    def get_user_hashtags(self):
        return self.user_hashtags
    
    def corpus_to_token_list(self, corpus, pretrained_model):
        token_lst = []
        
        for sentence in corpus:
            for word in sentence:
                if (word in pretrained_model.vocab): #this will count the duplicate words
                    token_lst.append((word, pretrained_model[word]))
        
        return token_lst
    
    def data_augmentation(self, token_lst, num):
        new_lst = []
        
        for i in range (num):
            new_lst.extend(token_lst)
            
        return new_lst
    
def plot_graph_(mean_generator_loss_lst, mean_discriminator_loss_lst):
    plt.clf()
    
    print("len(mean_generator_loss_lst) = ", len(mean_generator_loss_lst))
    x = range(0, len(mean_generator_loss_lst), 1)
    # plotting the line 1 points    
    plt.plot(x, mean_generator_loss_lst, marker='o', label = "generator_loss")
    plt.plot(x, mean_discriminator_loss_lst, marker='v', label = "discriminator_loss")
 
    plt.xlabel('Steps - n')
    
    plt.ylabel('Mean loss at step n')

        
    # Set a title of the current axes.
    plt.title("Mean loss of Generator vs Discriminator")

    # show a legend on the plot
    plt.legend(ncol=2)
    #save the figure
    #plt.savefig(title + '.png')
    #display a figure.
    plt.show()
    #close the figure
    plt.close()
    
    #return 1

def find_user_vocab(user_token_lst):
    stopset = set(stopwords.words('english'))
    stopset.add("could")
    stopset.add("would")
    
    #print("len(user_token_lst) = ", len(user_token_lst))
    user_vocab = set()
    for item in user_token_lst:
        if item[0] not in user_vocab:
            user_vocab.add(item[0])
    return user_vocab - stopset

def restrict_wordset_English(train_user_token_lst):
    stopset = set(stopwords.words('english'))
    stopset.add("could")
    stopset.add("would")
    
    w_English = set(nltk.corpus.words.words())
    w_English = w_English - stopset
    user_dictionary = find_user_vocab(train_user_token_lst) #train_dataset.get_user_dictionary()
    #print("len(user_dictionary) = ", len(user_dictionary))
    #print("Before len(w_English) = ", len(w_English))
    w_English = w_English.union(user_dictionary)
    #print("After len(w_English) = ", len(w_English))
    
    return w_English

def restrict_pretrained_model(w2v, restricted_word_set):
    new_vectors = []
    new_vocab = {}
    new_index2entity = []
    new_vectors_norm = []
    
    for i in range(len(w2v.vocab)):
        word = w2v.index2entity[i]
        vec = w2v.vectors[i]
        vocab = w2v.vocab[word]
            
        vec_norm = w2v.vectors_norm[i]
        if word in restricted_word_set:
            vocab.index = len(new_index2entity)
            new_index2entity.append(word)
            new_vocab[word] = vocab
            new_vectors.append(vec)
            new_vectors_norm.append(vec_norm)

    w2v.vocab = new_vocab
    w2v.vectors = new_vectors
    #w2v.vectors = numpy.array(new_vectors)
    w2v.index2entity = new_index2entity
    w2v.index2word = new_index2entity
    w2v.vectors_norm = new_vectors_norm
    
    return w2v

def find_reconstructed_word(pretrained_model, user_dictionary, dataloader, vae_model):

    restricted_model = restrict_pretrained_model(pretrained_model, user_dictionary)

    #restricted_model = pretrained_model
    print("len(restricted_pretrained_model.vocab) = ", len(restricted_model.vocab))
    
    for i, (original_vector_bacth, true_word_batch) in enumerate(dataloader):
        reconstruct_vector_batch, mu, logvar = vae_model(original_vector_bacth)
        
        #if i < 10: #just inspect the first batch and compare reconstructed words with original ones
        #    print("Batch: {}".format(i))
        
        #reconstruct_vector_lst = []
        for j in range (len(true_word_batch)): 
            reconstruct_vector_jth = reconstruct_vector_batch[j].detach().numpy()
            tmp = restricted_model.similar_by_vector(reconstruct_vector_jth, topn = 10)
            if j < 10 and i == 0:
                print(true_word_batch[j])
                print([i for i in tmp])
            #print(pretrained_model.similar_by_vector(reconstruct_vector_jth))
                
            #if j > 10:
                #break
  
    word = "test"
    
    return word

def train_VAE(dataloader, num_epochs = 30):
    model = VAE()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print_per = 3646   
    model.train()
    #BETA = 0.01
    
    loss_record = []
    for epoch in range(num_epochs):
        train_loss = 0
        print_loss = 0
        for i, (data_batch, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data_batch)
            bce_loss, kl_loss = vae_loss_function(recon_batch, data_batch, mu, logvar)
            loss = bce_loss + kl_loss
            loss.backward()
            if i == len(dataloader) - 1:
                loss_record.append(loss.item())
            train_loss += loss.item()
            print_loss += loss.item()
            optimizer.step()
    
    return model

def train_GAN(dataloader, n_epochs = 200, lr = 0.00001, batch_size = 128, z_dim = 10, device = 'cpu'):
    criterion = nn.MSELoss(reduction='sum')  
    display_step = 50 #500

    gen = Generator(z_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc = Discriminator().to(device) 
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    
    
    # UNQ_C8 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
    # GRADED FUNCTION: 

    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    test_generator = True # Whether the generator should be tested
    gen_loss = False
    error = False

    mean_generator_loss_lst = []
    mean_discriminator_loss_lst = []
    for epoch in range(n_epochs):
  
        # Dataloader returns the batches
        for real, _ in tqdm(dataloader):
            cur_batch_size = len(real)

            # Flatten the batch of real images from the dataset
            real = real.view(cur_batch_size, -1).to(device)

            ### Update discriminator ###
            # Zero out the gradients before backpropagation
            disc_opt.zero_grad()

            # Calculate discriminator loss
            disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)

            # Update gradients
            disc_loss.backward(retain_graph=True)

            # Update optimizer
            disc_opt.step()

            # For testing purposes, to keep track of the generator weights
            if test_generator:
                old_generator_weights = gen.gen[0][0].weight.detach().clone()

            ### Update generator ###
            #     Hint: This code will look a lot like the discriminator updates!
            #     These are the steps you will need to complete:
            #       1) Zero out the gradients.
            #       2) Calculate the generator loss, assigning it to gen_loss.
            #       3) Backprop through the generator: update the gradients and optimizer.
            #### START CODE HERE ####
            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
            gen_loss.backward(retain_graph=True)
            gen_opt.step()
            #### END CODE HERE ####

            # For testing purposes, to check that your code changes the generator weights
            if test_generator:
                try:
                    assert lr > 0.0000002 or (gen.gen[0][0].weight.grad.abs().max() < 0.0005 and epoch == 0)
                    assert torch.any(gen.gen[0][0].weight.detach().clone() != old_generator_weights)
                except:
                    error = True
                    print("Runtime tests have failed")

            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_loss.item() / display_step

            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step            
        
            cur_step += 1
    
    #Display loss graph
    #plot_graph_(mean_generator_loss_lst, mean_discriminator_loss_lst)

    return gen

def extract_vocab(sampling_word_lst):
    vocab_dict = {}
    for word_batch in sampling_word_lst:
        for word in word_batch:
            if word[0] not in vocab_dict:
                vocab_dict[word[0]] = 1
    #print(vocab_dict)
    res = sorted((vocab_dict.keys()))
    #print("res = ", res[:10])
    return res

def dst_mass_inferring(sampling_word_lst, profile_length, num_OMEGA, threshold = 0.68):
    
    vocab_lst = extract_vocab(sampling_word_lst)    
    
    #Compute mass for all sampling bacths, each of which is stored in the sampling_word_lst
    mass_lst = []
    for word_batch in sampling_word_lst:
        tmp_lst = [item[0] for item in word_batch if item[1] > threshold]
        tmp_lst.sort()
        mass_lst.append(tmp_lst)

    print("mass_lst[:2] = ", mass_lst[:2])
    
    combinatorial_word_lst = []
    comb1 = list(combinations(vocab_lst, 1)) 
    comb2 = list(combinations(vocab_lst, 2)) 
    combinatorial_word_lst.extend(comb1)
    combinatorial_word_lst.extend(comb2)
    #print("finished combinatorial listing")
    
    d = {}
    for item in combinatorial_word_lst:
        d[item] = 1
        
    #Compute mass_lst via MAP
    #print("update mass via MAP is running")
    for item in mass_lst:
        tmp_comb = []
        tmp_comb1 = list(combinations(item, 1))
        tmp_comb2 = list(combinations(item, 2))
        tmp_comb.extend(tmp_comb1)
        tmp_comb.extend(tmp_comb2)
        
        #Add tmp_comb in d: adding the values with common key 
        for key in tmp_comb: 
            if key in d: 
                d[key] += 1 
                #print(str(key) + " is increased by 1")
            else:
                d[key] = 1
                #print(str(key) + " is not in user dictionary")
                pass
    
    d[('Z_OMEGA',)] = num_OMEGA + 1
    #print("Compute mass via MAP is finished")
    
    #Compute mass for all singleton via Smeth principle 
    result = {}
    for k, v in d.items():
        if len(k) == 1:
            result[k[0]] = v
        else: #len(k) > 1
            num_ele = len(k)
            for j in range (num_ele):
                if k[j] in result:
                    result[k[j]] += v/num_ele
                else:
                    result[k[j]] = v/num_ele
    
    sorted_d = dict( sorted(result.items(), key=operator.itemgetter(1),reverse=True))
    print("finished computing singleton_mass")
    
    # Using items() + list slicing 
    # Get first K items in dictionary 
    #res = dict(list(sorted_d.items())[0: profile_length]) 

    return sorted_d

def vae_sampling(model, pre_mu, pre_std, pretrained_model, pretrained_model_newVocab, dataloader, num_draw = 1000, topn = 5, lower_bound = 0.6, upper_bound = 0.9, alpha = 0.2):
    with torch.no_grad():
        count = 0
        num_EMPTY = 0
        num_OMEGA = 0
        mu_lst = []
        std_lst = []
        #print("len(dataloader) = ", len(dataloader))
        #batch_id = np.array(torch.randint(0, len(train_loader), [1,])[0]
        for i, (original_vector_bacth, true_word_batch) in enumerate(dataloader):
            recon_batch, mu, log_var = model(original_vector_bacth)
            mu = torch.mean(mu, dim = 0, keepdim=True)
            log_var = torch.mean(log_var, dim = 0, keepdim=True)
            std = torch.exp(0.5*log_var)
            mu_lst.append(mu)
            std_lst.append(std)
            #print("Shape(mu) = ", mu.shape)
            #print("Shape(log_var) = ", log_var.shape)
            
        #Get the mu and std for all data points      
        mu = torch.mean(torch.stack(mu_lst), 0, True)[0]  
        std = torch.mean(torch.stack(std_lst), 0, True)[0]
        
        if pre_mu != None:
            xyz = mu
            mu = (alpha * pre_mu) + ((1-alpha) * mu)
            std = (alpha * pre_std) + ((1-alpha) * std)
            
            #print("mu = ", xyz)
            #print("pre_mu = ", pre_mu)
            #print("new_mu = ", mu)
        
        #print("Shape(mu) after sum = ", mu.shape)
        #print("Shape(log_var) after sum = ", log_var.shape)
        sampling_word_lst = []
        #num_draw = 0
        #for ctr in range(0, 500, 5):
        for i in range (num_draw): #len(word_count(sampling_word_lst)) < profile_length: #or num_draw < 1000:
            eps_val = torch.randn((1, mu.shape[1]))
            #print("eps_val = ", eps_val)
            #print("mu = ", mu)
            #print("std = ", std)
            #eps_val = torch.full_like(mu, fill_value = ctr * 0.01 )
            z = eps_val.mul(std).add_(mu)
            recon_vector = model.decode(z)[0].detach().numpy()
            #print("recon_vector = ", recon_vector)
            #idx = np.array(torch.randint(0, len(dataloader), [1,]))[0]
            #idx = np.array(torch.randint(0, 20, [1,]))[0]
            #print("idx = ", idx)
            sampling_n_word = pretrained_model.similar_by_vector(recon_vector, topn = topn)#[0][0]
            #if i < 5:
            #    print('len(sampling_n_word) = ', len(sampling_n_word))
            #    print("sampling_n_word = ", sampling_n_word)
            #print(type(sampling_5_word))
            #sampling_word_lst.extend(sampling_n_word)
            #If the highest similar word is less than lower_bound, then it should be EMPTY set
            if sampling_n_word[0][1] < lower_bound: 
                num_EMPTY += 1
                #find new_word_lst from pretrained_model2.vocab
                sampling_word_lst.append(pretrained_model_newVocab.similar_by_vector(recon_vector, topn = int(len(pretrained_model.vocab)/10)))
                #print("EMPTY vae called")
                continue
                       
            #If the lowest similar word is greater than upper_bound, then it should be OMEGA set
            if sampling_n_word[topn - 1][1] > upper_bound: #OMEGA set
                num_OMEGA += 1 #no add to sampling_word_lst
                #print("OMEGA vae called")
                continue
            
            sampling_word_lst.append(sampling_n_word)
            #Just counting
            #foo(sampling_5_word)
    #print("len(sampling_word_lst)", len(sampling_word_lst))    
    return sampling_word_lst, num_EMPTY, num_OMEGA, mu, std

def gan_sampling(gen_model, pre_gan, pretrained_model, pretrained_model_newVocab, z_dim = 10, num_draw = 1000, topn = 5, lower_bound = 0.6, upper_bound = 0.9, device = 'cpu', alpha = 0.2):
    
    fake_noise = get_noise(num_draw, z_dim = z_dim, device=device)
    fake = gen_model(fake_noise)
    
    if pre_gan != None:
        pre_fake = pre_gan(fake_noise)
        
    num_EMPTY = 0
    num_OMEGA = 0
    sampling_word_lst = []
    for i in range (num_draw):
        tmp_vector = fake[i].cpu().detach().numpy()
        
        if pre_gan != None:
            pre_vector = pre_fake[i].cpu().detach().numpy()
            #print("current gan = ", tmp_vector)
            tmp_vector = (alpha * pre_vector) + (1 - alpha) * tmp_vector
            #print("pre gan = ", pre_vector)
            #print("combine vector = ", tmp_vector)
        
        #print(pretrained_model.similar_by_vector(tmp_vector, topn = 10))
        sampling_n_word = pretrained_model.similar_by_vector(tmp_vector, topn = topn)#[0][0]
        #if i < 10:
        #    print(sampling_10_word)
        #print(type(sampling_5_word))
        if sampling_n_word[0][1] < lower_bound: 
            num_EMPTY += 1
            #find new_word_lst from pretrained_model2.vocab
            sampling_word_lst.append(pretrained_model_newVocab.similar_by_vector(tmp_vector, topn = int(len(pretrained_model.vocab)/10)))
            #print("EMPTY gan called")
            continue
                       
            #If the lowest similar word is greater than upper_bound, then it should be OMEGA set
        if sampling_n_word[topn - 1][1] > upper_bound: #OMEGA set
            num_OMEGA += 1 #no add to sampling_word_lst
            #print("OMEGA gan called")
            continue
        
        sampling_word_lst.append(sampling_n_word)

    return sampling_word_lst, num_EMPTY, num_OMEGA

def normalize_mass(input_dict):
    key_lst = list(input_dict.keys())
    val_lst = list(input_dict.values())
    #print("key_lst = ", key_lst)
    #print("val_lst = ", val_lst)
    
    res = {}
    s = np.sum(val_lst)
    for k, v in input_dict.items():
        res[k] = v/s
    
    #print("res = ", res)
    #print("sum = ", np.sum(list(res.values())))
    return res

def combine_2_source_naive(s1, s2, weight1 = 0.5, weight2 = 0.5, topn = 100):
    d = {}
    #Copy s1 into d
    for k1, v1 in s1.items():
        d[k1] = v1
        
    #Merge s2 into d
    for k2, v2 in s2.items():
        if k2 in d:
            d[k2] += v2
        else:
            d[k2] = v2
            
    #Sort the dictionary in descending order of values
    sorted_d = dict( sorted(d.items(), key=operator.itemgetter(1),reverse=True))
    
    #res = dict(list(sorted_d.items())[0: topn]) 
    res = list(sorted_d.keys())[0: topn]
    
    return res

def output_to_file(filename, out_tuple):
    with open(filename, 'w') as f:
        #output user hashtags
        json.dump(out_tuple[0], f) #write hashtags to file
        f.write('\n')
        
        #output user dst
        json.dump(out_tuple[1], f) #write profile to file: each is a tuple (profile_lst, ground_truth_lst)
        f.write('\n')
        
        #output txtrk
        json.dump(out_tuple[2], f)
        f.write('\n')
        
        #output rake
        json.dump(out_tuple[3], f)
        f.write('\n')
        
        #output tfidf
        json.dump(out_tuple[4], f)
        f.write('\n')
        
        #output lda
        json.dump(out_tuple[5], f)
        f.write('\n')
        
        #output gsdmm
        json.dump(out_tuple[6], f)
        f.write('\n')
        
        f.close()
    print("outfile is written successfully")
    return 1

def load_pre_trained_model(name):
    # download the model and return as object ready for use
    model = api.load(name)
    print("Model loaded successfully")
    return model

def load_corpus_from_file(filename):
    data = []
    with open(filename,'r', encoding='utf-8') as f:
        for line in f:
            l = json.loads(line)
            if len(l) > 0:
                #data.append(l)
                data.insert(0, l) 
    return data.pop(len(data) - 1), data



# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import Word2Vec
import gensim.downloader as api

import pickle
import timeit
import json 
import nltk
from nltk.cluster.kmeans import KMeansClusterer
import numpy as np
import matplotlib.pyplot as plt
import spacy
from nltk.tokenize.treebank import TreebankWordDetokenizer
from rake_nltk import Rake
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from GSDMM import GSDMM


class Child_DMM(GSDMM.DMM):
    def writeTopTopicalWords(self, selected_topics):
        #file = open("_%s_DMM_topics_Kstart.topWords" % (self.nTopics),"w") 
        coherence_index_all=[]
        profiles = []
        for t in selected_topics:
            wordCount = {w:self.topicWordCount[t][w] for w in range(len(self.id2word))}
            count = 0
            #string = ""
            keys = []
            coherence_index_per_topic=[]
            for index in sorted(wordCount, key=wordCount.get, reverse=True):
                coherence_index_per_topic.append(index)
                #string += self.id2word[index]+" "
                keys.append(self.id2word[index])
                count+=1
                #print(count)
                if count>=self.twords:
                    #file.write(string+"\n") 
                    #file.write(str(keys)+"\n") 
                    #print(string)
                    #print(keys)
                    break
            coherence_index_all.append(coherence_index_per_topic)
            profiles.append(keys)
        #file.close()
        #print("my method")
        return coherence_index_all, profiles

def tfidf_based_user_profile(user_corpus, num_keywords):
    #hashtags, user_corpus = load_user_preprocessed_textfile(filename)
    #refine_corpus = update_user_corpus(pre_train_model, user_corpus)

    user_corpus = [TreebankWordDetokenizer().detokenize(token_list) for token_list in user_corpus] 
    #print("len(user_corpus) = ", len(user_corpus))
    #print("user_corpus[0] = ", (user_corpus[0]))
    #print("user_corpus[1] = ", (user_corpus[1]))
    #print("num_keywords = ", num_keywords)
    #partition user's posts into n documents (n = num_keywords)
    #each document consists of m posts
    chunk_size = int(len(user_corpus)/num_keywords)
    #print("chunk_size = ", chunk_size)
    num_chunks = num_keywords
    #print("num_chunks = ", num_chunks)
    remain = len(user_corpus) - (num_chunks * chunk_size)
    #print("remain = ", remain)
    chunks = [user_corpus[x:x + chunk_size] for x in range(0, len(user_corpus) - remain, chunk_size)]

    #Add the remaining posts into the final chunk
    for i in range (0, remain):
        chunks[num_chunks - 1].append(user_corpus[chunk_size*num_chunks + i])
    
    #for i in range (0, len(chunks)):
    #print("len(chunks) = ", len(chunks))
        
    stop_words = stopwords.words('english')
    stop_words.extend(['rt', 'actually', 'via', 'by','ah'])
    
    vectorizer = TfidfVectorizer(max_df=1.0,
                                 min_df=0.0, 
                                 stop_words=stop_words, 
                                 use_idf=True)
    
    all_docs = []
    for i in range (0, len(chunks)):
        X = vectorizer.fit_transform(chunks[i])
        indices = np.argsort(vectorizer.idf_)[::-1]
        features = vectorizer.get_feature_names()
        keywords = [features[i] for i in indices]
        #print("type(keywords) = ", type(keywords))
        all_docs.append(keywords)
        
    #print("len(all_docs) = ", len(all_docs))
    #print("all_docs[0] = ", all_docs[0])
    
    result = []
    while len(result) < num_keywords:
        for i in range (0, len(all_docs)):
            if len(all_docs[i]) > 0:
                word = all_docs[i].pop(0)
                #print("word = ", word)
                if word not in result:
                    result.append(word)
                if len(result) == num_keywords:
                    break
    
    #print("len of TFIDF = ", len(result))
    #print("TFIDF = ",result)
    
    return result
    
def build_lda_model(user_corpus, num_topics):
    # Create Dictionary
    id2word = corpora.Dictionary(user_corpus)

    # Create Corpus
    texts = user_corpus

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=num_topics, 
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
    
    
    # Print the Keyword in the 10 topics
    #pprint(lda_model.print_topics())
    #doc_lda = lda_model[corpus]
    
    # Compute Perplexity
    #perplexity = lda_model.log_perplexity(corpus)
    #print('\nPerplexity: ', perplexity)  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    #coherence_model_lda = CoherenceModel(model=lda_model, texts=user_corpus, dictionary=id2word, coherence='c_v')
    #coherence_lda = coherence_model_lda.get_coherence()
    #print('\nCoherence Score: ', coherence_lda)
    #visualize_lda(lda_model, corpus, id2word)
    return lda_model

def lda_based_user_profile(user_corpus, num_keywords):
    
    #hashtags, user_corpus = load_user_preprocessed_textfile(filename)
    #refine_corpus = update_user_corpus(pre_train_model, data)
    lda_model = build_lda_model(user_corpus, num_topics=num_keywords)
    
    n_keywords = []
    index_of_the_word_i_in_topic = 0
    while len(n_keywords) < num_keywords:
        for i in range (num_keywords):
            word = lda_model.show_topic(i,topn=num_keywords)[index_of_the_word_i_in_topic][0]
            if word not in n_keywords:
                n_keywords.append(word)
                if len(n_keywords) == num_keywords:
                    break
        #print("n_keywords = ", n_keywords)
        index_of_the_word_i_in_topic += 1
    
    #result_dict = {}
    #for w in n_keywords:
    #    result_dict[w] = user_dictionary[w]
    
    #print("result_dict = ", result_dict)
    #hashtags = {k:v for k,v in hashtags.items() if v > 1}
    
    return n_keywords#, result_dict

def rake_based_user_profile(user_corpus, num_keywords):
    #read preprocessed data from file
    #hashtags, data = load_user_preprocessed_textfile(filename)
    #print("data len = ", len(data))
    #print("filename = ", filename)
    
    #Refine corpus by eliminating all words that are not in the pretrain model
    #refine_corpus = update_user_corpus(pre_train_model, data)
    #print("refine_corpus len = ", len(refine_corpus))
    
    #partition user's posts into n documents (n = num_keywords)
    #each document consists of m posts
    chunk_size = int(len(user_corpus)/num_keywords)
    num_chunks = num_keywords
    remain = len(user_corpus) - (num_chunks * chunk_size)
    chunks = [user_corpus[x:x + chunk_size] for x in range(0, len(user_corpus) - remain, chunk_size)]

    #Add the remaining posts into the final chunk
    for i in range (0, remain):
        chunks[num_chunks - 1].append(user_corpus[chunk_size*num_chunks + i])
  
    #Reformat the data into sentence so that we can apply builtin function in RAKE library
    sentence_lst = []
    for i in range(0, len(chunks)):  
        #print("len(chunk[i]) = ", len(chunk))
        str_x = ''
        for post in chunks[i]:
            tmp = TreebankWordDetokenizer().detokenize(post)
            if len(tmp) > 0:
                str_x = str_x + ' ' + tmp + '.'
        
        #print("chunk i = ", str_x[:1000])  
        sentence_lst.append(str_x)
    
    #print("Len(sentence_lst = )", len(sentence_lst))
    
    #apply rake for n documents
    #each document, we get a set of key words
    r = Rake()
    keywords = []
    for i in range(0, len(chunks)):
        r.extract_keywords_from_text(sentence_lst[i])
        rank_lst = r.get_ranked_phrases() #To get keyword phrases ranked highest to lowest.
        #print("i = ", i)
        #print(rank_lst)
        keyword = rank_lst[0].split()[0] #Extract one keyword with highest score
        #print("tokens = ", keyword)
        keywords.append(keyword)
    
    #hashtags = {k:v for k,v in hashtags.items() if v > 1}
    
    return keywords

def txtrk_keywords_extraction(textrank_results):
    keywords = []
    for item in textrank_results:
        tmp = item.split(' ')
        #print('tmp = ', tmp)
        if len(tmp) == 1:
            keywords.append(tmp[0])
            continue
        if len(tmp) > 1:
            for k in tmp:
                keywords.append(k)
    #print(keywords)
    return keywords

def txtrk_based_user_profile(user_corpus, num_keywords):
    #read preprocessed data from file
    #hashtags, data = load_user_preprocessed_textfile(filename)
    #print("data len = ", len(data))
    #print("filename = ", filename)
    
    #Refine corpus by eliminating all words that are not in the pretrain model
    #refine_corpus = update_user_corpus(pre_train_model, data)
    #print("refine_corpus len = ", len(refine_corpus))
    
    #partition user's posts into n documents (n = num_keywords)
    #each document consists of m posts
    
    chunk_size = int(len(user_corpus)/num_keywords)
    num_chunks = num_keywords
    remain = len(user_corpus) - (num_chunks * chunk_size)
    chunks = [user_corpus[x:x + chunk_size] for x in range(0, len(user_corpus) - remain, chunk_size)]

    #Add the remaining posts into the final chunk
    for i in range (0, remain):
        chunks[num_chunks - 1].append(user_corpus[chunk_size*num_chunks + i])
  
    #Reformat the data into sentence so that we can apply builtin function in Gensim library
    sentence_lst = []
    for i in range(0, len(chunks)):  
        #print("len(chunk[i]) = ", len(chunk))
        str_x = ''
        for post in chunks[i]:
            tmp = TreebankWordDetokenizer().detokenize(post)
            if len(tmp) > 0:
                str_x = str_x + ' ' + tmp + '.'
        
        #print("chunk i = ", str_x[:1000])  
        sentence_lst.append(str_x)
    
    #print("Len(sentence_lst = )", len(sentence_lst))

    from gensim.summarization import keywords as textrank
    #apply textrank for n documents
    #each document, we get a set of key words
    keywords = []
    for i in range(0, len(chunks)):
        tmp = txtrk_keywords_extraction(textrank(sentence_lst[i]).split('\n'))
        for word in tmp:
            if word not in keywords:
                keywords.append(word)
                break
    
    #hashtags = {k:v for k,v in hashtags.items() if v > 1}
    
    return keywords

def gsdmm_based_user_profile(user_corpus, num_keywords):
    data_dmm = Child_DMM(user_corpus, nTopWords = 12, nTopics = num_keywords) # Initialize the object, with default parameters.
    data_dmm.topicAssigmentInitialise() # Performs the inital document assignments and counts
    data_dmm.inference()
    finalAssignments = data_dmm.writeTopicAssignments() # Records the final topic assignments for the documents
    _, topics = data_dmm.writeTopTopicalWords(finalAssignments) # Record the top words for each document
    
    #print(topics)
    
    profile_at_K = []
    while len(profile_at_K) < num_keywords:
        for topic in topics:
            for word in topic:
                if word not in profile_at_K:
                    profile_at_K.append(word)
                    break
            if len(profile_at_K) == num_keywords:
                return profile_at_K
    
    #print('profile_at_K = ', profile_at_K)
    print("len(gsdmm) = ", len(profile_at_K))
    return profile_at_K
    
def baselines(user_corpus, ground_truth):
    txtrk_profile_at_k_lst = []
    txtrk_runtime_lst = []
    
    rake_profile_at_k_lst = []
    rake_runtime_lst = []

    tfidf_profile_at_k_lst = []
    tfidf_runtime_lst = []

    lda_profile_at_k_lst = []
    lda_runtime_lst = []

    gsdmm_profile_at_k_lst = []
    gsdmm_runtime_lst = []

    for j in range (5,
                    51, 5):
        #TextRank model
        start_time = timeit.default_timer()
        txtrk_profile_at_k = txtrk_based_user_profile(user_corpus, num_keywords = j)
        stop_time = timeit.default_timer()
        txtrk_profile_at_k_lst.append(txtrk_profile_at_k)
        txtrk_runtime_lst.append(stop_time - start_time)

        #RAKE model
        start_time = timeit.default_timer()
        rake_profile_at_k = rake_based_user_profile(user_corpus, num_keywords = j)
        stop_time = timeit.default_timer()
        rake_profile_at_k_lst.append(rake_profile_at_k)
        rake_runtime_lst.append(stop_time - start_time)
        
        #TDIDF model
        start_time = timeit.default_timer()
        tfidf_profile_at_k = tfidf_based_user_profile(user_corpus, num_keywords = j)
        stop_time = timeit.default_timer()
        tfidf_profile_at_k_lst.append(tfidf_profile_at_k)
        tfidf_runtime_lst.append(stop_time - start_time)
        
        #LDA model
        start_time = timeit.default_timer()
        lda_profile_at_k = lda_based_user_profile(user_corpus, num_keywords = j)
        stop_time = timeit.default_timer()
        lda_profile_at_k_lst.append(lda_profile_at_k)
        lda_runtime_lst.append(stop_time - start_time)
    
        #GSDMM model
        start_time = timeit.default_timer()
        gsdmm_profile_at_k = gsdmm_based_user_profile(user_corpus, num_keywords = j)
        stop_time = timeit.default_timer()
        gsdmm_profile_at_k_lst.append(gsdmm_profile_at_k)
        gsdmm_runtime_lst.append(stop_time - start_time)
    
    #Append ground_truth at the end of profile list
    txtrk_profile_at_k_lst.append(ground_truth)
    rake_profile_at_k_lst.append(ground_truth)
    tfidf_profile_at_k_lst.append(ground_truth)
    lda_profile_at_k_lst.append(ground_truth)
    gsdmm_profile_at_k_lst.append(ground_truth)
    
    #Append runtime list at the end of profile list
    txtrk_profile_at_k_lst.append(txtrk_runtime_lst)
    rake_profile_at_k_lst.append(rake_runtime_lst)
    tfidf_profile_at_k_lst.append(tfidf_runtime_lst)
    lda_profile_at_k_lst.append(lda_runtime_lst)
    gsdmm_profile_at_k_lst.append(gsdmm_runtime_lst)
    
    return txtrk_profile_at_k_lst, rake_profile_at_k_lst, tfidf_profile_at_k_lst, lda_profile_at_k_lst, gsdmm_profile_at_k_lst

def run_one_user(original_pretrained_model, in_file, out_file, num_windows = 10, device = 'cpu'):
    
    #Load the entire data of an user
    user_hashtags, user_corpus = load_corpus_from_file(in_file)     
    stepsize = int(len(user_corpus)/num_windows)
    
    print("len(user_corpus) = ", len(user_corpus))
    print("num_windows = ", num_windows)
    print("stepsize = len(user_corpus)/num_windows = ", stepsize)

    pre_mu = None
    pre_std = None
    pre_gan = None
    
    dst_lst = []
    txtrk_lst = []
    rake_lst = []
    tfidf_lst = []
    lda_lst = []
    gsdmm_lst = []
    #Start for loop here
    for i in range (0, num_windows):
        #1. PREPARE DATA
        #Copy the pretrained Glove model
        pretrained_model = copy.deepcopy(original_pretrained_model) 
        pretrained_model2 = copy.deepcopy(original_pretrained_model) 
        
        #Calculate the data within window_size
        from_idx = i * stepsize
        to_idx = i * stepsize + stepsize
        if i ==  num_windows - 1:
            to_idx = len(user_corpus)
        
        batch_corpus = user_corpus[from_idx:to_idx]
        print("len(batch_corpus) = ", len(batch_corpus))
        
        custom_dataset = CustomDatasetFromFile(batch_corpus, user_hashtags, pretrained_model, argmentation = 20)
        train_user_token_lst, test_user_token_lst = custom_dataset.get_user_token_lst()
        #user_hashtags = custom_dataset.get_user_hashtags()
        dataloader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=128, shuffle=False)
    
        print("Load data finished")
    
        #2. TRAIN VAE and GAN
        vae = train_VAE(dataloader, num_epochs = 40)
        #print("train_VAE finished")
            
        gan = train_GAN(dataloader, n_epochs = 260, device = device)
        #print("train_GAN finished")
    
        #restrict model to English language only
        #print("Before len(pretrained_model.vocab) = : ", len(pretrained_model.vocab))
        #print("Before len(pretrained_model2.vocab) = : ", len(pretrained_model2.vocab))
        pretrained_model = restrict_pretrained_model(pretrained_model, find_user_vocab(train_user_token_lst))
        pretrained_model2 = restrict_pretrained_model(pretrained_model2, restrict_wordset_English(train_user_token_lst))
        #print("After len(pretrained_model.vocab) = : ", len(pretrained_model.vocab))
        #print("After len(pretrained_model2.vocab) = : ", len(pretrained_model2.vocab))

        print("Reconstructed words of VAE")
        find_reconstructed_word(pretrained_model, find_user_vocab(train_user_token_lst), dataloader, vae)

        #3. SAMPLING
        topn = int(len(find_user_vocab(train_user_token_lst))/10)
        #print("topn = ", topn)
        
        #start_time = timeit.default_timer()        

        vae_samples, vae_num_EMPTY, vae_num_OMEGA, pre_mu, pre_std = vae_sampling(vae, pre_mu, pre_std, pretrained_model, pretrained_model2, dataloader, topn = topn)
    
        #print("len(vae_samples) = ", len(vae_samples))
        #print("vae_samples[:1] = ", vae_samples[:1])
        #print("vae_num_EMPTY = ", vae_num_EMPTY)
        #print("vae_num_OMEGA = ", vae_num_OMEGA)

        gan_samples, gan_num_EMPTY, gan_num_OMEGA = gan_sampling(gan, pre_gan, pretrained_model, pretrained_model2, z_dim = 10, topn = topn, lower_bound = 0.2, device = device)

        #print("len(gan_samples) = ", len(gan_samples))
        #print("gan_samples[:1] = ", gan_samples[:1])
        #print("gan_num_EMPTY = ", gan_num_EMPTY)
        #print("gan_num_OMEGA = ", gan_num_OMEGA)
    
        #4. MASS INFERRING
        #print("vae_mass = ")
        start_time = timeit.default_timer()
        vae_mass_inferring = dst_mass_inferring(vae_samples, 15, vae_num_OMEGA, 0.6)
        #print("gan_mass = ")
        gan_mass_inferring = dst_mass_inferring(gan_samples, 15, gan_num_OMEGA, 0.6)
    
        #5. MASS COMBINATION
        dst_out_profile = combine_2_source_naive(normalize_mass(vae_mass_inferring), normalize_mass(gan_mass_inferring))
        stop_time = timeit.default_timer()
        
        dst_lst.append((dst_out_profile,list(find_user_vocab(test_user_token_lst)), stop_time - start_time))
        
        if i > 0:
            pre_gan = gan
        
        train_corpus, test_corpus = custom_dataset.get_train_test_corpus()
        txtrk, rake, tfidf, lda, gsdmm = baselines(train_corpus, list(find_user_vocab(test_user_token_lst)))
        
        txtrk_lst.append(txtrk)
        rake_lst.append(rake)
        tfidf_lst.append(tfidf)
        lda_lst.append(lda)
        gsdmm_lst.append(gsdmm)
        
        print("finished ith run ", i)
        
    #6. WRITE out_profile TO FILE
    output_to_file(filename = out_file, out_tuple = (user_hashtags, dst_lst, txtrk_lst, rake_lst, tfidf_lst, lda_lst, gsdmm_lst))
    
    return dst_lst

if __name__ == "__main__": 

    #dataset = 'facebook_0500'
    dataset = sys.argv[1]            #'twitter' or 'facebook'
    model_name = sys.argv[2]         #50 or 100 or 200
    fromID = int(sys.argv[5])        #integer
    toID = int(sys.argv[6])          #integer
    
    
    #dataset = 'twitter_1196'         #argument
    #model_name = 'glove-twitter-50'  #argument
    #fromID = 0                       #argument 
    #toID = 13                         #argument
    #device = 'cuda'
    device = sys.argv[3]
    batch_num = int(sys.argv[4])
    

    
    original_pretrained_model = load_pre_trained_model(model_name)  
    print("Before len(original_pretrained_model): ", len(original_pretrained_model.vocab))
    print("background = ", original_pretrained_model.most_similar(positive=['background'], topn=5))
    #original_pretrained_model = restrict_pretrained_model(original_pretrained_model, restrict_wordset_English())    
    
    start_time = timeit.default_timer()
    for i in range (fromID, toID + 1):
        tem = run_one_user(original_pretrained_model, 
                           in_file = "data/processed/" + dataset + "/" + str(i) + ".json",
                           out_file = "data/output/" + dataset + "/" + str(i) + ".json",
                           num_windows = batch_num, 
                           device = device)
        
        print("len(tem) = ", len(tem))
        print("finished user ", i)
    stop_time = timeit.default_timer()
    
    print('Time in hours: ', (stop_time - start_time)/3600)
    print("THE END !!!!")