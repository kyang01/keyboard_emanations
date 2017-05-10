'''

    prediction_lib.py


    This library contains helper functions for the Processing notebook.

    These functions are used to extract features from the raw input after the
    characters have been labeled. In addition, there are functions that perform
    the clustering of labels, and build the hidden markov models 

'''

# Numpy and pandas
import pandas as pd
import numpy as np

# Matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

# SKLEARN
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.decomposition import PCA
from scipy.cluster.vq import kmeans2,vq, whiten
from sklearn.cluster import KMeans

# Core
import sys, re
from collections import Counter

# HMM models 
from hmmlearn.hmm import MultinomialHMM

# For feature extraction
from python_speech_features import mfcc


def extract_cepstrum(df, rate, mfcc_start=0, mfcc_end=-1, winlen = 0.025, winstep = 0.01,
                    numcep = 16, nfilt = 32,nfft=512, lowfreq = 400, highfreq = 12000, noise = None):
    '''
        Extracts the cepstrum features from the raw signal data 

            df : a dataframe where the indices are the timepoint for each 
                supposed key press 

            rate : The rate at which the sound file was processed either when call
                made to  spl.open_audio() or scipy.io.wavfile.wav()

            mfcc_start/mfcc_end : indices to slice the feature vector

            remainder of args are passed into the mfcc punction 
    '''
    rate = float(rate)
    # Convert raw signal into list of numpy arrays 
    char_data = df[df.columns[list(df.columns).index('0'):]].values
    if noise:
        char_data += np.random.normal(0, noise, char_data.shape) 
    keypress_sigs = [np.nan_to_num(np.squeeze(l)) for l in np.split(char_data, char_data.shape[0], axis=0)]
    
    # Create the keypress features one by one
    keypress_feats = []
    for keypress_sig in keypress_sigs:

        mfcc_feat = mfcc(keypress_sig, samplerate= rate, winlen=winlen, 
                         winstep=winstep, numcep=numcep, nfilt=nfilt, 
                         lowfreq=lowfreq, nfft = nfft, highfreq=highfreq)
        keypress_feats.append(np.concatenate(mfcc_feat[mfcc_start:mfcc_end, :]).T)

    # Create cepstrum dataframe
    cepstrum_df = pd.DataFrame(np.vstack(keypress_feats))

    # Copy over true char labels
    if 'char' in cepstrum_df:
        cepstrum_df['char'] = df['char']

    # Put the char labels at the front
    cepstrum_df = cepstrum_df.reindex(columns = [cepstrum_df.columns[-1]] + list(cepstrum_df.columns[:-1]))

    return cepstrum_df

def cluster(df, whiten_data=True, num_clusters=50, n_init = 100, n_components = 30):
    '''
        Cluster the data frame (expected to be of cepstrum features but no required)
        If n_components has a value, PCA will be performed with that value on the df
        to create the input for KMeans. If not the input will be the raw data.

            df : dataframe to cluster 

            whiten_data : whether or not to standardize the data

            num_clusters : the number of groups to target for kemans 

            n_init : number of random restarts for kmeans algorithm

            n_components : if specified, pca is performed on the df with this many components,
                if None, no PCA is performed

    '''

    # Inds of data
    inds = df.dtypes == np.float64

    data = df.ix[:,inds].values

    # Standardize data
    if whiten_data:
        data = whiten(data)

    # Perform PCA
    if n_components:  
        pca = PCA(n_components=n_components)

        # Transform data
        pca.fit(data)
        data = pca.transform(data)
   
    # Run kmeans on data
    kmeans = KMeans(n_clusters=num_clusters, n_init = n_init).fit(data)
    
    # Get labels from running clustering
    df['cluster'] = kmeans.labels_.reshape(-1, 1)
    return df

def to_hist(data, num_clusters, smooth = 0):
    '''
        Groups data by cluster and returns results for a bar plot histogram

            data : df to cluster 

            num_clusters : The number of clusters 

            smooth : A Smoothing paramter for histogram
    '''

    # Turn to df
    df = pd.DataFrame(data)
    df['Count'] = 1

    # Group by culuster
    counts = df.groupby('cluster').count() 

    # Determine counts
    return counts.reindex(index = range(num_clusters)).fillna(0) + smooth


def cluster_proportions(df, input_char, num_clusters = 50):
    '''
        Determine with what proportion input_char is in each of the num_clusters 
        clusters 

            df : dataframe with cluster/char labels 

            input_char : The char to cluster by

            num_clusters : The number of cluststers
    '''
    # Function that will plot histogram of cluster proportions
    plot_hist = lambda x : df[df['char'] == x]['cluster'].hist(bins = bins)

    if type(input_char) == list:
        n = len(input_char)
        fig, axes = plt.subplots((n-1)/4+1, 4, figsize=(24, n*3/2))

        # Repeat for each coefficient
        for idx, ch in enumerate(input_char):
            if ((n+1) / 4 ) > 1:
                axe = axes[idx / 4][idx % 4]
            else:
                axe = axes[idx]
            sub_df = df[df['char'] == ch]
            counts = to_hist(sub_df['cluster'], num_clusters)
            counts.plot(kind='bar', ax = axe)
            axe.set_title(ch)
    else:
        plot_hist(input_char)

def view_char(df, input_char, xlim = 6000, limit = 10):
    '''
        Visualize what the input_char labels look like in df

            df : The dataframe with char label and features 

            input_char : The character to visualize 

            xlim : The xlimit for the plot 

            limit : The number of items to view 

    '''

    # Grab inputs with this specific char 
    sub_df = df[df['char'] == input_char]

    # The number of items to show
    n = np.min([sub_df.shape[0], limit])

    if n < 1:
        print('No "%s" found' % input_char)
        return
    
    # Get the index where features begin
    try:
        lim = list(df.columns).index('0')
    except:
        lim = list(df.columns).index(0)


    print('Visualizing: %s' % input_char)
    
    # Plot histogram of coefficients, and report their confidence intervals 
    fig, axes = plt.subplots((n-1)/4+1, 4, figsize=(12, n*3/2))

    # Repeat for each coefficient
    for idx in range(n):
        if ((n+1) / 4 ) > 1:
            axe = axes[idx / 4][idx % 4]
        else:
            axe = axes[idx]

        # Plot histogram of coefficient values
        sub_df.iloc[idx].ix[lim:].plot(ax = axe)
        axe.set_ylim([-xlim, xlim])

def build_transmission(text, smooth = .2):
    '''
        Builds the transition matrix for the hmm by using the 
        frequencies in the target text 

            text : target text 

            smooth : smoothing paramter of transmission matrix 
    '''

    # Determine which chars appear
    unique_chars = np.unique(list(text))

    # Number of unique chars
    n_unique = len(unique_chars)

    # Map char to id and vice versa
    id_to_char = dict(zip(range(n_unique), unique_chars))
    char_to_id = dict(zip(unique_chars, range(n_unique)))

    # Determine bigrams in text
    char_bigrams = dict(Counter([text[i:i+2] for i in range(len(text)-1)]))

    # Initialize transmission to smoothing parameter
    A = np.zeros((n_unique,n_unique)) + smooth

    # For each bigram, add the counts
    for big in char_bigrams:
        x, y = map(lambda x : char_to_id[x], big)
        A[x,y] += char_bigrams[big]

    # Reindex to the characters
    A_df = pd.DataFrame(A, index = unique_chars, columns = unique_chars)

    # Make each row sum to 1
    A_df = A_df.apply(lambda x : x / x.sum(), axis = 1)
    
    return A_df, n_unique, unique_chars, id_to_char, char_to_id

def get_props(df,num_clusters,  ch = ' ', smooth_space = 1):
    ''' 
        Determine what proportion of clusters ch is in 

            df : dataframe with char and cluster labels 

            num_clusters : number of possible clusters 

            ch : The char of which we want to see the proportions for

    '''

    # Get the clusters for the char 
    counts = df[df['char'] == ch]['cluster']

    # Get proportions 
    props = to_hist(counts, num_clusters, smooth_space)['Count']
    props /= props.sum()
    return props

def build_eta(df, unique_chars, num_clusters, do_all = False):
    '''
        Build the emissions matrix eta by randomly initializing 
        and then filling the space row with the proportions from 
        the labeled dataset 

            df : dataframe of cluster/char labels

            unique_chars : The list of unique chars we are identifyinh

            num_clusters : The number of clusters we grouped 

            do_all : When true, replaces all rows of emissions matrix 
                with labeled proportions 
    '''
    # Initialize to random noise 
    Eta = np.random.rand(len(unique_chars), num_clusters)
    
    # Determine the label for a space
    space_ind = list(unique_chars).index(' ')

    # Fill in the space with its proportions 
    Eta[space_ind,:] = get_props(df,num_clusters,  ch = ' ').values
    
    # Fill in the others with their proportions
    if do_all:
        uc = list(unique_chars)
        for ch in uc:
            space_ind = uc.index(ch)
            Eta[space_ind,:] = get_props(df,num_clusters, ch = ch).values
    
    # Get rows to sum to 1
    Eta/=Eta.sum(axis=1)[:,None]
    return Eta

def get_char_counts():
    '''
        Returns the distribution over starting character frequencies for english 
    '''
    # Read in proportions of word starts
    word_starts = pd.read_csv('data/word_starts.txt', sep ='\t', index_col=0, header = None)[1]
    word_starts = word_starts.map(lambda x : float(x[:-1])/100.)
    return word_starts

def build_transmission_full(smooth = 0):
    '''
        Builds the transmission matrix from bigrams in english language 
    '''

    # Unique chars are a-z and space
    unique_chars = np.array([' '] + list(map(lambda x : chr(x + ord('a')), range(26))))

    # Number of unique
    n_unique = len(unique_chars)

    # Map char to id and vice versa
    id_to_char = dict(zip(range(n_unique), unique_chars))
    char_to_id = dict(zip(unique_chars, range(n_unique)))

    # Open file of raw text
    with open('data/guttenberg_text.txt', 'r') as f:
        all_text = f.read()
    
    # Determine character bigram frequencies 
    char_bigrams = dict(Counter(str(x+y) for x, y in zip(*[all_text[i:] for i in range(2)])))
        
    # Initialize transmission matrix to smoothing paramter
    A = np.zeros((n_unique,n_unique)) + smooth

    # Add each bigram count
    for big in char_bigrams:
        x, y = map(lambda x : char_to_id[x], big)
        A[x,y] += char_bigrams[big]

    assert(char_to_id[' '] == 0)

    # Reindex and sum rows to 1
    A_df = pd.DataFrame(A, index = unique_chars, columns = unique_chars)
    A_df = A_df.apply(lambda x : x / x.sum(), axis = 1)

    return A_df, n_unique, unique_chars, id_to_char, char_to_id


def to_text(results, id_to_char):
    '''
        Convert the Results from the hmm into plaintext

            results : list of char ids 

            id_to_char : mapping of id to character 
    '''
    return ''.join(map(lambda x : id_to_char[x], results))
def accuracy(a,b):
    '''
        Determine similarity of strings a and b 

            a, b : two strings of the same length 
    '''
    return 1- (sum ( a[i] != b[i] for i in range(len(a)) ) / float(len(a)))

def accuracywospace(a,b):
    '''
        Determine similarity of strings a and b not counting spaces

            a : The estimated text 

            b : The actual text
    '''

    # Indices of spaces 
    inds = np.where(np.array(list(b)) != ' ')

    # Function to remove spaces 
    remove_spaces = lambda x : list(np.array(list(x))[inds])
    a, b = remove_spaces(a), remove_spaces(b)

    return 1- (sum ( a[i] != b[i] for i in range(len(a)) ) / float(len(a)))

def print_color(estimate, text, form_str = "\x1b[{}m{}\x1b[0m"):
    '''
        Prints the estimate with red characters for incorrect and
        green characters for correct 
    '''
    correct = [estimate[i] == text[i] for i in range(len(estimate))]
    return ''.join(list(map(lambda x : form_str.format(32, estimate[x]) if correct[x] else form_str.format(31, estimate[x]), range(len(estimate)))))



def run_hmm_model(input_df, n_unique, A_df, Eta, n_iter = 10000, 
                        tol=1e-2, verbose = False, params = 'e', init_params = ''):
    '''
        Runs the hmm model and returns the predicted results, score and model 

            input_df : The dataframe of keypresses 

            n_unique : number of unqique chars 


            A_df : Dataframe of trasnmission matrix 

            Eta : Emissions matrix 

            n_iter : Max number of iterations for hmm

            tol : The value to stop the hmm model if score does not improve by more than this 

            verbose : Whether or not to print out 

            params : Parameters to tune 

            init_params : Paramters to initialize
    '''
    # Propotion of characters starting words in english 
    char_counts = get_char_counts()

    # Construct model 
    hmm = MultinomialHMM(n_components=n_unique, startprob_prior=np.append(0, char_counts.values), 
               transmat_prior=A_df.values, algorithm='viterbi', 
               random_state=None, n_iter=n_iter, tol=tol, 
               verbose=verbose, params=params, init_params=init_params)
    
    # Set values 
    hmm.emissionprob_ = Eta
    hmm.transmat_ = A_df.values
    hmm.startprob_ = np.append(0, char_counts.values)

    # Feed in the clusters as the expected output
    model_input = input_df['cluster'].values
    
    # Reshape    
    if len(model_input.shape) == 1:
        model_input = model_input.reshape((len(model_input), 1))
    
    # Fit the model
    hmm = hmm.fit(model_input)

    # Score model
    score, results = hmm.decode(model_input)

    return score, results, hmm  

def run_hmm(input_df, text, num_clusters, t_smooth = 1, verbose = True, 
                                    do_all = False, tol = 1e-2):
    '''
        Master function to run the hmm model 

            input_df : The dataframe of keypresses 

            text : targeted output

            num_clusters : Number of kmeans clusters

            t_smooth : Smoothing parameter for transmision matrix 

            verbose : Whether or not to print out 

            do_all : Whether or not to set emissions matrix to actual proportions 
                for all elements other than just space 

            tol : Limit for hmm model to stop at when score does not improve 
    '''
    # Build transmission matrix
    A_df, n_unique, unique_chars, id_to_char, char_to_id =  build_transmission_full(t_smooth)

    # Build emissions matrix 
    Eta = build_eta(input_df, unique_chars, num_clusters, do_all =  do_all,) 
    
    # Run the hmm
    score, results, hmm = run_hmm_model(input_df, n_unique, A_df, Eta, tol = tol, verbose = verbose)
    
    # Get the estimated result
    estimate = to_text(results, id_to_char)

    # Score result with and without spaces
    acc, acc_wospace = accuracy(estimate, text), accuracywospace(estimate, text)

    
    if verbose:
        print('Transmission smoothing:', t_smooth)
        print('Accuracy:', acc, 'Without spaces:', acc_wospace)
        print('guess:\n')
        print_color(estimate, text)

    return estimate, acc, acc_wospace, score, hmm

DEFAULT_WEBPAGES = ['https://www.gutenberg.org/files/11/11-h/11-h.htm', 'https://www.gutenberg.org/files/1342/1342-h/1342-h.htm',
               'https://www.gutenberg.org/files/46/46-h/46-h.htm', 'https://www.gutenberg.org/files/84/84-h/84-h.htm',
               'https://www.gutenberg.org/files/76/76-h/76-h.htm', 'https://www.gutenberg.org/files/844/844-h/844-h.htm',
               'https://www.gutenberg.org/files/53638/53638-h/53638-h.htm', 'https://www.gutenberg.org/files/2542/2542-h/2542-h.htm',
               'https://www.gutenberg.org/files/1400/1400-h/1400-h.htm', 'https://www.gutenberg.org/files/98/98-h/98-h.htm',
               'https://www.gutenberg.org/files/74/74-h/74-h.htm', 'https://www.gutenberg.org/files/53641/53641-h/53641-h.html',
               'https://www.gutenberg.org/files/1232/1232-h/1232-h.htm', 'https://www.gutenberg.org/files/1661/1661-h/1661-h.htm',
               'https://www.gutenberg.org/files/345/345-h/345-h.htm', 'https://www.gutenberg.org/files/160/160-h/160-h.htm',
               'https://www.gutenberg.org/files/5200/5200-h/5200-h.htm', 'http://www.gutenberg.org/cache/epub/30254/pg30254.html',
               'https://www.gutenberg.org/files/1952/1952-h/1952-h.htm', 'https://www.gutenberg.org/files/2600/2600-h/2600-h.htm',
               'https://www.gutenberg.org/files/174/174-h/174-h.htm', 'https://www.gutenberg.org/files/2701/2701-h/2701-h.htm']  


def scrape(webpages = DEFAULT_WEBPAGES):
    '''
        Scrapes a list of webpages  

    '''
    # Dictioanry of result for each webpage 
    results = {}

    # String to hold all text
    raw_text_all = ''

    # Chars to keep
    unique_chars = [' '] + map(lambda x : chr(x + ord('a')), range(26))

    for page in tqdm(webpages):

        # Grab the text from the page
        html = urllib.urlopen(webpage).read()
        soup = BeautifulSoup(html, 'html.parser')
        texts = soup.findAll(text=True)

        #  Only keep the visible portion
        def visible(element):
            if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
                return False
            elif re.match('<!--.*-->', str(element.encode('utf-8'))):
                return False
            return True

        # Grab the raw text and filter for our chars 
        raw_text = filter(lambda x : x in unique_chars, ' '.join(filter(visible, texts)).replace('\r\n', ' ').replace('\n', ' ').replace('\u2018', '').replace('\u2019', '').lower())
        raw_text = ' '.join(raw_text.split())
        
        # Save
        results[page] = raw_text
        raw_text_all += raw_text

        # Sleep
        time.sleep(2)
    return raw_text_all, results