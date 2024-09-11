# import required library
import re 
import string
import contractions
import streamlit as st
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords,wordnet
from nltk.stem import WordNetLemmatizer,LancasterStemmer
from nltk.classify import apply_features
from joblib import load
from textblob import TextBlob
from scipy.sparse import hstack
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load the TF-IDF vectorizer and all hate speech detection model
tfidf_loaded = load('tfidf_vectorizer.joblib')
linear_r_no_polarity_loaded = load('linear_regression_model_no_polarity.joblib')
linear_r_with_polarity_loaded = load('linear_regression_model_with_polarity.joblib')
logistic_r_loaded = load('logistic_regression_model.joblib')
knn_loaded = load('KNN_model.joblib')
svm_loaded = load('SVC_model.joblib')

# Main Func or start of the Web Application
def main():
    # Set Title of the Web
    st.title(":rainbow[Hate Speech Detection Web App]")
    # Sidebar for navigation
    st.sidebar.title("Input Options")
    option = st.sidebar.selectbox("Choose Method To Input Text Data/Comments", ["Manually Enter Text", "Upload File"])

    # Display the table of hate speech score information
    hate_speech_score_type = pd.DataFrame({
        'Range of Hate Speech Score ' : ['hate speech score > 0.5','-1 <= hate speech score <= 0.5','hate speech score < -1'],
        'Type of Text/Comment ' : ['hate speech','neutral speech or ambiguous speech','non-hate speech or supportive speech']
    })  
    st.table(hate_speech_score_type)
    
    # Option to manually enter text
    if option == "Manually Enter Text":       
        # Text box for user input
        st.subheader(":orange[Enter a sentence to check it's hate speech score and determine if it's hate speech or not]\n(Higher Hate Speech Score = More Hateful)")         
        user_input = st.text_input("Your Sentence:")

        # Predict button
        if st.button('Predict'):
            if user_input:  # Check if the input is not empty
                processed_user_input = preprocess_and_clean([user_input]) # Preprocess text
                if processed_user_input[0] == "":
                    st.error("Please enter another sentence for prediction.") #display error messages if the processed text is empty
                else:
                    predict_and_display([user_input],processed_user_input)  # Single sentence prediction
            else:
                st.error("Please enter a sentence for prediction.")
    else:  # Option to upload file
        st.subheader(":green[Select a text(.txt) or a csv(.csv) file to upload and check the hate speech score]")
        uploaded_file = st.file_uploader("Choose a file to upload", type=['txt', 'csv'])
        if uploaded_file is not None:
            if uploaded_file.type == "text/csv" or uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:  # Assume text file
                data = pd.read_table(uploaded_file, header=None, names=['text'])

            # Check if the file has content
            if not data.empty:
                sentences = data['text'].tolist()
                processed_sentences = preprocess_and_clean(sentences) # Preprocess text in the file
                predict_and_display(sentences,processed_sentences)  # File-based prediction

#preprocess the text input
def preprocess_and_clean(sentences):
    #remove any links or url (e.g. https://123abc.com)
    sentences_df = pd.DataFrame(sentences,columns=["Sentences"])
    sentences_df['Sentences'] = sentences_df['Sentences'].apply(lambda x: re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|''[!*,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','', x))

    #remove any mention in the text that start with @ (e.g @username)
    sentences_df['Sentences'] = sentences_df['Sentences'].apply(lambda x: re.sub(r'@\S+','', x))

    #remove any hashtag in the text that start with #
    sentences_df['Sentences'] = sentences_df['Sentences'].apply(lambda x: re.sub(r'#\S+','', x))

    #change all word to lowercase
    sentences_df['Sentences'] = sentences_df['Sentences'].apply(lambda x: x.lower())

    #create a dictionary of list of abbreviation and its original form
    #some popular abbreviations found on the Internet
    abbreviation_dict = {'dm': 'direct message',
                     'thx': 'thanks',
                     'plz': 'please',
                     'u': 'you',
                     'asap': 'as soon as possible',
                     'brb': 'be right back',
                     'diy': 'do it yourself',
                     'btw': 'by the way',
                     'r': 'are',
                     'stfu' : 'shut the fuck up',
                     'wtf' : 'what the fuck',
                     'noob' : 'newbie',
                     'eta': 'estimated time of arrival',
                     'nvm' : 'nevermind',
                     'fb' : 'facebook',
                     'ig' : 'instagram',
                     'fyi' : 'for your information',
                     'imo' : 'in my opinion',
                     'lol' : 'laughing out loud',
                     'jk': 'just kidding',
                     'lmao' : 'laughing my ass off',
                     'idc' : 'i don\'t care',
                     'zzz' : 'sleeping, bored, tired',
                     'tbh' : 'to be honest',
                     'pov' : 'point of view',
                     'smh' : 'shaking my head',
                     'irl' : 'in real life',
                     'j4f' : 'just for fun',
                     'idk' : 'i don\'t know',
                     'ppl' : 'people'
                    }
    #removing abbreviations and replace with original words
    sentences_df['Sentences'] = sentences_df['Sentences'].str.replace('[...…]','').str.split().apply(lambda x: ' '.join([abbreviation_dict.get(e, e) for e in x]))

    #remove punctuation(except apostrophes['])
    my_punctuation = string.punctuation.replace("'", "")
    sentences_df['Sentences'] = sentences_df['Sentences'].apply(lambda x: re.sub('[%s]' % re.escape(my_punctuation), ' ', x)) 
    
    #remove contractions (e.g remove We're and change to We are)
    sentences_df['Sentences'] = sentences_df['Sentences'].apply(lambda x: ' '.join([contractions.fix(word) for word in x.split()]))
    
    #remove apostrophe that are still remained after removing contractions
    sentences_df['Sentences'] = sentences_df['Sentences'].apply(lambda x: x.replace("'","")) 
    
    #remove alphanumeric
    sentences_df['Sentences'] = sentences_df['Sentences'].apply(lambda x: re.sub(r"""\w*\d\w*""", ' ', x)) 
    
    #change multiple space characters between words into one space character only
    sentences_df['Sentences'] = sentences_df['Sentences'].apply(lambda x: re.sub(r'\s+', ' ', x))
    
    #remove leading and trailing whitespace character
    sentences_df['Sentences'] = sentences_df['Sentences'].apply(lambda x: re.sub(r'^\s+|\s+?$','', x))

    #create stopword object
    stop = stopwords.words('english')
    #remove stopwords
    sentences_df['Sentences'] = sentences_df['Sentences'].apply(lambda x : ' '.join([word for word in x.split() if word not in (stop)]))

    #define a function that can assign the wordnet pos tag for every word based on the nltk pos tag for wordnet lemmatization purpose
    def assign_wordnet_pos_tag(nltk_pos_tag):
        if nltk_pos_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_pos_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_pos_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_pos_tag.startswith('R'):
            return wordnet.ADV
        else:          
            return None
    
    #apply nltk pos tag on every word in the input
    pos_tagged_text = sentences_df['Sentences'].apply(lambda x : nltk.pos_tag(nltk.word_tokenize(x)) )
    
    #convert the nltk pos tag to wordnet pos tag for lemmatization
    wordnet_tagged_text = pos_tagged_text.apply(lambda x: list(map(lambda y: (y[0], assign_wordnet_pos_tag(y[1])), x)))
    
    # create lemmatizer object
    lemmatizer = WordNetLemmatizer()
    #lemmatize each word in each sentence based on the wordnet pos tag
    lemmatized_sentence_list = []
    for sentence in wordnet_tagged_text:
        lemmatized_sentence = []
        for word, tag in sentence:
            if tag is None:
                lemmatized_sentence.append(lemmatizer.lemmatize(word)) #if no tag is found, lemmatize word with default tag
            else:        
                lemmatized_sentence.append(lemmatizer.lemmatize(word, tag)) #if there is a wordnet tag, lemmatize word with the tag
        lemmatized_sentence = " ".join(lemmatized_sentence)        #join all word into sentence
        lemmatized_sentence_list.append(lemmatized_sentence)       #append into the list variable of lemmatized sentence
    
    #convert the lemmatized sentences in list into the dataframe 
    sentences_df['Sentences'] = pd.DataFrame(lemmatized_sentence_list)
    
    # create stemming object
    stemmer = LancasterStemmer()
    # perform stemming on each word
    sentences_df['Sentences'] = sentences_df['Sentences'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()])) 

    return sentences_df["Sentences"].tolist()

def predict_and_display(unprocessed_sentences,sentences):
    # Transform the sentences
    transformed_sentences = tfidf_loaded.transform(sentences)

    # use textblob library to determine polarity score and transform the sentence
    sentences_df = pd.DataFrame(sentences)
    polarity_score = sentences_df.iloc[:, 0].apply(lambda x: TextBlob(x).sentiment.polarity)
    transformed_sentences_with_polarity = hstack([transformed_sentences, polarity_score.values.reshape(-1, 1)]) 
    
    # Make predictions for hate speech score (without polarity score as feature)
    score_results_no_polarity = linear_r_no_polarity_loaded.predict(transformed_sentences)
    text_type_no_polarity = []
    for x in score_results_no_polarity:
        if x > 0.5 :
            text_type_no_polarity.append("hate speech")
        elif x >= -1 :
            text_type_no_polarity.append("neutral speech or ambiguous speech")
        else :
            text_type_no_polarity.append("non-hate speech or supportive speech")

    # Convert the text type list to DataFrame to plot graph
    text_type_no_polarity_df = pd.DataFrame(text_type_no_polarity, columns=['type'])
    text_type_no_polarity_count = text_type_no_polarity_df['type'].value_counts()
    text_type_no_polarity_count_df = pd.DataFrame([["score < -1",0],["-1 <= score <= 0.5",0],["score > 0.5",0]], columns=['Type','Count'])
    for i in range(text_type_no_polarity_count.size):
        if text_type_no_polarity_count.index[i] == "non-hate speech or supportive speech":
            text_type_no_polarity_count_df["Count"].iloc[0] = text_type_no_polarity_count.values[i]
        if text_type_no_polarity_count.index[i] == "neutral speech or ambiguous speech":
            text_type_no_polarity_count_df["Count"].iloc[1] = text_type_no_polarity_count.values[i]
        if text_type_no_polarity_count.index[i] == "hate speech":
            text_type_no_polarity_count_df["Count"].iloc[2] = text_type_no_polarity_count.values[i]
    
    # Make predictions for hate speech score (with polarity score as feature)
    score_results_with_polarity = linear_r_with_polarity_loaded.predict(transformed_sentences_with_polarity)
    text_type_with_polarity = []
    for x in score_results_with_polarity:
        if x > 1 :
            text_type_with_polarity.append("hate speech")
        elif x >= -1 :
            text_type_with_polarity.append("neutral speech or ambiguous speech")
        else :
            text_type_with_polarity.append("non-hate speech or supportive speech")

    # Convert the text type list to DataFrame to plot graph
    text_type_with_polarity_df = pd.DataFrame(text_type_with_polarity, columns=['type'])
    text_type_with_polarity_count = text_type_with_polarity_df['type'].value_counts()
    text_type_with_polarity_count_df = pd.DataFrame([["score < -1",0],["-1 <= score <= 0.5",0],["score > 0.5",0]], columns=['Type','Count'])
    for i in range(text_type_with_polarity_count.size):
        if text_type_with_polarity_count.index[i] == "non-hate speech or supportive speech":
            text_type_with_polarity_count_df["Count"].iloc[0] = text_type_with_polarity_count.values[i]
        if text_type_with_polarity_count.index[i] == "neutral speech or ambiguous speech":
            text_type_with_polarity_count_df["Count"].iloc[1] = text_type_with_polarity_count.values[i]
        if text_type_with_polarity_count.index[i] == "hate speech":
            text_type_with_polarity_count_df["Count"].iloc[2] = text_type_with_polarity_count.values[i]

    # Make predictions for text target
    logistic_r_target_results = logistic_r_loaded.predict(transformed_sentences)
    knn_target_results = knn_loaded.predict(transformed_sentences)
    svm_target_results = svm_loaded.predict(transformed_sentences)

    # Display subheader for better presentation
    st.subheader(":violet[Hate Speech Score Prediction]") 
    
    # Combine the inputs and hate speech score predictions into a DataFrame
    score_results_no_polarity_df = pd.DataFrame({
        'Original Input': unprocessed_sentences,
        'Processed Input': sentences,
        'Predicted Hate Speech Score': score_results_no_polarity,
        'Type Or Category Of Input Text' : text_type_no_polarity
    })

    # Tabulate and display the results
    with st.expander("Show/Hide Prediction Table (Result With Hate Speech Score Only)"):
        st.table(score_results_no_polarity_df)

    # Combine the inputs, polarity score and hate speech score predictions into a DataFrame
    score_results_with_polarity_df = pd.DataFrame({
        'Original Input': unprocessed_sentences,
        'Processed Input': sentences,
        'Polarity Score' : polarity_score,
        'Predicted Hate Speech Score': score_results_with_polarity,
        'Type Or Category Of Input Text' : text_type_with_polarity
    })

    # Tabulate and display the results
    with st.expander("Show/Hide Prediction Table (Result With Polarity Score And Hate Speech Score)"):
        st.table(score_results_with_polarity_df)

    # Create bar chart
    st.write("Bar Chart Of Distribution Of Hate Speech Score Prediction(Model Trained Without Polarity Score):")
    fig, ax = plt.subplots()
    ax.bar(text_type_no_polarity_count_df['Type'],text_type_no_polarity_count_df['Count'],color=["cyan","magenta","blue"])
    ax.set_title("Bar Chart Of Distribution Of Hate Speech Score Prediction\n(Model Trained Without Polarity Score)")
    ax.set_xlabel("Hate Speech Score Range")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True,min_n_ticks=1))  # Ensure y-axis has integer ticks
    ax.set_ylim(0)
    for i in range (3):
        ax.text(i,text_type_no_polarity_count_df["Count"].iloc[i],text_type_no_polarity_count_df["Count"].iloc[i],ha = 'center')
    st.pyplot(fig)

    # Create bar chart
    st.write("Bar Chart Of Distribution Of Hate Speech Score Prediction(Model Trained With Polarity Score):")
    fig, ax = plt.subplots()
    ax.bar(text_type_with_polarity_count_df['Type'],text_type_with_polarity_count_df['Count'],color=["cyan","magenta","blue"])
    ax.set_title("Bar Chart Of Distribution Of Hate Speech Score Prediction\n(Model Trained With Polarity Score)")
    ax.set_xlabel("Hate Speech Score Range")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True,min_n_ticks=1))  # Ensure y-axis has integer ticks
    ax.set_ylim(0)
    for i in range (3):
        ax.text(i,text_type_with_polarity_count_df['Count'].iloc[i],text_type_with_polarity_count_df['Count'].iloc[i],ha = 'center')
    st.pyplot(fig)

    # Display subheader for better presentation
    st.subheader(":violet[Text Target Prediction]")
    
    # Combine all the target prediction into one DataFrame (logistic regression)
    logisitic_r_target_results_df = pd.DataFrame({
        'Processed Input': sentences,
        'Target Race': logistic_r_target_results[:,0],
        'Target Religion': logistic_r_target_results[:,1],
        'Target Origin': logistic_r_target_results[:,2],
        'Target Gender': logistic_r_target_results[:,3],
        'Target Sexuality': logistic_r_target_results[:,4],
        'Target Age': logistic_r_target_results[:,5],
        'Target Disability': logistic_r_target_results[:,6]
    })

    # Tabulate and display the results
    with st.expander("Show/Hide Prediction Table (Logistic Regression Model)"):
        st.table(logisitic_r_target_results_df)

    # Combine all the target prediction into one DataFrame (KNN)
    knn_target_results_df = pd.DataFrame({
        'Processed Input': sentences,
        'Target Race': knn_target_results[:,0],
        'Target Religion': knn_target_results[:,1],
        'Target Origin': knn_target_results[:,2],
        'Target Gender': knn_target_results[:,3],
        'Target Sexuality': knn_target_results[:,4],
        'Target Age': knn_target_results[:,5],
        'Target Disability': knn_target_results[:,6]
    })

    # Tabulate and display the results
    with st.expander("Show/Hide Prediction Table (KNN Model)"):
        st.table(knn_target_results_df)

    # Combine all the target prediction into one DataFrame (SVM)
    svm_target_results_df = pd.DataFrame({
        'Processed Input': sentences,
        'Target Race': svm_target_results[:,0],
        'Target Religion': svm_target_results[:,1],
        'Target Origin': svm_target_results[:,2],
        'Target Gender': svm_target_results[:,3],
        'Target Sexuality': svm_target_results[:,4],
        'Target Age': svm_target_results[:,5],
        'Target Disability': svm_target_results[:,6]
    })

    # Tabulate and display the results
    with st.expander("Show/Hide Prediction Table (SVM Model)"):
        st.table(svm_target_results_df)

    #--------------------------- Visualization Of The Result ---------------------------
    # Label for x-axis of bar chart
    x = np.array(["Race", "Religion", "Origin", "Gender", "Sexuality", "Age", "Disability"])

    #------------ Logistic Regression Model ------------
    # Convert result to dataframe
    logistic_r_result_df = pd.DataFrame(logistic_r_target_results)
    logistic_r_result_y = np.array([len(logistic_r_result_df[logistic_r_result_df[0]==True]),len(logistic_r_result_df[logistic_r_result_df[1]==True]),len(logistic_r_result_df[logistic_r_result_df[2]==True]),len(logistic_r_result_df[logistic_r_result_df[3]==True]),len(logistic_r_result_df[logistic_r_result_df[4]==True]),len(logistic_r_result_df[logistic_r_result_df[5]==True]),len(logistic_r_result_df[logistic_r_result_df[6]==True])])
    
    # Display barchart of predictions
    st.write("Bar Chart Of Distribution Of Prediction (Logistic Regression Model):")
    fig, ax = plt.subplots()
    ax.bar(x,logistic_r_result_y,color=["cyan","magenta","blue","orange","green","olive","darkcyan"])
    ax.set_title("Bar Chart Of Distribution Of Text Target Type (Logistic Regression Model)")
    ax.set_xlabel("Target Type")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True,min_n_ticks=1))  # Ensure y-axis has integer ticks
    ax.set_ylim(0)
    for i in range (7):
        ax.text(i,logistic_r_result_y[i],logistic_r_result_y[i],ha = 'center')
    st.pyplot(fig)

    #------------ KNN Model ------------
    # Convert result to dataframe
    knn_target_results_df = pd.DataFrame(knn_target_results)
    knn_target_results_y = np.array([len(knn_target_results_df[knn_target_results_df[0]==True]),len(knn_target_results_df[knn_target_results_df[1]==True]),len(knn_target_results_df[knn_target_results_df[2]==True]),len(knn_target_results_df[knn_target_results_df[3]==True]),len(knn_target_results_df[knn_target_results_df[4]==True]),len(knn_target_results_df[knn_target_results_df[5]==True]),len(knn_target_results_df[knn_target_results_df[6]==True])])
    
    # Display barchart of predictions
    st.write("Bar Chart Of Distribution Of Prediction (KNN Model):")
    fig, ax = plt.subplots()
    ax.bar(x,knn_target_results_y,color=["cyan","magenta","blue","orange","green","olive","darkcyan"])
    ax.set_title("Bar Chart Of Distribution Of Text Target Type (KNN Model)")
    ax.set_xlabel("Target Type")
    ax.set_ylabel("Count")    
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True,min_n_ticks=1))  # Ensure y-axis has integer ticks
    ax.set_ylim(0)
    for i in range (7):
        ax.text(i,knn_target_results_y[i],knn_target_results_y[i],ha = 'center')
    st.pyplot(fig)

    #------------ SVM Model ------------
    # Convert result to dataframe
    svm_target_results_df = pd.DataFrame(svm_target_results)
    svm_target_results_y = np.array([len(svm_target_results_df[svm_target_results_df[0]==True]),len(svm_target_results_df[svm_target_results_df[1]==True]),len(svm_target_results_df[svm_target_results_df[2]==True]),len(svm_target_results_df[svm_target_results_df[3]==True]),len(svm_target_results_df[svm_target_results_df[4]==True]),len(svm_target_results_df[svm_target_results_df[5]==True]),len(svm_target_results_df[svm_target_results_df[6]==True])])
    
    # Display barchart of predictions
    st.write("Bar Chart Of Distribution Of Prediction (SVM Model):")
    fig, ax = plt.subplots()
    ax.bar(x,svm_target_results_y,color=["cyan","magenta","blue","orange","green","olive","darkcyan"])
    ax.set_title("Bar Chart Of Distribution Of Text Target Type (SVM Model)")
    ax.set_xlabel("Target Type")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True,min_n_ticks=1))  # Ensure y-axis has integer ticks
    ax.set_ylim(0)
    for i in range (7):
        ax.text(i,svm_target_results_y[i],svm_target_results_y[i],ha = 'center')
    st.pyplot(fig)


if __name__ == '__main__':
    main()
