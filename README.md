
# 🗞️ Fake-News-Detection-From-Real-Time-Data

Spreading fake news is nothing new in our modern society. We've had this problem for a long time, but now, with the increasing popularity of social media, people are facing a load of different problems as a result of the spread of fake news. Manually checking whether news is fake or not is difficult, time-consuming, and expensive, as we all know. It is frequently influenced as a result of various news checks conducted by people, and its validity may be called into doubt. Nowadays, the spread of fake news or information has a negative impact on society. Because of the spread of fake news, we sometimes get the impression that a lot of it is true. Fake news detection is a challenging task, as a result researchers have been exploring automated solutions to these problems in the past few years. All of these methods for automatic detection will improve the false news detection process while saving time and effort. As a result, we proposed a system to detect fake news. However, the amount of data on the web or on social media is rapidly increasing, making it extremely difficult to determine whether news is fake or not by looking at all data, which is time consuming, so we used classification techniques to classify large amounts of data. We present our deep learning-based techniques to detect fake news. We are using Long Short-Term Memory (LSTM), Bidirectional LSTM and Gated Recurrent Unit (GRU) models for fake news detection using stock data provided by Kaggle as well as real time data from twitter API.


## 📨 Keywords

Fake News, Detection, Deep Learning, Classification, Long Short Term Memory (LSTM), Bidirectional LSTM And Gated Recurrent Unit (GRU), Tweepy, NLP.

## 🛠 Languages and Methods
Machine Learning, Deep Learning, PyTorch, Tensorflow, Keras, Neural Networks



## 📚 Related

There has been a lot of research on false news so far. False news has recently become a source of public concern. Many research articles on fake news have been published in a variety of languages. Machine learning methods and deep learning algorithms have been used by several academics to detect fake information. Girgis[13] suggested a classifier that can identify whether a piece of news is false or not based simply on its content, utilizing RNN method models (vanilla, GRU) and LSTMs to take a purely deep learning approach to the problem. They were using the LAIR dataset to point out the differences and analyze the outcomes. They discovered that the results are close, but that the GRU, with a score of (0.217), is the best of their findings, followed by LSTM (0.2166), and vanilla (0.215). They attempted to improve accuracy as a result of these discoveries by using a combined approach that combines GRU and CNN algorithms on the same data set. Pritika Bahad, Preeti Saxena , Raj Kamal[3] said the challenge of detecting fake news is a binary classification problem. The suggested approach detects fake news by detecting whether the content in the article is correct by measuring the bias of a published news article and analyzing the link between the title and the body of the article. The architecture of the system and the data processing approach utilized for the experiments are described in this section. [2] Bidirectional LSTM is a kind of Recurrent Neural Network (RNN) made up of two Long Short-Term Memory (LSTM) oriented in opposing directions. The architecture wants to expand the memory capacity of LSTMs[1] by providing context information from the past and future. A deep learning-based approach was used to differentiate fake news from real news. An LSTM neural network was used to generate the suggested model. In addition to the neural network, a mask word embedding was used for vector representation of textual words. Feature extraction and vectorization have also been done using tokenization. The N-grams concept is used to improve the proposed model. For covid-19, Shahi et al[10] used cross domains to gather articles regarding false news. They have technology that detects fake news in more than 40 languages across 105 countries in their library. As part of their research, they acquired data from a variety of internet sources. They categorized fake material regarding covid-19 using machine learning approaches, stopping the spread of misleading information during the early phases of the pandemic. Ignacio Palacio Marn et al[12] provided a unique occurrence in the taxonomy of the challenging continuity of fake and wrong information occurrences by using tagged information conveying fake news. For this study, they used text data classification, deep learning route and classification approaches. For NLP-related work, the BERT model is critical in resolving issues. S. P. Akula and N. Kamati[11] utilized Kaggle’s Fake News Detection Dataset to train the model, which incorporates social media news articles. In terms of accuracy and performance, they suggested Bi-directional LSTM model beats Convolutional Neural Networks (CNNs), Gated Recurrent Units (GRU), Unidirectional Long Short-Term Memory – Recurrent neural networks, and existing Bi-directional Long Short-Term Memory models.  F.Torgheh, B.Masoumi, and S.V.Shojaedini[14] suggested a false news detection approach based on the propagation path. In order to provide a more accurate outcome, the news distribution path was integrated with user characteristics that were involved in distributing the news. Finally, a deep learning technique was used to detect fake news using a combination of recurrent and convolutional networks. The results demonstrated that the proposed approach has the potential to improve the accuracy of false news detection when compared to many current strategies. Keya et al[15]. Mridha proposed a Convolutional Neural Network (CNN) and Gated Recurrent Unit (GRU) ensemble methodology with a pre-trained GloVe embedding method that obtained a test data accuracy of 98.71 percent. Long short-term memory (LSTM) and CNN are both trained using the same dataset and parameters. They also evaluated their suggested model on a testing dataset of English news, achieving a 98.94 percent accuracy. A range of assessment metrics are used to evaluate their model's performance, including accuracy, recall, precision, and the f1-score.
## 🔃Materials and Methods

In this section we have shown the data sets preparation and proposed model building. We used LSTM, Bi-LSTM, and GRU architectures with tensor flow 2.2.0 GPU for model training. However, before we could train the models, we needed to gather and review datasets. For data preparation, we followed the methods below. 


## Running Tests

Data-set Preparation :

Make use of Twitter-API 

The Twitter API allows users to interact and access public Twitter data. This twitter API allows us to connect to the Twitter platform and perform a search operation for the given keyword. The data needs to be prepared, once the tweets are obtained from the API, the tweets are analyzed by using sentiment analysis. Because the data gathered (i.e. tweets) from social media are not in a structured way to perform any type of analysis. Then, the gathered data is analyzed by using some techniques like sentiment analysis , corpus . 
Corpus is used to cleanse data in such ways like to remove hypertexts , emojis and any type of unwanted special characters, which makes the data useful for good predictions.

Data Preprocessing 

 Preprocessing a Twitter dataset entails removing all forms of extraneous data such as emojis, special characters, and excess blank spaces, among other things. It may also entail making format changes, deleting duplicate tweets, or deleting tweets with less than three characters.ax.

a.	Remove Stop words 

	For data processing, we used Stop Word. NLTK can delete the stop word and trace the catalogue of stop words in the corpus module for NLP work. It can be used in a sentence without changing its meaning.

b.	Spelling Checker

In this phase, we double-check our dataset's spelling. We used this method to correct a lot of spelling issues in the dataset.


c.	Remove Punctuation

There are a lot of unnecessary punctuations in this dataset. With this technique, we were able to get rid of them. It's a portion of our model's purification test.
