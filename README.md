# Detecting Signs of Depression from Social Media Text-LT-EDI@ACL 2022

Author: Minh-Cong Vo

## I - Introduction

#### 1.1 - Target

Depression is a common mental illness that involves sadness and lack of interest in all day-to-day activities [1][2]. Detecting depression is important since it has to be observed and treated at an early stage to avoid severe consequences [3]. DepSign-LT-EDI@ACL-2022 aims to detect the signs of depression of a person from their social media postings wherein people share their feelings and emotions. Given social media postings in English, the system should classify the signs of depression into three labels namely “not depressed”, “moderately depressed”, and “severely depressed”.

Challenge Website: https://competitions.codalab.org/competitions/36410#learn_the_details-overview

#### 1.2 - Metrics

Classification system’s performance will be measured in terms of **macro averaged Precision**, **macro averaged Recall** and **macro averaged F-Score** across all the classes.

#### 1.3 - Timelines

- ~~Task announcement: Nov 20, 2021~~
- ~~Release of Training data: Nov 20, 2021~~
- **Release of Test data: Jan 14, 2022**
- Run submission deadline: Jan 30, 2022
- Results declared: Feb 10, 2022
- Paper submission: March 10, 2022
- Peer review notification: March 26, 2022
- Camera-ready paper due: April 5, 2022
- Workshop Dates: May 26-28, 2022

## II - Related Works

### 2.1 - Text Preprocessing

[5, 6] show that people with depression tend to have common patterns of behaviour. Particularly, People talk more about relationships and life (e.g., friends, home, dating and health); become more concerned with themselves (use first person pronoun very often); use more emoticons (e.g., :(, :c); words of negative emotions (anger, sad, anxiety, etc.) anbd denial terms (e.g., no, none, or never); constantly remember the past and worry about the future.

### 2.2 - Emotion Inference

## III - References

[1] Institute of Health Metrics and Evaluation. Global Health Data Exchange (GHDx). http://ghdx.healthdata.org/gbd-results-tool?params=gbd-api-2019-permalink/d780dffbe8a381b25e1416884959e88b \
[2] Evans-Lacko S, Aguilar-Gaxiola S, Al-Hamzawi A, et al. Socio-economic variations in the mental health treatment gap for people with anxiety, mood, and substance use disorders: results from the WHO World Mental Health (WMH) surveys. Psychol Med. 2018;48(9):1560-1571.\
[3] Losada, D. E., Crestani, F., & Parapar, J. (2017, September). eRISK 2017: CLEF lab on early risk prediction on the internet: experimental foundations. In the International Conference of the Cross-Language Evaluation Forum for European Languages (pp. 346-360). Springer, Cham. \
[4] [Figueredo, José, and Rodrigo Calumby. "On Text Preprocessing for Early Detection of Depression on Social Media." Anais do XX Simpósio Brasileiro de Computação Aplicada à Saúde. SBC, 2020.](https://sol.sbc.org.br/index.php/sbcas/article/view/11504/11367) \
[5] Nakamura, T., Kubo, K., Usuda, Y., and Aramaki, E. (2014). Defining patients withdepressive disorder by using textual information. In 2014 AAAI Spring Symposia, Stanford University, Palo Alto, California, USA, March 24-26, 2014. \
[6] Vedula, N. and Parthasarathy, S. (2017). Emotional and linguistic cues of depression fromsocial media. InProceedings of the 2017 International Conference on Digital Health,London, United Kingdom, July 2-5, 2017, pages 127–136.

## IV - How to run

```bash
python preprocessing_v2.py
python train-bert-512.py
python train-roberta-512.py
```

## V - Current Results

<!-- <style>
  .double {
    border-top: 4px double #999;
    padding: 10px 0;
    }
</style> -->

<table>
  <tr>
    <th>Model</th>
    <th>Ver</th>
    <th><center>Macro-Precision</center></th>
    <th><center>Macro-Recall</center></th>
    <th><center>Macro-F1</center></th>
  </tr>
  <tr class="double">
    <td>BERT@512</td>
    <td><center>0</center></td>
    <td><center>0.5434</center></td>
    <td><center>0.5219</center></td>
    <td><center>0.4971</center></td>
  </tr>
  <tr>
    <td>BERT@512</td>
    <td><center>4</center></td>
    <td><center>0.5089</center></td>
    <td><center>0.5346</center></td>
    <td><center>0.4643</center></td>
  </tr>
  <tr class="double">
    <td>RoBERTa@512</td>
    <td><center>0</center></td>
    <td><center>0.5100</center></td>
    <td><center>0.5676</center></td>
    <td><center>0.5067</center></td>
  </tr>
  <tr class="double">
    <td>LSTM</td>
    <td><center>-</center></td>
    <td><center>-</center></td>
    <td><center>-</center></td>
    <td><center>-</center></td>
  </tr>
</table>

## Code/Challenges References

- Challenge Website: https://competitions.codalab.org/competitions/36410
- https://towardsdatascience.com/how-to-fine-tune-bert-with-pytorch-lightning-ba3ad2f928d2?p=91965d1e6425
- https://gaussian37.github.io/dl-pytorch-lr_scheduler/#cosineannealingwarmrestarts-1
- https://github.com/prateekjoshi565/Fine-Tuning-BERT/blob/master/Fine_Tuning_BERT_for_Spam_Classification.ipynb
- https://colab.research.google.com/github/DhavalTaunk08/Transformers_scripts/blob/master/Transformers_multilabel_distilbert.ipynb
- https://colab.research.google.com/github/digitalepidemiologylab/covid-twitter-bert/blob/master/CT_BERT_Huggingface_(GPU_training).ipynb#scrollTo=jvvPnOFQH2pR
- Text-based Emotion Classification Survey: https://onlinelibrary.wiley.com/doi/full/10.1002/eng2.12189
