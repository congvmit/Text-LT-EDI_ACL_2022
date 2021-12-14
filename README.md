# Detecting Signs of Depression from Social Media Text-LT-EDI@ACL 2022

Author: Minh-Cong Vo

## 1 - Introduction

#### 1.1 - Target

Depression is a common mental illness that involves sadness and lack of interest in all day-to-day activities [1][2]. Detecting depression is important since it has to be observed and treated at an early stage to avoid severe consequences [3]. DepSign-LT-EDI@ACL-2022 aims to detect the signs of depression of a person from their social media postings wherein people share their feelings and emotions. Given social media postings in English, the system should classify the signs of depression into three labels namely “not depressed”, “moderately depressed”, and “severely depressed”.

Challenge Website: https://competitions.codalab.org/competitions/36410#learn_the_details-overview

#### 1.2 - Metrics

Classification system’s performance will be measured in terms of macro averaged Precision, macro averaged Recall and macro averaged F-Score across all the classes.

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

### 1.4 - References

[1] Institute of Health Metrics and Evaluation. Global Health Data Exchange (GHDx). http://ghdx.healthdata.org/gbd-results-tool?params=gbd-api-2019-permalink/d780dffbe8a381b25e1416884959e88b \
[2] Evans-Lacko S, Aguilar-Gaxiola S, Al-Hamzawi A, et al. Socio-economic variations in the mental health treatment gap for people with anxiety, mood, and substance use disorders: results from the WHO World Mental Health (WMH) surveys. Psychol Med. 2018;48(9):1560-1571.\
[3] Losada, D. E., Crestani, F., & Parapar, J. (2017, September). eRISK 2017: CLEF lab on early risk prediction on the internet: experimental foundations. In the International Conference of the Cross-Language Evaluation Forum for European Languages (pp. 346-360). Springer, Cham.

## 2 - How to run

```bash
python train-longformer-4096.py --is-training
python train-bert-512.py --is-training
```
## 3 - Current Results

<table>
  <tr>
    <th>Model</th>
    <th><center>Dev1-Loss</center></th>
    <th><center>Dev1-Acc</center></th>
    <th><center>Dev2-Loss</center></th>
    <th><center>Dev2-Acc</center></th>
    <th><center>Test-Acc</center></th>
  </tr>
  <tr>
    <td>Longformer@4096</td>
    <td><center>0.378</center></td>
    <td><center>0.878</center></td>
    <td><center>1.105</center></td>
    <td><center>0.568</center></td>
    <td><center>-</center></td>
  </tr>

  <tr>
    <td>multinomialNB</td>
    <td><center>-</center></td>
    <td><center>-</center></td>
    <td><center>-</center></td>
    <td><center>-</center></td>
    <td><center>-</center></td>
  </tr>
</table>

## 4 - References

- Website: https://competitions.codalab.org/competitions/36410
- https://towardsdatascience.com/how-to-fine-tune-bert-with-pytorch-lightning-ba3ad2f928d2?p=91965d1e6425
- https://github.com/prateekjoshi565/Fine-Tuning-BERT/blob/master/Fine_Tuning_BERT_for_Spam_Classification.ipynb
