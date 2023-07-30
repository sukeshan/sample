import fasttext
import pandas as pd
import re
from utils import preprocess_text



def process_data():
    jd = pd.read_csv(r'DB\Skill_matching.csv')
    jd['job role'] = list(map(lambda x : ' '.join(x.replace(' / ','/').replace(';','/').replace('& ','/').replace(' &','/').replace(' & ','/').replace('(','/').replace(')','/').replace('- ','/').replace(' -','/').replace(' - ','/').replace('/ ','/').replace('  /','/').replace(' /  ','/').replace('  /  ','/').replace('(','/').replace(')','/').replace('-','/').replace('"','').replace(' ','_').lower().split('/')) ,jd['job role']  ))
    jd['job keyword'] = list(map(lambda i : i.replace(' ','_').lower(),jd['job keyword'] ))
    jd['skills raw'] = list(map(lambda i : i.replace('[','').replace(']','').replace('-','').replace(' - ','').replace("'",'').replace('"','').replace(';',',').replace(', ',',').replace(' ,',',').replace(' , ',',').replace(' ','_').replace(',',' ').lower(),jd['skills raw'] ))
    jd['job description raw'] = jd['job description raw'].apply(preprocess_text) 
    jd['merge'] = jd['job keyword'] + ' '+ jd['job role'] + ' '+  jd['job keyword'] + ' '+ jd['job description raw'] + ' '+ jd['skills raw']
    jd.to_csv(r'DB\data.txt' ,header = None ,columns = ['a'] ,index = False)

def train():
    process_data()
    model = fasttext.train_unsupervised(r'DB\data.txt')
    model.save(r'DB\weigth.bin')