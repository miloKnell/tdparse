import spacy
import json
import re
import nltk
import os

nlp=spacy.load('en_core_web_lg')

def write_conll(x,dataf):
    os.remove(dataf)
    with open(dataf,'w') as f:
        for text in x:
            off_index=0 #counts white space as word, count white spaces and subtract from position
            doc=nlp(text.strip())
            for token in doc:
                if token.is_space:
                    off_index+=1
                    continue
                if 'targetinpt' in token.lower_:
                    txt='targetinpt'
                else:
                    txt=token.lower_
                feats = (str(token.i-off_index),txt,'_',token.tag_,'_','_',str(token.head.i-off_index),token.dep_)
                f.write('    '.join(feats))
                f.write('\n')
                #position, token, _, tag, _, _, parser, rel

def write_sentihood(raw_x,raw_y,dataf):
    os.remove(dataf)
    with open(dataf,'w') as f:
        for x,y in zip(raw_x,raw_y):
            f.write(x.strip())
            f.write('\n')
            f.write('TARGETINPT')
            f.write('\n')
            f.write(str(y))
            f.write('\n')

def read_sentihood(file):
    with open(file,'r') as f:
        data=json.load(f)
    x = []
    y = []
    for entry in data:
        aspects = [(e['target_entity'],e['sentiment']) for e in entry['opinions'] if e['aspect']=='general']
        if len(aspects) == 0:
            continue
        x.append(entry['text'])
        y.append(aspects)
    labels = []
    new_x = []
    new_y=[]
    for text,entry in zip(x,y):
        for label in entry:
            new_x.append(re.sub(label[0],'TARGETINPT',text))
            #new_y.append(label)
            if label[1]=='Positive':
                labels.append(1)
            elif label[1]=='Negative':
                labels.append(0)
            else:
                print('Error on {}'.format(text))
    return new_x,labels#,new_y
'''
        
        if len(entry) == 1:
            if entry[0][1]=='Positive': #first entry only
                labels.append(1)
            elif entry[0][1] == 'Negative':
                labels.append(0)
            new_x.append(text)
            new_y.append(entry)
        elif len(entry)==2:
            if entry[0][1]=='Positive': #do first entry
                labels.append(1)
            elif entry[0][1] == 'Negative':
                labels.append(0)
            new_x.append(text)
            new_y.append(entry)
            
            if entry[1][1]=='Positive': #do second entry
                labels.append(1)
            elif entry[1][1] == 'Negative':
                labels.append(0)
            new_x.append(text)
            new_y.append(entry)
            #new_x.append(re.sub('LOCATION1|LOCATION2',lambda m: 'LOCATION1' if m.group() =='LOCATION2' else 'LOCATION2',text)) #change loc 1 and loc 2 for second entry
        else:
            print('Error in splitting inputs')'''
    #new_x=[fix(x) for x in new_x]
    



def fix(text):
    tok = nltk.tokenize.word_tokenize(text)
    if 'location1' not in [t.lower for t in tok]:
        for i,t in enumerate(tok):
            if '[target]' in t.lower():
                tok[i]='[target]'
    return ' '.join(tok)


x,y=read_sentihood('sentihood-train.json')
write_sentihood(x,y,r'C:\Users\milok\tdparse\data\sentihood\training')
write_conll(x,r'C:\Users\milok\tdparse\data\sentihood\parses\sentihood.train.conll')

x,y=read_sentihood('sentihood-test.json')
write_sentihood(x,y,r'C:\Users\milok\tdparse\data\sentihood\testing')
write_conll(x,r'C:\Users\milok\tdparse\data\sentihood\parses\sentihood.test.conll')

r'''with open(r'C:\Users\milok\tdparse\data\sentihood\parses\sentihood.train.conll','r') as f:
	for i,line in enumerate(f):
		con.append(line.strip().split('    '))'''
#x[397-8]

#tr: 104
