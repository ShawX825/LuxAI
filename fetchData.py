import re, codecs, os, pickle
from os import listdir
import urllib.request, json 

'''
# for loading files, use:
with open('data/{}.json'.format(eps),'r') as f:
    data = json.load(f)
'''

root = 'data/luxai-htmls'
url_root = 'https://www.kaggleusercontent.com/episodes/{}.json'
pattern = 'episodes-episode-\d+'
episodes = []
error_eps = []

existing_eps = [x.split('.')[0] for x in listdir('data/episodes')]

for fpath in listdir(root):
    with codecs.open(os.path.join(root,fpath),'r') as f:
        raw_html = f.read()
        episodes.extend([x.split('-')[-1] for x in re.findall(pattern,raw_html)]) 

episodes = [x for x in episodes if episodes not in existing_eps]
episodes = list(set(episodes))

for eps in episodes:
    try:
        with urllib.request.urlopen(url_root.format(eps)) as webpage:
            data = json.loads(webpage.read().decode())
        with open('data/episodes/{}.json'.format(eps),'w') as f:
            json.dump(data,f)
        print('done ', eps)
    except:
        print('error ', eps)
        error_eps.append(eps)


with open('data/{}.pkl'.format('error_eps'),'wb') as f:
    pickle.dump(error_eps,f)

print('work done.')