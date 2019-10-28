import os
import re

dialog_mark_1 = re.compile('[「](.*)[」]')
dialog_mark_2 = re.compile('[{}…「」]')
dialog_mark_3 = re.compile('&.*;')
dialog_mark_4 = re.compile('\n')
dialog_mark_5 = re.compile('[0-9a-zA-Z]')

def text_extract(originFile,questionFile,answerFile):
    with open(originFile,'r',encoding = 'utf-8') as file:
        lines  = file.readlines()
        dialogs = []
        for i in lines:
            x = dialog_mark_1.search(i.strip())
            if x is not None :
                x = x.group(1)
                x = re.sub(dialog_mark_2,'',x)
                x = re.sub(dialog_mark_3,'',x)
                if x != '\n' and x != ' ' and x != '':
                    dialogs.append(x)
        dialog_q = [dialogs[i] for i in range(1,len(dialogs) - 1)]
        dialog_a = [dialogs[i+1] for i in range(1,len(dialogs) - 1)]
        x = list(zip(dialog_q,dialog_a))
    f1 = open(questionFile,'a',encoding = 'utf-8')
    f2 = open(answerFile,'a',encoding = 'utf-8')
    for dia in x:
        f1.write(dia[0] + '\n')
        f2.write(dia[1] + '\n')
    f1.close()
    f2.close()

base_path = os.path.abspath('./novels')
for novel in os.listdir('novels'):
    paths = os.path.join(base_path,novel)
    text_extract(paths,'question.txt','answer.txt')
    a = 1



    


    


