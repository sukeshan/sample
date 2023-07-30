from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import datetime

def update_csv(link ,driver ):
    driver.get(link)
    time.sleep(3)
    soup = BeautifulSoup(driver.page_source,'html.parser')

    try :title = str(soup.find_all('h1' ,class_ = 'jd-header-title')[0]).split('"')[3]
    except IndexError or Exception:
        try : title = str(soup.find_all('h1' ,class_ = 'av-special-heading-tag')[0]).split('>')[1].split('<')[0]
        except IndexError or Exception : title = ''
    
    try :skills = ', '.join([ re.sub(r'[^\w\s]', '', str(skill).split('span')[1]) for skill in soup.find_all('a' ,class_ = 'chip clickable')])
    except Exception :skills=''
    if len(skills) <1 :
        try :skills = ', '.join([ re.sub(r'[^\w\s]', '', str(skill).split('span')[1]) for skill in soup.find_all('a' ,class_ = 'chip non-clickable')])
        except Exception :skills=''

    try : experience = str(soup.find_all("div", class_="exp")[0]).split('<')[4].split('>')[1]
    except IndexError or Exception:
        try : experience = str(soup.find_all('div' ,class_ = 'slide-meta getExperience')[0]).split('</span>')[1]
        except IndexError or Exception: experience =''

    exp = [i for i in experience.split(' ') if i.isnumeric()]
    if len(exp) != 0 : min = exp[0]
    if len(exp) >= 2 : max =exp[1]

    try : job_date = str(soup.find_all("div", class_="jd-stats")[0]).split('Posted: </label><span>')[1].split('<')[0]
    except IndexError or Exception:
        try : job_date = str(soup.find_all("div", class_="sumFoot")[0]).split('<strong>')[1].split('<')[0]
        except IndexError or Exception: job_date =''

    current = datetime.datetime.now()
    day = current.day
    month = current.month
    year =current.year
    if 'days' in job_date:
        temp = int(job_date.split(' ')[0])
        day -= temp 
    if 'month' in job_date:
        temp = int(job_date.split(' ')[0])
        if temp == 'a':temp =1
        month -= temp
    if 'year' in job_date:
        temp = int(job_date.split(' ')[0])
        if temp == 'a':temp =1
        year -= temp
    if job_date.isnumeric():
        day-=int(job_date)

    job_date = f'{day}:{month}:{year}' 

    try : role = ' '.join(re.findall(r'target="_blank">(.*?)</a>', str(soup.find_all('div' ,class_ = 'details')[0])))
    except IndexError or Exception:
        try : role = str(soup.find_all('p' ,class_ = 'coPE getRoleLabel')[0]).split('itemprop="">')[1].split('<')[0]
        except IndexError or Exception: role =''

    try : industry = ' '.join(re.findall(r'target="_blank">(.*?)</a>', str(soup.find_all('div' ,class_ = 'details')[1])))
    except IndexError or Exception:
        try :industry = str(soup.find_all('p' ,class_ = 'coPE getIndustryLabel')).split('<span>')[1].split('<')[0]
        except IndexError or Exception: industry =''

    try : department = ' '.join(re.findall(r'target="_blank">(.*?)</a>', str(soup.find_all('div' ,class_ = 'details')[2])))
    except IndexError or Exception:
        try :department = str(soup.find_all('p' ,class_ = 'coPE getFareaLabel')[0]).split('<span>')[1].split('<')[0]
        except IndexError or Exception: department =''

    try : employment_type = str(soup.find_all("div", class_="other-details")[0]).split('Employment Type: </label><span><span>')[1].split('<')[0]
    except IndexError or Exception:
        try :employment_type = str(soup.find_all('p' ,class_ = 'coPE getEmploymentType')[0]).split('"">')[1].split('<')[0]
        except IndexError or Exception: employment_type =''
    
    try : description  = BeautifulSoup(str(soup.find_all('div' ,class_ = 'dang-inner-html')[0]), 'html.parser').get_text()
    except IndexError or Exception:
        try :description = str(soup.find_all('div' ,class_ = 'clearboth description')[0]).replace('<div class="clearboth description">','').replace('<br/><br/>','')
        except IndexError or Exception: description =''

    try : 
        html  = str(soup.find_all("div", class_="education")[0])
        ug_info = re.findall(r'<label>UG: </label><span class="">(.*?)</span>', html)
        pg_info = re.findall(r'<label>PG: </label><span class="">(.*?)</span>', html)
        doctoral_info = re.findall(r'<label>Doctorate: </label><span class="">(.*?)</span>', html)
        education = str([str(ug_info).replace('[','').replace(']','').replace("'",'') ,str(pg_info).replace('[','').replace(']','').replace("'",'') ,str(doctoral_info).replace('[','').replace(']','').replace("'",'')]).replace('[','').replace(']','').replace("'",'').replace(', ',' ')

    except IndexError or Exception : 
        try : education = str(soup.find_all("p", class_="coPE getUGCourse")[0]).split('itemprop="">')[1].split('<')[0] + ' | '+ str(soup.find_all("p", class_="coPE getPGCourse")[0]).split('itemprop="">')[1].split('<')[0]
        except IndexError or Exception : education = ' '

    try : company = str(soup.find_all('div' ,class_ = 'jd-header-comp-name')[0]).split('</a><a')[0].split('">')[-1]
    except IndexError or Exception:
        try :company = str(soup.find_all('div' ,class_ = 'f14 lh18 alignJ')[0]).split('"name">')[1].split('</')[0]
        except IndexError or Exception: company =''

       
    return {"role" : role ,"description" : description ,"skills" : skills.split(', ') }

def scrap( ):

    file = open('E:\Senseloaf\Z\page.txt' ,'r')
    page = int(file.read())
    role ,des ,skill = [] ,[] ,[]
    driver = webdriver.Chrome()

    driver.get(f'https://www.naukri.com/jobs-in-india-{page}?jobAge=30')   
    time.sleep(3)
    soup = BeautifulSoup(driver.page_source,'html.parser')
    soup1 = BeautifulSoup(str(soup.find_all("a", class_="title ellipsis")), 'html.parser')

    for link in [link['href'] for link in soup1.find_all('a')]:
        out = update_csv(link ,driver )  
        role.append(out[0])
        des.append(out[1])
        skill.append(ou[2])

    pd.concat(pd.read_csv('Existing_DataFrame') ,pd.DataFrame({'Role' : role ,'Description' : des ,'Skills' : skill}) )

    file = open('E:\Senseloaf\Z\page.txt' ,'w+')
    file.write(str(page+=1))
