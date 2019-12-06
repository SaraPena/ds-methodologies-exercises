from requests import get
from bs4 import BeautifulSoup
from os import path
import re
import pandas as pd
import json

import warnings

def create_blog_urls():
    if path.exists('blog_urls.csv'):
        blog_urls = pd.read_csv('blog_urls.csv')
        return blog_urls


    url = 'https://codeup.com/blog/'
    headers = {'User-Agent': 'Codeup Bayes Data Science'}
    response = get(url, headers = headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    first_page = int(soup.select('.mk-pagination-inner')[0].select_one('a')['data-page-id'])
    pages = [x for x in range(first_page, len(soup.select('.mk-pagination-inner')[0].select('a'))+1)]
    codeup_blog_pages = [url+str(x) for x in pages]

    blog_urls = []
    for x in codeup_blog_pages:
        response = get(x,headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        page_articles = soup.select('#loop-8')[0].select('article')
        for x in range(len(page_articles)):
            blog_urls.append(page_articles[x].select_one('a')['href'])
    blog_urls = pd.DataFrame(blog_urls)
    blog_urls.columns = ['urls']


    blog_urls.to_csv('blog_urls.csv', index = False)
    return blog_urls

def make_blog_dictionary():
    # if path.exists('blog_dictionary.json'):
    #     with open('blog_dictionary.json', 'r') as f:
    #         articles = json.load(f)
    #     return articles

    blog_urls = create_blog_urls()
    headers = {'User-Agent': 'Codeup Bayes Data Science Student'}
    articles = []
    for url in blog_urls.urls:
        #url = 'https://codeup.com/a-quest-through-codeup/'
        blog = {}
        response = get(url=url, headers=headers)
        soup = BeautifulSoup(response.content,'html.parser')
        blog_article = []
        for x in range(len(soup.select('.mk-single-content')[0].select('p'))):
            blog_article.append(soup.select('.mk-single-content')[0].select('p')[x].get_text())
        title = soup.find('h1').get_text()
        blog['title'] = title
        blog['body'] = " ".join(blog_article)
        articles.append(blog)
    
    # with open('blog_dictionary.json', 'w') as f:
    #     json.dump(articles, f)

    return articles



def get_blog_articles():
    if path.exists('codeup_blogs.csv'):
        codeup_blogs = pd.read_csv('codeup_blogs.csv')
        return codeup_blogs


    articles = make_blog_dictionary()
  
   
    titles = []
    content = []
    for x in articles:
        titles.append(x['title'])
        content.append(x['body'])
    
    codeup_blogs = pd.DataFrame({'titles': titles, 'content': content})
    
    codeup_blogs.to_csv('codeup_blogs.csv', index = False)

    codeup_blogs = pd.read_csv('codeup_blogs.csv')

    return codeup_blogs


# create_blog_urls()

# articles = make_blog_dictionary()
# len(articles)
# print(articles)


# df = get_blog_articles()
# df.content[0]