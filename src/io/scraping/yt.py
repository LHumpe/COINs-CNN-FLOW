from requests_html import HTMLSession
from bs4 import BeautifulSoup as bs
import pandas as pd
from tqdm import tqdm

df = pd.read_csv('results/yt_analysis/video_stats.csv', sep=';').drop(columns=['favoriteCount'])

# init session
session = HTMLSession()

data = list()

for index, row in tqdm(df.iterrows()):
    if row['id'] == 8:
        continue
    elif row['id'] == 53:
        row['url'] = row['url'].split(',')[0]

    new_row = row.to_dict()

    # download HTML code
    response = session.get(row['url'])
    # execute Javascript
    response.html.render(sleep=4)
    # create beautiful soup object to parse HTML
    soup = bs(response.html.html, "html.parser")

    # date published
    new_row["date_published"] = soup.find("div", {"id": "date"}).text[1:]

    # get the duration of the video
    new_row["duration"] = soup.find("span", {"class": "ytp-time-duration"}).text

    # number of subscribers as str
    new_row['channel_subscribers'] = soup.find("yt-formatted-string", {"id": "owner-sub-count"}).text.strip()

    data.append(new_row)

pd.DataFrame(data).to_csv('results/yt_analysis/video_stats_enhanced.csv', index=False)
