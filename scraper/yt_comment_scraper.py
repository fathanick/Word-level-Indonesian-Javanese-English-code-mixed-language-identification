import time
import pandas as pd
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def get_yt_comments(url):
    comments = []
    with Chrome() as driver:
        wait = WebDriverWait(driver,10)
        driver.get(url)

        for item in range(3): #by increasing the highest range you can get more content
            wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body"))).send_keys(Keys.END)
            time.sleep(3)

        for comment in wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#comment #content-text"))):
            print(comment.text)
            comments.append(comment.text)

    return comments

if __name__ == '__main__':
    yt_list_file = '../youtube/yt_url_list.txt'
    all_comments = []
    with open(yt_list_file, 'r') as f:
        for url in f:
            comments = get_yt_comments(url)
        all_comments.append(comments)

    df = pd.DataFrame(all_comments)
    df.to_excel('../raw dataset/yt-comments-040422.xlsx')