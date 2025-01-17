import pandas as pd
import time
from time import sleep
import concurrent.futures
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

start_time = time.time()

def google_translate(input_text:str, output_text:str)-> str:
    return output_text


def scrape_target(question:str) -> str:
    driver.maximize_window()
    text_box = driver.find_element(By.ID, id_gpt_input)
    text_box.send_keys(question)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, xpath_button)), message="Could not find element/button")
    submit_button = driver.find_element(By.XPATH, xpath_button)
    submit_button.click()
    sleep(30)
    try:
        scroll_button = driver.find_element(By.XPATH, "//button[contains(@class,'cursor-pointer')]")
        scroll_button.click()
    except:
        print("Already scrolled")
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, xpath_reply_box)))
    response = driver.find_element(By.XPATH, xpath_reply_box).text
    return response


# variables
url = "https://chat.openai.com/"
id_gpt_input = "prompt-textarea"
xpath_button = "//button[@data-testid='send-button']"
xpath_reply = ".//div[@class='markdown prose w-full break-words dark:prose-invert dark']/parent::*"
xpath_scroll = "//div[@class='w-full text-token-text-primary']"
num = 3
csv_loc = "Data_complete.csv"
sucess = 0 
errors = 0

driver = webdriver.Firefox()
driver.get(url)
data_df = pd.read_csv(csv_loc)
question_df = data_df['Description']
data = []

for question_inp in question_df[675:-1]:
    try:
        xpath_reply_box = f"//div[@data-testid='conversation-turn-{num}']"
        reply_ans = scrape_target(question_inp + "\nI have these datapoints for an image\nGive me some question and answers")
        print(reply_ans)
        data.append({'Description': question_inp, 'Answer': reply_ans})
        num = num+2
        sucess = sucess + 1
        qa_data_df = pd.DataFrame(data)
        data_final_df = pd.merge(data_df, qa_data_df, on='Description', how='inner')
        data_final_df.to_csv(f"Q&A_epoch_1.csv")
    except:
        print("___________________________Error in browser______________________________")
        qa_data_df = pd.DataFrame(data)
        data_final_df = pd.merge(data_df, qa_data_df, on='Description', how='inner')
        data_final_df.to_csv(f"Q&A_{sucess}_error.csv")
        driver.quit()
        num = 3
        driver = webdriver.Firefox()
        driver.get(url)
        xpath_reply_box = f"//div[@data-testid='conversation-turn-{num}']"
        reply_ans = scrape_target(question_inp + "\nI have these datapoints for an image\nGive me some question and answers")
        print(reply_ans)
        data.append({'Description': question_inp, 'Answer': reply_ans})
        num = num+2
        errors = errors + 1

qa_data_df = pd.DataFrame(data)
data_final_df = pd.merge(data_df, qa_data_df, on='Description', how='inner')
data_final_df.to_csv("Q&A_final.csv")

print(f" Error : {errors} \nSuccess : {sucess}")

end_time = time.time()

print(f"Start time : {start_time} \nEnd time : {end_time}\nTotal time in mins = {(end_time-start_time)/60}")

print("Scraping finished. Take care")
driver.quit()


