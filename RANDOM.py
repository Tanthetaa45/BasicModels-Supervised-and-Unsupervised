import time

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

PATH = "chromedriver.exe"

product_name = input("Enter the product name: ")

service = Service(executable_path=PATH)
driver = webdriver.Chrome(service=service)

driver.get("https://www.bigbasket.com/")

search_bar_xpath = "/html/body/div[2]/div[1]/header[2]/div[1]/div[1]/div/div/div/div/input"
product_item_xpath = "/html/body/div[2]/div[1]/div[6]/div[2]/section[2]/section/ul/li[1]"
product_claim_xpath = "/html/body/div[2]/div[1]/div/div/section[2]/div[1]/div[2]/div"
product_ings_xpath = "/html/body/div[2]/div[1]/div/div/section[2]/div[2]/div[2]"
product_nut_facts_xpath = "/html/body/div[2]/div[1]/div/div/section[2]/div[3]"
product_details_class = "/html/body/div[2]/div[1]/div/div/section[2]"
plus_button_xpath = "/html/body/div[2]/div[1]/div/div/section[2]/div[3]/div[1]/div"


WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, search_bar_xpath)))
search_bar = driver.find_element(By.XPATH, search_bar_xpath)
search_bar.clear()
search_bar.send_keys(product_name + Keys.ENTER)

WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, product_item_xpath)))
product_item = driver.find_element(By.XPATH, product_item_xpath)
product_item.click()

driver.switch_to.window(driver.window_handles[-1])

# WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, product_details_class)))
# product_details = driver.find_element(By.XPATH, product_details_class).text
# print(product_details)

WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, product_claim_xpath)))
product_claim = driver.find_element(By.XPATH, product_claim_xpath).text
print(product_claim)
# WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, product_ings_xpath)))

product_ings = driver.find_element(By.XPATH, product_ings_xpath).text
print(product_ings)

plus = driver.find_element(By.XPATH, plus_button_xpath)
plus.click()

product_nut_facts = driver.find_element(By.XPATH, product_nut_facts_xpath).text
print(product_nut_facts)

time.sleep(10)

driver.quit()