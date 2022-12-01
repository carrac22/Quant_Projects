from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd

#defining the path of the selenium chrome driver
PATH = "/usr/local/bin/chromedriver"
driver= webdriver.Chrome(PATH)
#defining the url to be used
url = "https://www.google.com/maps/"
website = driver.get(url)
#grabbing the search function and posting keywords
search = driver.find_element_by_name("q")
search.send_keys("apartments in chapel hill, nc")
#allowing time for the page to load
time.sleep(3)
#sending the query
search.send_keys(Keys.RETURN)
time.sleep(15)

#defining the scroll functionality on the side bar
def infinite_scroll(driver):
    number_of_elements_found = 0
    while True:
        els = driver.find_elements(By.CLASS_NAME, 'y7PRA')
        if number_of_elements_found == len(els):
            # Reached the end of loadable elements
            break

        try:
            driver.execute_script("arguments[0].scrollIntoView();", els[-1])
            number_of_elements_found = len(els)
            time.sleep(5)

        except StaleElementReferenceException:
            # Possible to get a StaleElementReferenceException. Ignore it and retry.
            pass

infinite_scroll(driver=driver)
time.sleep(5)

#accessing the target text
main = driver.find_element_by_xpath("//*[@role='main']")
subset = main.find_elements_by_xpath("//*[@class='y7PRA']")

#combining the webelements into a dataframe
dataframe=pd.DataFrame(columns=["Name", "Rating","Number of Ratings","Type","Location" ])
for element in subset:
    element= str(element.text)
    split=element.split("\n")
    name = split[0]
    rating_comb= split[1]
    if "(" not in rating_comb:
        rating= "No reviews"
        num_ratings="0"
    else:
        rating=rating_comb.split("(")[0]
        num_ratings=rating_comb.split("(")[1][:-1]

    typ_comb=split[2]
    if "·" not in typ_comb:
        location= typ_comb
        typ="N/A"
    else:    
        typ=typ_comb.split("·")[0]
        location=typ_comb.split("·")[1]
    df= {"Name": name, "Rating": rating, "Number of Ratings":num_ratings,"Type":typ, "Location":location}
    dataframe = dataframe.append(df, ignore_index=True)

#printing for verification and converting to excel sheet
print(dataframe)
dataframe.to_csv("Apartments.csv", sep=',')
#quitting the program
driver.quit()
