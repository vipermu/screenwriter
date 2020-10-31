import os
from typing import *

import bs4
import requests
from bs4 import BeautifulSoup


def main():
    # NOTE: possible genres: Action	Adventure	Animation
    # Comedy	Crime	Drama Family	Fantasy	Film-Noir
    # Horror	Musical	Mystery Romance	Sci-Fi	Short
    # Thriller	War	Western
    output_path = "./extracted-data/script-data.txt"
    genre = "Horror"
    base_url = "https://www.imsdb.com/"
    genre_url = os.path.join(base_url, "genre", genre)

    title_list = get_titles_from_url(genre_url)
    title_list = ["-".join(title.split()) for title in title_list]
    script_url_list = [os.path.join(base_url, "scripts", f"{title}.html")
                       for title in title_list]

    for script_url in script_url_list:
        store_script_from_url(script_url, output_path)


def get_titles_from_url(url: str) -> List[str]:
    r = requests.get(url)           # Sends HTTP GET Request
    soup = BeautifulSoup(r.text, "html.parser")  # Parses HTTP Response
    title_list = [item.text for item in soup.select("td p a")]
    return title_list


def store_script_from_url(url: str, output_path: str) -> List[str]:
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    script_text = "\n".join([item.text for item in soup.select("pre")])

    with open(output_path, "a") as script_file:
        script_file.write(script_text)


if __name__ == "__main__":
    main()
