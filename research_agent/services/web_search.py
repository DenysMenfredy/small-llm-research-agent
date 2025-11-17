
import requests
from bs4 import BeautifulSoup

def web_search(step: str):
    query = step.replace("Search for", "").strip()

    url = f"https://duckduckgo.com/html/?q={query}"
    html = requests.get(url).text

    soup = BeautifulSoup(html, "html.parser")
    results = [a.get_text() for a in soup.select(".result__title")][:5]

    return f"Search results for '{query}':\n" + "\n".join(results)
