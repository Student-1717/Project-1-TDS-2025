import csv
import time
from datetime import datetime
from playwright.sync_api import sync_playwright
import requests
from bs4 import BeautifulSoup

# ----------- Course content scraper -----------

def scrape_lectures_and_save_csv():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        dashboard_url = "https://tds.s-anand.net/#/2025-01"
        print(f"Opening dashboard: {dashboard_url}")
        page.goto(dashboard_url, wait_until="networkidle")
        page.wait_for_timeout(5000)  # wait for JS content

        links = page.query_selector_all("a")

        lecture_links = []
        seen_urls = set()

        for link in links:
            href = link.get_attribute("href")
            text = link.inner_text().strip()
            if href and text:
                if href.startswith('#'):
                    continue
                if any(k in href.lower() for k in ['lecture', 'unit', 'topic', 'lesson']) or href.startswith('/'):
                    full_url = href if href.startswith('http') else "https://tds.s-anand.net" + href
                    if full_url not in seen_urls:
                        lecture_links.append((text, full_url))
                        seen_urls.add(full_url)

        print(f"Found {len(lecture_links)} potential lecture links.")

        with open("tds_lectures_content.csv", mode="w", encoding="utf-8", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Lecture Title", "Content"])

            for title, url in lecture_links:
                print(f"Scraping lecture: {title} -> {url}")
                try:
                    page.goto(url, wait_until="networkidle")
                    page.wait_for_timeout(4000)

                    content_element = (
                        page.query_selector("article") or
                        page.query_selector("main") or
                        page.query_selector("div.course-content") or
                        page.query_selector("section.lecture-body")
                    )

                    content_text = content_element.inner_text().strip() if content_element else ""
                    print(f"Content length: {len(content_text)} characters")

                    writer.writerow([title, content_text])

                    time.sleep(1)  # polite delay

                except Exception as e:
                    print(f"Failed to scrape {title}: {e}")

        browser.close()
        print("Course content scraping complete! Data saved to tds_lectures_content.csv")


# ----------- Discourse posts scraper -----------

BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
CATEGORY_API = f"{BASE_URL}/c/courses/tds-kb/34.json"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"  # e.g. 2025-04-15T12:34:56.789Z

START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 4, 14, 23, 59, 59)

def fetch_topics():
    topics = []
    page = 0
    while True:
        url = f"{CATEGORY_API}?page={page}"
        print(f"Fetching topics page {page}: {url}")
        resp = requests.get(url)
        data = resp.json()

        current_topics = data.get("topic_list", {}).get("topics", [])
        if not current_topics:
            break

        for topic in current_topics:
            created_at = datetime.strptime(topic["created_at"], DATE_FORMAT)
            if created_at < START_DATE:
                # Older than start date — stop fetching more pages
                return topics
            if START_DATE <= created_at <= END_DATE:
                topics.append(topic)

        page += 1
        time.sleep(1)  # polite delay

    return topics

def fetch_posts_for_topic(topic_id):
    url = f"{BASE_URL}/t/{topic_id}.json"
    resp = requests.get(url)
    data = resp.json()
    posts = data.get("post_stream", {}).get("posts", [])
    return posts

def scrape_discourse_posts():
    topics = fetch_topics()
    print(f"Found {len(topics)} topics in date range.")

    with open("tds_discourse_posts.csv", "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Topic ID", "Topic Title", "Post ID", "Author", "Created At", "Content"])

        for topic in topics:
            topic_id = topic["id"]
            topic_title = topic["title"]
            print(f"Fetching posts for topic {topic_id}: {topic_title}")

            posts = fetch_posts_for_topic(topic_id)

            for post in posts:
                post_id = post["id"]
                author = post["username"]
                created_at = post["created_at"]
                content_html = post["cooked"]

                soup = BeautifulSoup(content_html, "html.parser")
                text_content = soup.get_text(separator="\n").strip()

                writer.writerow([topic_id, topic_title, post_id, author, created_at, text_content])

            time.sleep(1)  # polite delay

    print("Discourse posts scraping complete! Data saved to tds_discourse_posts.csv")


# ----------- Main entry point -----------

if __name__ == "__main__":
    print("Starting scraping of course content...")
    scrape_lectures_and_save_csv()
    print("\nStarting scraping of discourse posts...")
    scrape_discourse_posts()
    print("\nAll scraping complete!")
